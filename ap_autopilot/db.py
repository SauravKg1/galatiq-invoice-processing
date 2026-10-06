"""SQLite persistence: mock inventory (required by the case), plus the records a
real AP system needs: vendor master, invoice ledger, payments, the human review
queue, and an append-only audit log.

One file (`inventory.db`) so every agent sees the same state, as the brief asks.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .normalize import normalize_sku, normalize_vendor

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    item        TEXT PRIMARY KEY,
    stock       INTEGER NOT NULL,
    unit_price  REAL,
    category    TEXT
);
CREATE TABLE IF NOT EXISTS vendors (
    name        TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'approved',
    currency    TEXT NOT NULL DEFAULT 'USD',
    address     TEXT,
    notes       TEXT,
    email       TEXT,                               -- AP contact on file; the ONLY address we ever reply to
    po_required INTEGER NOT NULL DEFAULT 0          -- 1 = invoices without a purchase order are held
);
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number   TEXT PRIMARY KEY,
    vendor      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',       -- open | closed
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS po_lines (
    po_number    TEXT NOT NULL,
    line_no      INTEGER NOT NULL,
    item         TEXT NOT NULL,
    qty_ordered  REAL NOT NULL,
    unit_price   REAL NOT NULL,
    qty_invoiced REAL NOT NULL DEFAULT 0,           -- grows as invoices against this line are paid (GR/IR idea)
    PRIMARY KEY (po_number, line_no)
);
CREATE TABLE IF NOT EXISTS goods_receipts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    gr_number    TEXT NOT NULL,
    po_number    TEXT NOT NULL,
    item         TEXT NOT NULL,
    qty_received REAL NOT NULL,
    received_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    invoice_key   TEXT,
    invoice_number TEXT,
    vendor        TEXT,
    invoice_date  TEXT,
    total         REAL,
    currency      TEXT,
    content_hash  TEXT,
    source_file   TEXT,
    status        TEXT NOT NULL,
    risk_score    INTEGER,
    decided_by    TEXT,
    rationale     TEXT,
    report_json   TEXT,
    processed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_invoices_key ON invoices(invoice_key);
CREATE TABLE IF NOT EXISTS payments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_key  TEXT NOT NULL UNIQUE,          -- idempotency: one payment per bill, ever
    vendor       TEXT NOT NULL,
    amount       REAL NOT NULL,
    currency     TEXT NOT NULL,
    reference    TEXT,
    status       TEXT NOT NULL,
    paid_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approval_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_key  TEXT,
    run_id       TEXT NOT NULL,
    source_file  TEXT,
    vendor       TEXT,
    amount       REAL,
    currency     TEXT,
    reason       TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    reviewer     TEXT,
    review_note  TEXT,
    created_at   TEXT NOT NULL,
    decided_at   TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    invoice_key  TEXT,
    source_file  TEXT,
    audience     TEXT NOT NULL,                     -- vendor | internal
    kind         TEXT NOT NULL,
    to_addr      TEXT NOT NULL,
    subject      TEXT NOT NULL,
    body         TEXT NOT NULL,
    author       TEXT NOT NULL,                     -- llm | template
    status       TEXT NOT NULL DEFAULT 'draft',     -- draft | sent | discarded (nothing is ever auto-sent)
    note         TEXT,
    created_at   TEXT NOT NULL,
    sent_at      TEXT,
    sent_by      TEXT
);
CREATE TABLE IF NOT EXISTS historical_invoices (   -- SYNTHETIC past invoices (see history.py)
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor         TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    invoice_date   TEXT NOT NULL,
    total          REAL NOT NULL,
    lines_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_vendor ON historical_invoices(vendor);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    run_id      TEXT,
    invoice_key TEXT,
    stage       TEXT NOT NULL,
    event       TEXT NOT NULL,
    payload     TEXT
);
"""

# Minimum seed from the case brief, extended with catalog prices so we can
# also catch price variance (overbilling is a top AP leakage source).
SEED_INVENTORY = [
    ("WidgetA", 15, 250.00, "widgets"),
    ("WidgetB", 10, 500.00, "widgets"),
    ("GadgetX", 5, 750.00, "gadgets"),
    ("FakeItem", 0, None, "unknown"),
]

# Purchasing data for the three-way match. The brief's samples carry no PO
# numbers, so only INV-1012's reference (PO-20260115, issued to FastShip) and the
# new 2012-2016 scenarios have POs. (po_number, vendor, status, created, lines, receipts)
SEED_POS = [
    ("PO-20260115", "FastShip Ltd.", "open", "2026-01-15",
     [("WidgetA", 12, 250.0), ("WidgetB", 7, 500.0), ("GadgetX", 4, 750.0)],
     [("GR-5001", "WidgetA", 12, "2026-01-24"), ("GR-5001", "WidgetB", 7, "2026-01-24"), ("GR-5001", "GadgetX", 4, "2026-01-24")]),
    ("PO-4500012", "Widgets Inc.", "open", "2026-02-10",
     [("WidgetA", 10, 250.0)],
     [("GR-5012", "WidgetA", 6, "2026-02-20")]),                       # only 6 of 10 delivered so far
    ("PO-4500013", "Summit Manufacturing Co.", "open", "2026-02-12",
     [("WidgetB", 4, 480.0)],                                          # negotiated below the $500 list price
     [("GR-5013", "WidgetB", 4, "2026-02-22")]),
    ("PO-4500015", "Reliable Components Inc.", "open", "2026-02-25",
     [("GadgetX", 3, 750.0), ("WidgetA", 4, 250.0)],
     [("GR-5015", "GadgetX", 3, "2026-03-02"), ("GR-5015", "WidgetA", 4, "2026-03-02")]),
]
VENDORS_REQUIRING_PO = {"Atlas Industrial Supply", "MegaWidgets Corp"}

# Vendor master: who Acme has actually onboarded. Anything else is a new,
# unverified payee. Assumption documented in the README.
SEED_VENDORS = [
    ("Widgets Inc.", "approved", "USD", "100 Main St, Chicago, IL 60601", None, "ar@widgetsinc.example"),
    ("Gadgets Co.", "approved", "USD", None, None, "billing@gadgetsco.example"),
    ("Precision Parts Ltd.", "approved", "USD", "742 Evergreen Terrace, Springfield, IL 62704", None, "accounts@precisionparts.example"),
    ("Acme Industrial Supplies", "approved", "USD", None, None, "ar@acmeindustrial.example"),
    ("MegaWidgets Corp", "approved", "USD", None, None, "receivables@megawidgets.example"),
    ("Consolidated Materials Group", "approved", "USD", None, None, "ar@consolidatedmaterials.example"),
    ("Summit Manufacturing Co.", "approved", "USD", None, None, "accounts@summitmfg.example"),
    ("FastShip Ltd.", "approved", "USD", None, "Bank details verified 2025-11", "billing@fastship.example"),
    ("Atlas Industrial Supply", "approved", "USD", "500 Commerce Blvd, Detroit, MI 48201", None, "ar@atlasindustrial.example"),
    ("TechParts International", "approved", "EUR", None, "EU supplier, invoices in EUR", "invoices@techparts.example"),
    ("Reliable Components Inc.", "approved", "USD", None, None, "ar@reliablecomponents.example"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ setup
    def init(self, reset: bool = False) -> None:
        if reset and Path(self.path).exists():
            Path(self.path).unlink()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0] == 0:
                conn.executemany("INSERT INTO inventory VALUES (?,?,?,?)", SEED_INVENTORY)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(vendors)")}
            if "email" not in columns:  # migrate databases created before vendor contacts existed
                conn.execute("ALTER TABLE vendors ADD COLUMN email TEXT")
                conn.executemany("UPDATE vendors SET email = ? WHERE name = ?", [(v[5], v[0]) for v in SEED_VENDORS])
            if "po_required" not in {r[1] for r in conn.execute("PRAGMA table_info(vendors)")}:
                conn.execute("ALTER TABLE vendors ADD COLUMN po_required INTEGER NOT NULL DEFAULT 0")
                conn.executemany("UPDATE vendors SET po_required = 1 WHERE name = ?", [(v,) for v in VENDORS_REQUIRING_PO])
            if conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0] == 0:
                conn.executemany("INSERT INTO vendors (name, status, currency, address, notes, email, po_required) "
                                 "VALUES (?,?,?,?,?,?,?)",
                                 [(*v, int(v[0] in VENDORS_REQUIRING_PO)) for v in SEED_VENDORS])
            if conn.execute("SELECT COUNT(*) FROM historical_invoices").fetchone()[0] == 0:
                from .history import generate
                conn.executemany("INSERT INTO historical_invoices (vendor, invoice_number, invoice_date, total, lines_json) "
                                 "VALUES (?,?,?,?,?)", generate())
            if conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0] == 0:
                for po, vendor, status, created, lines, receipts in SEED_POS:
                    conn.execute("INSERT INTO purchase_orders VALUES (?,?,?,?)", (po, vendor, status, created))
                    conn.executemany("INSERT INTO po_lines (po_number, line_no, item, qty_ordered, unit_price) VALUES (?,?,?,?,?)",
                                     [(po, i + 1, item, qty, price) for i, (item, qty, price) in enumerate(lines)])
                    conn.executemany("INSERT INTO goods_receipts (gr_number, po_number, item, qty_received, received_at) "
                                     "VALUES (?,?,?,?,?)", [(gr, po, item, qty, at) for gr, item, qty, at in receipts])

    def ensure(self) -> "Database":
        self.init(reset=False)
        return self

    # -------------------------------------------------------------- inventory
    def catalog(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM inventory ORDER BY item")]

    def find_item(self, name: Optional[str]) -> Optional[dict[str, Any]]:
        """Exact match after formatting normalization only ('Widget A' == 'WidgetA')."""
        key = normalize_sku(name)
        if not key:
            return None
        for row in self.catalog():
            if normalize_sku(row["item"]) == key:
                return row
        return None

    # ---------------------------------------------------------------- vendors
    def vendors(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM vendors ORDER BY name")]

    def find_vendor(self, name: Optional[str]) -> Optional[dict[str, Any]]:
        key = normalize_vendor(name)
        if not key:
            return None
        for row in self.vendors():
            if normalize_vendor(row["name"]) == key:
                return row
        return None

    # ---------------------------------------------------------------- history
    def vendor_history(self, vendor_name: Optional[str]) -> list[dict[str, Any]]:
        """Past paid invoices for this vendor (synthetic seed history)."""
        key = normalize_vendor(vendor_name)
        if not key:
            return []
        with self.connect() as conn:
            rows = conn.execute("SELECT vendor, invoice_number, invoice_date, total, lines_json FROM historical_invoices "
                                "ORDER BY invoice_date")
            return [{**{k: r[k] for k in ("vendor", "invoice_number", "invoice_date", "total")},
                     "lines": json.loads(r["lines_json"])} for r in rows if normalize_vendor(r["vendor"]) == key]

    def history_by_vendor(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        with self.connect() as conn:
            for r in conn.execute("SELECT vendor, invoice_number, invoice_date, total, lines_json FROM historical_invoices "
                                  "ORDER BY invoice_date"):
                out.setdefault(r["vendor"], []).append({"invoice_number": r["invoice_number"], "invoice_date": r["invoice_date"],
                                                        "total": r["total"], "lines": json.loads(r["lines_json"])})
        return out

    # -------------------------------------------------------- purchase orders
    def get_po(self, po_number: Optional[str]) -> Optional[dict[str, Any]]:
        if not po_number:
            return None
        with self.connect() as conn:
            head = conn.execute("SELECT * FROM purchase_orders WHERE po_number = ?", (po_number,)).fetchone()
            if not head:
                return None
            lines = [dict(r) for r in conn.execute("SELECT * FROM po_lines WHERE po_number = ? ORDER BY line_no", (po_number,))]
            receipts = [dict(r) for r in conn.execute("SELECT * FROM goods_receipts WHERE po_number = ? ORDER BY id", (po_number,))]
        return {**dict(head), "lines": lines, "receipts": receipts}

    def open_pos_for_vendor(self, vendor_name: Optional[str]) -> list[dict[str, Any]]:
        key = normalize_vendor(vendor_name)
        with self.connect() as conn:
            rows = conn.execute("SELECT po_number, vendor FROM purchase_orders WHERE status = 'open' ORDER BY created_at DESC")
            numbers = [r["po_number"] for r in rows if normalize_vendor(r["vendor"]) == key]
        return [self.get_po(n) for n in numbers]

    def add_po_invoiced(self, po_number: str, item: str, qty: float) -> None:
        """Record a paid quantity against the PO. Fills matching lines in order; anything beyond
        the ordered quantity lands on the last matching line, so over-billing stays visible.
        Closes the PO once every line is fully invoiced."""
        with self.connect() as conn:
            lines = [r for r in conn.execute("SELECT line_no, item, qty_ordered, qty_invoiced FROM po_lines "
                                             "WHERE po_number = ? ORDER BY line_no", (po_number,))
                     if normalize_sku(r["item"]) == normalize_sku(item)]
            remaining = qty
            for i, ln in enumerate(lines):
                last = i == len(lines) - 1
                take = remaining if last else min(remaining, max(0.0, ln["qty_ordered"] - ln["qty_invoiced"]))
                if take > 0:
                    conn.execute("UPDATE po_lines SET qty_invoiced = qty_invoiced + ? WHERE po_number = ? AND line_no = ?",
                                 (take, po_number, ln["line_no"]))
                    remaining -= take
            totals = conn.execute("SELECT qty_ordered, qty_invoiced FROM po_lines WHERE po_number = ?", (po_number,)).fetchall()
            if totals and all(r["qty_invoiced"] >= r["qty_ordered"] for r in totals):
                conn.execute("UPDATE purchase_orders SET status = 'closed' WHERE po_number = ?", (po_number,))

    # --------------------------------------------------------------- invoices
    def invoice_history(self, key: Optional[str]) -> list[dict[str, Any]]:
        if not key:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, run_id, invoice_number, vendor, total, currency, content_hash, source_file,"
                " status, decided_by, rationale, processed_at FROM invoices WHERE invoice_key = ? ORDER BY id",
                (key,),
            )
            return [dict(r) for r in rows]

    def invoices_for_vendor(self, vendor_name: Optional[str]) -> list[dict[str, Any]]:
        key = normalize_vendor(vendor_name)
        if not key:
            return []
        with self.connect() as conn:
            rows = conn.execute("SELECT invoice_key, invoice_number, vendor, invoice_date, total, status, source_file"
                                " FROM invoices ORDER BY id")
            return [dict(r) for r in rows if normalize_vendor(r["vendor"]) == key]

    def record_invoice(self, **fields: Any) -> int:
        fields.setdefault("processed_at", _now())
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.connect() as conn:
            cur = conn.execute(f"INSERT INTO invoices ({cols}) VALUES ({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def update_invoice_status(self, key: str, run_id: str, status: str, decided_by: str, rationale: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE invoices SET status = ?, decided_by = ?, rationale = ? WHERE invoice_key = ? AND run_id = ?",
                (status, decided_by, rationale, key, run_id),
            )

    def recent_invoices(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM invoices ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    # --------------------------------------------------------------- payments
    def payment_for(self, key: Optional[str]) -> Optional[dict[str, Any]]:
        if not key:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM payments WHERE invoice_key = ?", (key,)).fetchone()
            return dict(row) if row else None

    def record_payment(self, key: str, vendor: str, amount: float, currency: str, reference: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO payments (invoice_key, vendor, amount, currency, reference, status, paid_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (key, vendor, amount, currency, reference, status, _now()),
            )

    def payments(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM payments ORDER BY id")]

    # ------------------------------------------------------------ human queue
    def enqueue_review(self, **fields: Any) -> int:
        fields.setdefault("created_at", _now())
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.connect() as conn:
            cur = conn.execute(f"INSERT INTO approval_queue ({cols}) VALUES ({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def review_queue(self, status: Optional[str] = "pending") -> list[dict[str, Any]]:
        with self.connect() as conn:
            if status:
                rows = conn.execute("SELECT * FROM approval_queue WHERE status = ? ORDER BY id", (status,))
            else:
                rows = conn.execute("SELECT * FROM approval_queue ORDER BY id")
            return [dict(r) for r in rows]

    def get_review(self, review_id: int) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM approval_queue WHERE id = ?", (review_id,)).fetchone()
            return dict(row) if row else None

    def resolve_review(self, review_id: int, status: str, reviewer: str, note: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE approval_queue SET status = ?, reviewer = ?, review_note = ?, decided_at = ? WHERE id = ?",
                (status, reviewer, note, _now(), review_id),
            )

    # ----------------------------------------------------------------- outbox
    def add_draft(self, **fields: Any) -> int:
        fields.setdefault("created_at", _now())
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.connect() as conn:
            cur = conn.execute(f"INSERT INTO outbox ({cols}) VALUES ({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def outbox(self, status: Optional[str] = "draft", run_id: Optional[str] = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM outbox WHERE 1=1", []
        if status:
            query, args = query + " AND status = ?", args + [status]
        if run_id:
            query, args = query + " AND run_id = ?", args + [run_id]
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query + " ORDER BY id", args)]

    def get_draft(self, draft_id: int) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM outbox WHERE id = ?", (draft_id,)).fetchone()
            return dict(row) if row else None

    def update_draft(self, draft_id: int, subject: str, body: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE outbox SET subject = ?, body = ? WHERE id = ? AND status = 'draft'", (subject, body, draft_id))

    def set_draft_status(self, draft_id: int, status: str, by: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE outbox SET status = ?, sent_at = ?, sent_by = ? WHERE id = ? AND status = 'draft'",
                         (status, _now(), by, draft_id))

    # ------------------------------------------------------------------ audit
    def audit(self, run_id: Optional[str], key: Optional[str], stage: str, event: str, payload: Any = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, run_id, invoice_key, stage, event, payload) VALUES (?,?,?,?,?,?)",
                (_now(), run_id, key, stage, event, json.dumps(payload, default=str) if payload is not None else None),
            )

    def audit_trail(self, run_id: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if run_id:
                rows = conn.execute("SELECT * FROM audit_log WHERE run_id = ? ORDER BY id", (run_id,))
            else:
                rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]
