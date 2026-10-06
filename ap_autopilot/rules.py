"""Deterministic validation rules.

Each rule is a pure function returning Findings. Rules encode what finance
already knows to be true (math, stock, vendor master, duplicates). They run
before any LLM, are authoritative, and cannot be overridden by a model.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from datetime import timedelta
from typing import Any, Optional

from .config import Settings
from .db import Database
from .models import Finding, Invoice, Severity
from .normalize import find_po_reference, normalize_po, normalize_sku, normalize_vendor, parse_date
from .security import scan_for_injection

C, W, I = Severity.CRITICAL, Severity.WARNING, Severity.INFO

_URGENCY = re.compile(r"\b(urgent|immediately|asap|wire\s+transfer|avoid\s+penalt\w*|final\s+notice|act\s+now|today\s+only)\b", re.I)
_PREMIUM_REASON = re.compile(r"\b(rush|expedit\w*|express|overnight|priority)\b", re.I)
_BANK_CHANGE = re.compile(r"\b(new|updated|changed?|different)\b.{0,40}\b(bank|account|routing|iban|swift|remittance)\b|\bbank\s+details\s+have\s+changed\b", re.I)
_FORMERLY = re.compile(r"\b(?:formerly|previously|f/k/a|fka)\s+([A-Za-z0-9 .,&'-]+?)(?:\)|$|\.\s)", re.I)


def _money(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def check_required_fields(inv: Invoice) -> list[Finding]:
    out: list[Finding] = []
    if not (inv.vendor_name or "").strip():
        out.append(Finding(code="MISSING_VENDOR", severity=C, message="No vendor name: there is no one we could legitimately pay."))
    if not inv.invoice_number:
        out.append(Finding(code="MISSING_INVOICE_NUMBER", severity=W, message="No invoice number, so duplicate payments cannot be ruled out."))
    if inv.total is None:
        out.append(Finding(code="MISSING_TOTAL", severity=C, message="No total amount could be read from the document."))
    if not inv.line_items:
        out.append(Finding(code="NO_LINE_ITEMS", severity=C, message="No line items: nothing to match against inventory."))
    if inv.due_date is None:
        if inv.due_date_raw and inv.due_date_raw.strip():
            out.append(Finding(code="UNPARSEABLE_DUE_DATE", severity=W,
                               message=f"Due date '{inv.due_date_raw}' is not a real date. Relative dates are a pressure tactic.",
                               evidence={"raw": inv.due_date_raw}))
        else:
            out.append(Finding(code="MISSING_DUE_DATE", severity=W, message="No due date on the invoice."))
    if inv.invoice_date is None:
        out.append(Finding(code="MISSING_INVOICE_DATE", severity=I, message="No invoice date found."))
    return out


def check_line_integrity(inv: Invoice, tol: float) -> list[Finding]:
    out: list[Finding] = []
    for li in inv.line_items:
        name = li.item or li.description
        if li.quantity is None:
            out.append(Finding(code="MISSING_QUANTITY", severity=W, item=name, message=f"{name}: quantity missing."))
        elif li.quantity < 0:
            out.append(Finding(code="NEGATIVE_QUANTITY", severity=C, item=name,
                               message=f"{name}: negative quantity ({li.quantity:g}). Credits must come as a credit memo, not inside an invoice.",
                               evidence={"quantity": li.quantity}))
        elif li.quantity == 0:
            out.append(Finding(code="ZERO_QUANTITY", severity=W, item=name, message=f"{name}: quantity is zero."))
        if li.unit_price is None:
            out.append(Finding(code="MISSING_UNIT_PRICE", severity=W, item=name, message=f"{name}: unit price missing."))
        elif li.unit_price < 0:
            out.append(Finding(code="NEGATIVE_UNIT_PRICE", severity=C, item=name, message=f"{name}: negative unit price."))
        if li.amount is not None and li.quantity is not None and li.unit_price is not None:
            expected = li.quantity * li.unit_price
            if not _close(expected, li.amount, tol):
                out.append(Finding(code="LINE_AMOUNT_MISMATCH", severity=W, item=name,
                                   message=f"{name}: {li.quantity:g} x {_money(li.unit_price)} = {_money(expected)}, invoice says {_money(li.amount)}.",
                                   evidence={"expected": expected, "stated": li.amount}))
    return out


def check_totals(inv: Invoice, tol: float) -> list[Finding]:
    out: list[Finding] = []
    if inv.total is not None and inv.total <= 0:
        out.append(Finding(code="NON_POSITIVE_TOTAL", severity=C, message=f"Total is {_money(inv.total)}. An invoice must be a positive amount owed."))
    priced = [li for li in inv.line_items if li.quantity is not None and li.unit_price is not None]
    if not priced:
        return out
    lines_sum = round(sum((li.amount if li.amount is not None else li.quantity * li.unit_price) for li in priced), 2)
    if inv.subtotal is not None and not _close(lines_sum, inv.subtotal, tol):
        out.append(Finding(code="SUBTOTAL_MISMATCH", severity=W,
                           message=f"Line items add up to {_money(lines_sum)} but the subtotal says {_money(inv.subtotal)}.",
                           evidence={"lines_sum": lines_sum, "stated_subtotal": inv.subtotal}))
    if inv.total is not None:
        base = inv.subtotal if inv.subtotal is not None else lines_sum
        expected_total = round(base + (inv.tax_amount or 0) + (inv.shipping or 0), 2)
        if not _close(expected_total, inv.total, tol):
            out.append(Finding(code="TOTAL_MISMATCH", severity=W,
                               message=f"Subtotal + tax + shipping = {_money(expected_total)} but the total says {_money(inv.total)} "
                                       f"(difference {_money(inv.total - expected_total)}).",
                               evidence={"expected_total": expected_total, "stated_total": inv.total}))
    if inv.tax_rate and inv.subtotal and inv.tax_amount is not None:
        expected_tax = round(inv.subtotal * inv.tax_rate, 2)
        if not _close(expected_tax, inv.tax_amount, tol):
            out.append(Finding(code="TAX_MISMATCH", severity=W,
                               message=f"Tax at {inv.tax_rate:.0%} of {_money(inv.subtotal)} is {_money(expected_tax)}, invoice says {_money(inv.tax_amount)}."))
    return out


def check_inventory(inv: Invoice, db: Database) -> tuple[list[Finding], dict[str, Optional[str]]]:
    """Match each line to the catalog, then compare stock against the SUM of all
    lines for the same SKU. Splitting one item across lines is how over-orders hide."""
    out: list[Finding] = []
    matches: dict[str, Optional[str]] = {}
    catalog = db.catalog()
    by_sku: dict[str, dict[str, Any]] = {}
    requested: dict[str, float] = defaultdict(float)
    line_count: dict[str, int] = defaultdict(int)
    reported_unknown: set[str] = set()

    for li in inv.line_items:
        raw = li.item or li.description
        row = db.find_item(raw) or db.find_item(li.description)
        matches[li.description] = row["item"] if row else None
        if row is None:
            key = normalize_sku(raw)
            if key in reported_unknown:
                continue
            reported_unknown.add(key)
            suggestion = difflib.get_close_matches(key, [normalize_sku(r["item"]) for r in catalog], n=1, cutoff=0.75)
            hint = ""
            if suggestion:
                hint_row = next(r for r in catalog if normalize_sku(r["item"]) == suggestion[0])
                hint = f" Closest catalog item is {hint_row['item']}, NOT auto-matched: mapping an unknown SKU to a known one is how overbilling slips through."
            out.append(Finding(code="UNKNOWN_ITEM", severity=W, item=raw,
                               message=f"'{raw}' is not in the inventory catalog.{hint}",
                               evidence={"suggestion": hint_row["item"] if suggestion else None}))
            continue
        by_sku[row["item"]] = row
        if li.quantity and li.quantity > 0:
            requested[row["item"]] += li.quantity
            line_count[row["item"]] += 1

    for sku, row in by_sku.items():
        if row["stock"] == 0:
            out.append(Finding(code="ZERO_STOCK_ITEM", severity=C, item=sku,
                               message=f"{sku} has zero stock and no catalog price: we have never carried it. Billing for it matches a known fraud pattern.",
                               evidence={"stock": 0, "requested": requested.get(sku, 0)}))
        elif requested.get(sku, 0) > row["stock"]:
            spread = f" across {line_count[sku]} lines" if line_count[sku] > 1 else ""
            out.append(Finding(code="STOCK_EXCEEDED", severity=W, item=sku,
                               message=f"{sku}: invoice bills {requested[sku]:g} units{spread}, inventory shows only {row['stock']}.",
                               evidence={"requested": requested[sku], "available": row["stock"], "lines": line_count[sku]}))
    if inv.line_items and all(v is None for v in matches.values()):
        out.append(Finding(code="ALL_ITEMS_UNKNOWN", severity=W, message="None of the billed items exist in our catalog."))
    return out, matches


def check_pricing(inv: Invoice, db: Database, variance_pct: float) -> list[Finding]:
    out: list[Finding] = []
    for li in inv.line_items:
        row = db.find_item(li.item or li.description)
        if not row or row["unit_price"] is None or li.unit_price is None:
            continue
        catalog_price = row["unit_price"]
        if li.unit_price > catalog_price * (1 + variance_pct):
            reason = " ".join(filter(None, [li.note, li.description]))
            if _PREMIUM_REASON.search(reason):
                out.append(Finding(code="PRICE_PREMIUM_EXPLAINED", severity=I, item=row["item"],
                                   message=f"{row['item']} billed at {_money(li.unit_price)} vs catalog {_money(catalog_price)}; premium labelled '{li.note or li.description}'."))
            else:
                out.append(Finding(code="PRICE_ABOVE_CATALOG", severity=W, item=row["item"],
                                   message=f"{row['item']} billed at {_money(li.unit_price)}, {li.unit_price / catalog_price - 1:.0%} above catalog {_money(catalog_price)}, no reason given.",
                                   evidence={"billed": li.unit_price, "catalog": catalog_price}))
        elif li.unit_price < catalog_price:
            out.append(Finding(code="PRICE_BELOW_CATALOG", severity=I, item=row["item"],
                               message=f"{row['item']} billed below catalog ({_money(li.unit_price)} vs {_money(catalog_price)}){'; ' + li.note if li.note else ''}."))
    return out


def check_vendor(inv: Invoice, db: Database) -> tuple[list[Finding], Optional[dict[str, Any]]]:
    out: list[Finding] = []
    if not inv.vendor_name:
        return out, None
    vendor = db.find_vendor(inv.vendor_name)
    previous_names = _FORMERLY.findall(" ".join(filter(None, [inv.notes, inv.vendor_name])))
    if previous_names:
        former = previous_names[0].strip(" .")
        known_former = db.find_vendor(former)
        detail = (f" '{former}' IS in our vendor master, so this looks like a rebrand, or an impersonation of a real supplier."
                  if known_former else "")
        out.append(Finding(code="VENDOR_NAME_CHANGE", severity=W,
                           message=f"Vendor says it was formerly '{former}'.{detail} Verify bank details by phone on file before paying (top business-email-compromise pattern).",
                           evidence={"former_name": former, "former_in_master": bool(known_former)}))
        if vendor is None and known_former:
            return out, None
    if vendor is None:
        out.append(Finding(code="UNKNOWN_VENDOR", severity=W,
                           message=f"'{inv.vendor_name}' is not in the vendor master. New payees need onboarding (tax ID, verified bank account) before any payment.",
                           evidence={"vendor": inv.vendor_name}))
        return out, None
    if vendor.get("address") and inv.vendor_address and normalize_vendor(vendor["address"]) != normalize_vendor(inv.vendor_address):
        out.append(Finding(code="VENDOR_ADDRESS_MISMATCH", severity=W,
                           message=f"Address on invoice differs from vendor master ({vendor['address']})."))
    return out, vendor


def check_currency(inv: Invoice, vendor: Optional[dict[str, Any]], base_currency: str) -> list[Finding]:
    out: list[Finding] = []
    if not inv.currency:
        out.append(Finding(code="CURRENCY_ASSUMED", severity=I, message=f"No currency stated; assumed {base_currency}."))
        currency = base_currency
    else:
        currency = inv.currency
    if currency != base_currency:
        out.append(Finding(code="FOREIGN_CURRENCY", severity=W,
                           message=f"Invoice is in {currency}; the payment rail settles in {base_currency}. Needs FX conversion and treasury sign-off.",
                           evidence={"currency": currency}))
    if vendor and vendor.get("currency") and vendor["currency"] != currency:
        out.append(Finding(code="VENDOR_CURRENCY_MISMATCH", severity=W,
                           message=f"Vendor master says this vendor bills in {vendor['currency']}, invoice is in {currency}."))
    return out


def check_dates(inv: Invoice) -> list[Finding]:
    out: list[Finding] = []
    if inv.invoice_date and inv.due_date:
        if inv.due_date < inv.invoice_date:
            out.append(Finding(code="DUE_BEFORE_INVOICE_DATE", severity=W, message=f"Due date {inv.due_date} is before the invoice date {inv.invoice_date}."))
        terms = re.search(r"net\s*(\d+)", inv.payment_terms or "", re.I)
        if terms:
            expected = inv.invoice_date + timedelta(days=int(terms.group(1)))
            if abs((inv.due_date - expected).days) > 3:
                out.append(Finding(code="TERMS_DATE_MISMATCH", severity=I,
                                   message=f"Terms are {inv.payment_terms} (due {expected}) but the invoice asks for {inv.due_date}.",
                                   evidence={"expected_due": str(expected), "stated_due": str(inv.due_date)}))
    return out


def check_prompt_injection(inv: Invoice, raw_text: str) -> list[Finding]:
    """Instructions aimed at our software or reviewers, anywhere in the document
    (including text invisible to a person, such as white 1-pt text in a PDF)."""
    text = "\n".join(filter(None, [raw_text, inv.notes]))
    hits = scan_for_injection(text)
    out: list[Finding] = []
    blatant = [h for h in hits if h.tier == "blatant"]
    social = [h for h in hits if h.tier == "social"]
    if blatant:
        out.append(Finding(code="PROMPT_INJECTION", severity=C,
                           message="The document contains instructions aimed at an automated system ("
                                   + "; ".join(dict.fromkeys(h.reason for h in blatant)) + f"): \"{blatant[0].excerpt}\". "
                                   "A legitimate vendor never writes to AP software. Treated as an attack.",
                           evidence={"excerpts": [h.excerpt for h in blatant]}))
    if social:
        out.append(Finding(code="SOCIAL_ENGINEERING", severity=W,
                           message="The document pressures reviewers ("
                                   + "; ".join(dict.fromkeys(h.reason for h in social)) + f"): \"{social[0].excerpt}\". "
                                   "Approvals are never taken from the invoice itself; verify with the named approver.",
                           evidence={"excerpts": [h.excerpt for h in social]}))
    return out


def check_bank_change(inv: Invoice) -> list[Finding]:
    """Remittance-change requests inside an invoice are the classic BEC attack.
    Bank details are only ever changed through vendor onboarding, never from an invoice."""
    if _BANK_CHANGE.search(inv.notes or ""):
        return [Finding(code="BANK_DETAILS_CHANGE", severity=W,
                        message="Invoice asks us to pay a new or changed bank account. Never act on this from an invoice: "
                                "verify with the vendor using the phone number already on file.",
                        evidence={"notes": (inv.notes or "")[:200]})]
    return []


def check_split_billing(inv: Invoice, key: Optional[str], db: Database, settings: Settings, window_days: int = 7) -> list[Finding]:
    """Several sub-limit invoices from one vendor in a short window that together exceed the
    approval limit: the 'split purchase' pattern used to avoid VP review."""
    limit = settings.approval_threshold
    if not inv.total or inv.total > limit or not inv.invoice_date:
        return []
    related = []
    for row in db.invoices_for_vendor(inv.vendor_name):
        if row["invoice_key"] == key or not row["invoice_date"] or not row["total"] or row["total"] > limit:
            continue
        other = parse_date(row["invoice_date"])
        if other and abs((other - inv.invoice_date).days) <= window_days:
            related.append(row)
    combined = round(inv.total + sum(r["total"] for r in related), 2)
    if related and combined > limit:
        numbers = ", ".join(r["invoice_number"] or "?" for r in related)
        return [Finding(code="SPLIT_INVOICE_PATTERN", severity=W,
                        message=f"With {numbers} from the same vendor within {window_days} days, the combined amount is "
                                f"{_money(combined)}, above the {_money(limit)} limit, while each invoice is below it.",
                        evidence={"related": numbers, "combined": combined})]
    return []


PO_PRICE_TOLERANCE = 0.02   # 2%: rounding and small freight allocations


def check_three_way_match(inv: Invoice, db: Database) -> tuple[list[Finding], Optional[dict[str, Any]]]:
    """Invoice vs purchase order (what we agreed to buy, at what price) vs goods receipt
    (what actually arrived). The SAP flow: PO, goods receipt, invoice verification, GR/IR.

    Returns findings plus a per-line match table for the UI."""
    out: list[Finding] = []
    vendor = db.find_vendor(inv.vendor_name) if inv.vendor_name else None
    po_number = normalize_po(inv.po_number) or find_po_reference(inv.notes)
    inferred = False
    po = None

    if po_number:
        po = db.get_po(po_number)
        if po is None:
            out.append(Finding(code="PO_NOT_FOUND", severity=W, evidence={"po_number": po_number},
                               message=f"The invoice cites {po_number}, which does not exist in purchasing."))
            return out, {"po_number": po_number, "found": False, "inferred": False, "lines": []}
    else:
        skus = {normalize_sku((db.find_item(li.item or li.description) or {}).get("item") or li.item) for li in inv.line_items}
        for candidate in db.open_pos_for_vendor(inv.vendor_name):
            if skus and skus <= {normalize_sku(ln["item"]) for ln in candidate["lines"]}:
                po, inferred = candidate, True
                break
        if po is None:
            if vendor and vendor.get("po_required"):
                out.append(Finding(code="PO_REQUIRED_MISSING", severity=W,
                                   message=f"{vendor['name']} is set up as PO-required, but this invoice cites no purchase "
                                           "order and none matches its items. Purchasing must confirm the order."))
            else:
                out.append(Finding(code="NO_PO", severity=I, message="No purchase order referenced or matched; checked against inventory only."))
            return out, None
        out.append(Finding(code="PO_INFERRED", severity=I,
                           message=f"No PO number on the invoice; matched open {po['po_number']} from the same vendor by its items."))

    if normalize_vendor(po["vendor"]) != normalize_vendor(inv.vendor_name):
        out.append(Finding(code="PO_VENDOR_MISMATCH", severity=W, evidence={"po_vendor": po["vendor"]},
                           message=f"{po['po_number']} was issued to {po['vendor']}, not to {inv.vendor_name or 'the sender'}. "
                                   "Confirm who we actually bought from before paying."))
    if po["status"] != "open":
        out.append(Finding(code="PO_CLOSED", severity=W,
                           message=f"{po['po_number']} is {po['status']}: it has already been fully invoiced or cancelled."))

    rows, issues = [], 0
    billed: dict[str, dict[str, Any]] = {}
    for li in inv.line_items:
        row = db.find_item(li.item or li.description)
        sku = row["item"] if row else (li.item or li.description)
        b = billed.setdefault(normalize_sku(sku), {"item": sku, "qty": 0.0, "price": None})
        b["qty"] += max(0.0, li.quantity or 0.0)
        if li.unit_price is not None:
            b["price"] = max(b["price"] or 0.0, li.unit_price)

    for key, b in billed.items():
        lines = [ln for ln in po["lines"] if normalize_sku(ln["item"]) == key]
        received = sum(r["qty_received"] for r in po["receipts"] if normalize_sku(r["item"]) == key)
        line = {"item": b["item"], "invoiced_qty": b["qty"], "invoiced_price": b["price"], "po_qty": None, "po_price": None,
                "already_invoiced": None, "received": received, "status": []}
        if not lines:
            line["status"].append("not on PO")
            out.append(Finding(code="ITEM_NOT_ON_PO", severity=W, item=b["item"],
                               message=f"{b['item']} is billed but not on {po['po_number']}."))
            rows.append(line)
            issues += 1
            continue
        ordered = sum(ln["qty_ordered"] for ln in lines)
        invoiced = sum(ln["qty_invoiced"] for ln in lines)
        price = lines[0]["unit_price"]
        line.update(po_qty=ordered, po_price=price, already_invoiced=invoiced)
        if b["price"] is not None and b["price"] > price * (1 + PO_PRICE_TOLERANCE):
            line["status"].append("price above PO")
            out.append(Finding(code="PRICE_ABOVE_PO", severity=W, item=b["item"],
                               evidence={"billed": b["price"], "po_price": price},
                               message=f"{b['item']}: billed at {_money(b['price'])}, {po['po_number']} agreed {_money(price)} "
                                       f"({b['price'] / price - 1:+.0%})."))
        if b["qty"] > ordered - invoiced:
            line["status"].append("more than ordered")
            already = f" ({invoiced:g} already invoiced)" if invoiced else ""
            out.append(Finding(code="QTY_EXCEEDS_PO", severity=W, item=b["item"],
                               evidence={"billed": b["qty"], "ordered": ordered, "already_invoiced": invoiced},
                               message=f"{b['item']}: bills {b['qty']:g}, but {po['po_number']} has only "
                                       f"{max(0.0, ordered - invoiced):g} of {ordered:g} left to invoice{already}."))
        if b["qty"] > received - invoiced:
            line["status"].append("not yet received")
            out.append(Finding(code="QTY_NOT_RECEIVED", severity=W, item=b["item"],
                               evidence={"billed": b["qty"], "received": received, "already_invoiced": invoiced},
                               message=(f"{b['item']}: bills {b['qty']:g}, but only {received:g} of {ordered:g} ordered have been received. "
                                        if not invoiced else
                                        f"{b['item']}: bills {b['qty']:g}, but only {max(0.0, received - invoiced):g} received and not yet "
                                        f"invoiced ({received:g} received, {invoiced:g} already invoiced). ") + "Pay only for what arrived."))
        issues += bool(line["status"])
        rows.append(line)

    if not issues and not any(f.code in {"PO_VENDOR_MISMATCH", "PO_CLOSED"} for f in out):
        out.append(Finding(code="THREE_WAY_MATCHED", severity=I,
                           message=f"Three-way match: every line agrees with {po['po_number']} and its goods receipts."))
    for line in rows:
        line["status"] = ", ".join(line["status"]) or "match"
    return out, {"po_number": po["po_number"], "found": True, "inferred": inferred, "po_vendor": po["vendor"],
                 "po_status": po["status"], "lines": rows}


def check_history(inv: Invoice, db: Database) -> tuple[list[Finding], Optional[dict[str, Any]]]:
    """Amount outlier and price drift against this vendor's past invoices, plus a vendor-level
    Benford note when the vendor's amounts stand out from its peers."""
    from .anomaly import check_amount, check_price_drift, vendor_benford_table

    if not inv.vendor_name or not db.find_vendor(inv.vendor_name):
        return [], None  # unknown vendors are handled by UNKNOWN_VENDOR; no history to compare
    history = db.vendor_history(inv.vendor_name)
    findings, profile = check_amount(inv, history)
    findings += check_price_drift(inv, history)
    vendor = db.find_vendor(inv.vendor_name)["name"]
    for row in vendor_benford_table(db.history_by_vendor()):
        if row["vendor"] == vendor and row["outlier"]:
            findings.append(Finding(code="VENDOR_BENFORD_OUTLIER", severity=I,
                                    message=f"{vendor}'s past amounts deviate from Benford's law {row['vs_peers']:.1f}x more than "
                                            f"the typical vendor ({row['invoices']} invoices). Worth a periodic audit of this vendor."))
    context = None
    if profile:
        context = {"n": profile["n"], "median": round(profile["median"], 2), "normal_high": round(profile["normal_high"], 2),
                   "z": profile.get("z"), "ratio": round(profile["ratio"], 2) if profile.get("ratio") else None}
    return findings, context


def check_duplicates(key: Optional[str], digest: str, db: Database) -> list[Finding]:
    """Same vendor + number already seen. Identical content = resubmission; different = revision."""
    out: list[Finding] = []
    if not key:
        return out
    history = db.invoice_history(key)
    payment = db.payment_for(key)
    if payment:
        same = any(h["content_hash"] == digest for h in history if h["status"] == "PAID")
        if same:
            out.append(Finding(code="DUPLICATE_OF_PAID", severity=C,
                               message=f"Already paid {_money(payment['amount'])} on {payment['paid_at'][:10]} (ref {payment['reference']}). This is a resubmission.",
                               evidence={"payment_reference": payment["reference"]}))
        else:
            out.append(Finding(code="REVISED_AFTER_PAYMENT", severity=W,
                               message=f"Same invoice number was already paid ({_money(payment['amount'])}, ref {payment['reference']}) but the contents changed. "
                                       "Paying it again would double-pay the original lines; needs a credit/re-bill decision by AP.",
                               evidence={"paid_amount": payment["amount"]}))
    elif history:
        last = history[-1]
        same = any(h["content_hash"] == digest for h in history)
        code = "DUPLICATE_SUBMISSION" if same else "REVISION_OF_PENDING"
        message = (f"Identical invoice already received from {last['source_file']} (status {last['status']})."
                   if same else f"An earlier version from {last['source_file']} is still {last['status']}; contents differ.")
        out.append(Finding(code=code, severity=W, message=message, evidence={"previous_status": last["status"]}))
    return out


def score_fraud(inv: Invoice, findings: list[Finding], settings: Settings) -> tuple[int, list[str], list[Finding]]:
    """Explainable additive risk score. Every point maps to a named signal."""
    codes = {f.code for f in findings}
    text = " ".join(filter(None, [inv.notes, inv.payment_terms, inv.due_date_raw]))
    signals: list[tuple[str, int]] = []
    if _URGENCY.search(text):
        signals.append(("pressure / urgency language", 25))
    if re.search(r"\b(immediate|due on receipt|upon receipt)\b", inv.payment_terms or "", re.I):
        signals.append(("immediate payment terms", 10))
    if "UNPARSEABLE_DUE_DATE" in codes:
        signals.append(("relative or invalid due date", 15))
    if "ZERO_STOCK_ITEM" in codes:
        signals.append(("bills an item we have never stocked", 40))
    if "UNKNOWN_VENDOR" in codes:
        signals.append(("payee not in vendor master", 20))
    unknown = sum(1 for f in findings if f.code == "UNKNOWN_ITEM")
    if unknown:
        signals.append((f"{unknown} unknown item(s)", min(20, 10 * unknown)))
    if "ALL_ITEMS_UNKNOWN" in codes:
        signals.append(("no billed item exists in catalog", 10))
    if "VENDOR_NAME_CHANGE" in codes:
        signals.append(("vendor identity changed", 20))
    if "PROMPT_INJECTION" in codes:
        signals.append(("instructions aimed at the AP system", 40))
    if "PO_VENDOR_MISMATCH" in codes:
        signals.append(("purchase order issued to a different vendor", 15))
    if "AMOUNT_ANOMALY" in codes:
        signals.append(("amount far above this vendor's normal range", 10))
    if "PRICE_DRIFT" in codes:
        signals.append(("unit price above anything this vendor has charged", 5))
    if "PO_NOT_FOUND" in codes:
        signals.append(("cites a purchase order that does not exist", 15))
    if "SOCIAL_ENGINEERING" in codes:
        signals.append(("claims approval or asks to skip review", 20))
    if "BANK_DETAILS_CHANGE" in codes:
        signals.append(("request to pay a new bank account", 30))
    if "SPLIT_INVOICE_PATTERN" in codes:
        signals.append(("possible split billing under the limit", 15))
    limit = settings.approval_threshold
    if inv.total and limit * (1 - settings.threshold_proximity_pct) <= inv.total < limit:
        signals.append((f"amount just under the {_money(limit)} approval limit", 10))
    if inv.total and inv.total >= 10_000 and inv.total % 1_000 == 0:
        signals.append(("large round-number amount", 5))
    if "DUE_BEFORE_INVOICE_DATE" in codes:
        signals.append(("due date precedes invoice date", 5))

    score = min(100, sum(points for _, points in signals))
    labels = [f"{name} (+{points})" for name, points in signals]
    extra: list[Finding] = []
    if inv.total and limit * (1 - settings.threshold_proximity_pct) <= inv.total < limit:
        extra.append(Finding(code="NEAR_APPROVAL_THRESHOLD", severity=I,
                             message=f"{_money(inv.total)} sits just under the {_money(limit)} extra-scrutiny limit. Alone it means nothing; watch for split invoices."))
    if score >= settings.fraud_reject_score:
        extra.append(Finding(code="FRAUD_RISK_HIGH", severity=C, message=f"Fraud risk score {score}/100: " + "; ".join(labels),
                             evidence={"score": score}))
    elif score >= settings.fraud_review_score:
        extra.append(Finding(code="FRAUD_RISK_ELEVATED", severity=W, message=f"Fraud risk score {score}/100: " + "; ".join(labels),
                             evidence={"score": score}))
    return score, labels, extra
