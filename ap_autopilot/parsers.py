"""Deterministic extraction.

* Structured formats (JSON, CSV, XML) are mapped field by field. An LLM adds no
  accuracy here, only cost, latency and a chance to hallucinate, so we do not
  use one. (Ruthless scoping: use the model where it earns its place.)
* Free text and PDFs get a heuristic parser. In online mode it is the
  cross-check for the LLM extraction; in offline mode it is the extractor.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .loaders import SourceDocument
from .models import Invoice, LineItem
from .normalize import fix_ocr_digits, parse_date, split_description, to_float

# --------------------------------------------------------------------------- #
# Structured formats
# --------------------------------------------------------------------------- #
_FIELD_ALIASES = {
    "invoice_number": ("invoice_number", "invoice_no", "invoice", "number", "inv_no", "id"),
    "vendor_name": ("vendor", "vendor_name", "supplier", "seller", "from"),
    "invoice_date": ("date", "invoice_date", "issue_date", "issued"),
    "due_date": ("due_date", "due", "payment_due"),
    "currency": ("currency", "ccy"),
    "subtotal": ("subtotal", "sub_total"),
    "tax_rate": ("tax_rate",),
    "tax_amount": ("tax_amount", "tax", "vat"),
    "shipping": ("shipping", "freight"),
    "total": ("total", "total_amount", "amount_due", "grand_total"),
    "payment_terms": ("payment_terms", "terms"),
    "notes": ("notes", "note", "memo"),
    "revision": ("revision", "rev"),
    "po_number": ("po_number", "po", "purchase_order", "po_ref", "po_no", "purchase_order_number"),
}


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _pick(data: dict[str, Any], field: str) -> Any:
    lowered = {_norm_key(k): v for k, v in data.items()}
    for alias in _FIELD_ALIASES[field]:
        if alias in lowered and lowered[alias] not in (None, ""):
            return lowered[alias]
    return None


def _line_item(raw: dict[str, Any]) -> LineItem:
    d = {_norm_key(k): v for k, v in raw.items()}
    description = str(d.get("item") or d.get("name") or d.get("description") or d.get("sku") or "").strip()
    name, paren_note = split_description(description)
    note = d.get("note") or d.get("notes") or paren_note
    return LineItem(
        description=description,
        item=name or None,
        quantity=to_float(d.get("quantity", d.get("qty"))),
        unit_price=to_float(d.get("unit_price", d.get("price", d.get("rate")))),
        amount=to_float(d.get("amount", d.get("line_total", d.get("total")))),
        note=str(note).strip() if note else None,
    )


def _invoice_from_mapping(data: dict[str, Any], items: list[dict[str, Any]]) -> Invoice:
    vendor = _pick(data, "vendor_name")
    address = None
    if isinstance(vendor, dict):
        address = vendor.get("address")
        vendor = vendor.get("name")
    due_raw = _pick(data, "due_date")
    tax_rate = to_float(_pick(data, "tax_rate"))
    return Invoice(
        invoice_number=str(_pick(data, "invoice_number") or "").strip() or None,
        vendor_name=(str(vendor).strip() or None) if vendor else None,
        vendor_address=address,
        invoice_date=parse_date(_pick(data, "invoice_date")),
        due_date=parse_date(due_raw),
        due_date_raw=str(due_raw) if due_raw is not None else None,
        currency=_pick(data, "currency"),
        line_items=[_line_item(i) for i in items],
        subtotal=to_float(_pick(data, "subtotal")),
        tax_rate=tax_rate / 100 if tax_rate and tax_rate > 1 else tax_rate,
        tax_amount=to_float(_pick(data, "tax_amount")),
        shipping=to_float(_pick(data, "shipping")),
        total=to_float(_pick(data, "total")),
        payment_terms=_pick(data, "payment_terms") or None,
        notes=_pick(data, "notes"),
        revision=_pick(data, "revision"),
        po_number=str(_pick(data, "po_number")) if _pick(data, "po_number") else None,
    )


def _parse_json(data: Any) -> Invoice:
    if isinstance(data, list):  # a list holding one invoice
        data = data[0] if data else {}
    items = data.get("line_items") or data.get("items") or []
    return _invoice_from_mapping(data, items)


def _parse_xml(data: dict[str, Any]) -> Invoice:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and key not in {"line_items", "items"}:
            flat.update(value)  # header / totals sections
        else:
            flat[key] = value
    raw_items = data.get("line_items") or data.get("items") or {}
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("item") or raw_items.get("line_item") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    return _invoice_from_mapping(flat, raw_items)


_SUMMARY_LABEL = re.compile(r"^\s*(sub\s*total|tax|vat|shipping|freight|total|grand total)\b\s*(?:\(([\d.]+)\s*%\))?", re.I)


def _apply_summary(fields: dict[str, Any], label: str, value: Any) -> None:
    match = _SUMMARY_LABEL.match(label)
    if not match:
        return
    kind = match.group(1).lower().replace(" ", "")
    amount = to_float(value)
    if kind == "subtotal":
        fields["subtotal"] = amount
    elif kind in {"tax", "vat"}:
        fields["tax_amount"] = amount
        if match.group(2):
            fields["tax_rate"] = float(match.group(2)) / 100
    elif kind in {"shipping", "freight"}:
        fields["shipping"] = amount
    else:
        fields["total"] = amount


def _parse_csv(rows: list[list[str]]) -> Invoice:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return Invoice()
    header = [_norm_key(c) for c in rows[0]]

    # Layout 1: key/value pairs ("field,value") with repeated item/quantity/unit_price keys
    if header[:2] == ["field", "value"]:
        fields: dict[str, Any] = {}
        items: list[dict[str, Any]] = []
        for row in rows[1:]:
            key, value = _norm_key(row[0]), (row[1] if len(row) > 1 else "")
            if key in {"item", "name", "description"}:
                items.append({"item": value})
            elif key in {"quantity", "qty", "unit_price", "price", "amount", "line_total"} and items:
                items[-1][key] = value
            elif key in {"tax"} :
                fields["tax_amount"] = value
            else:
                fields[key] = value
        return _invoice_from_mapping(fields, items)

    # Layout 2: tabular, one row per line item, summary rows at the bottom
    fields = {}
    items = []
    for row in rows[1:]:
        record = dict(zip(header, row))
        if (record.get("item") or "").strip():
            for key in ("invoice_number", "vendor", "date", "due_date", "currency", "payment_terms"):
                if record.get(key):
                    fields.setdefault(key, record[key])
            items.append(record)
        else:  # summary row: find the "Label:" cell and the value after it
            cells = [c.strip() for c in row]
            for i, cell in enumerate(cells):
                if cell.endswith(":") and i + 1 < len(cells):
                    _apply_summary(fields, cell.rstrip(":"), cells[i + 1])
    return _invoice_from_mapping(fields, items)


def parse_structured(doc: SourceDocument) -> Optional[Invoice]:
    if not doc.is_structured:
        return None
    if doc.format == "json":
        return _parse_json(doc.data)
    if doc.format == "xml":
        return _parse_xml(doc.data)
    if doc.format == "csv":
        return _parse_csv(doc.data)
    return None


# --------------------------------------------------------------------------- #
# Free text (TXT, email bodies, PDF text layers)
# --------------------------------------------------------------------------- #
_LABELS = {
    "invoice_number": {"invoicenumber", "invoiceno", "invoice", "inv", "invno", "invoicenum", "invnumber"},
    "vendor_priority": {"vendor", "vndr", "supplier", "billfrom", "remitto", "seller"},
    "vendor_fallback": {"from"},
    "invoice_date": {"date", "dt", "invoicedate", "issued", "issuedate"},
    "due_date": {"duedate", "duedt", "due", "paymentdue", "dueby"},
    "payment_terms": {"paymentterms", "terms", "pymntterms", "pymtterms", "paymtterms"},
    "notes": {"notes", "note", "memo", "comments", "remarks"},
    "po_number": {"po", "ponumber", "pono", "purchaseorder", "purchaseordernumber", "poref"},
    "currency": {"currency"},
}
_KNOWN_LABELS = set().union(*_LABELS.values()) | {"subject", "to", "attn", "billto", "shipto"}

_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?P<label>"
    r"invoice\s*(?:number|no\.?|num|#)?|inv\s*(?:no\.?|#|number)?|"
    r"vendor|vndr|supplier|bill\s*from|remit\s*to|seller|from|"
    r"(?:invoice\s*)?date|dt|issue(?:d|\s*date)?|"
    r"due\s*(?:date|dt|by)?|payment\s*due|"
    r"(?:payment|pymnt|pymt|paymt)?\s*terms|"
    r"notes?|memo|comments|remarks|currency|subject|bill\s*to|ship\s*to|attn|to|"
    r"p\.?\s?o\.?\s*(?:number|no\.?|#|ref)?|purchase\s+order(?:\s+number)?"
    r")\s*:\s*",
    re.I,
)
_INLINE_INVOICE_NO = re.compile(r"\binvoice\s*#\s*([A-Z]{0,4}-?\s?\d+)", re.I)
_MONEY = r"\$?\s*(-?[\d,]+(?:\.\d+)?)"
_SUBTOTAL = re.compile(r"\bsub\s*-?\s*total\b\s*:?\s*" + _MONEY, re.I)
_TAX = re.compile(r"\b(?:sales\s+)?(?:tax|vat)\b\s*(?:\(\s*([\d.]+)\s*%\s*\))?\s*:?\s*" + _MONEY, re.I)
_SHIPPING = re.compile(r"\b(?:shipping|freight|delivery)\b\s*:?\s*" + _MONEY, re.I)
_TOTAL = re.compile(r"(?<!sub)(?<!sub )\b(?:grand\s+total|total\s+amount|total\s+due|amount\s+due|total|amt)\b\s*:?\s*" + _MONEY, re.I)
_ITEM = re.compile(
    r"""^\s*(?:[-*•]\s*)?
    (?P<name>[A-Za-z][A-Za-z0-9 ()/&.\-]*?)\s+
    (?:qty\s*:?\s*|x(?=-?\d))?(?P<qty>-?\d+(?:\.\d+)?)\s+
    (?:unit\s*price\s*:?\s*|rate\s*:?\s*|@\s*)?\$?\s*(?P<price>-?[\d,]+(?:\.\d+)?)
    (?:\s*(?:ea|each|/unit|per\s+unit))?
    (?:\s+\$?\s*(?P<amount>-?[\d,]+\.\d{2}))?
    (?P<rest>\s+.*)?$""",
    re.I | re.X,
)
_NOT_ITEM_NAMES = {"subtotal", "total", "tax", "shipping", "amount", "amt", "date", "invoice", "qty"}


def _label_key(label: str) -> str:
    return re.sub(r"[^a-z]", "", label.lower())


def _label_segments(line: str) -> list[tuple[str, str]]:
    """Split 'Vendor: Atlas Industrial Supply Due: 2026-03-24' into labelled values.
    Only known labels count, so values containing words like 'Supply' stay intact."""
    matches = list(_LABEL_RE.finditer(line))
    segments = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        segments.append((_label_key(match.group("label")), line[match.end():end].strip()))
    return segments


def parse_text_heuristic(text: str) -> Invoice:
    text = fix_ocr_digits(text)
    lines = text.splitlines()
    found: dict[str, str] = {}
    notes: list[str] = []
    vendor_note: Optional[str] = None
    in_notes = False
    items: list[LineItem] = []
    totals: dict[str, Any] = {}

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            in_notes = False
            continue

        # money summary lines
        if _SUBTOTAL.search(stripped):
            totals["subtotal"] = to_float(_SUBTOTAL.search(stripped).group(1))
            continue
        tax = _TAX.search(stripped)
        if tax and not _ITEM.match(stripped):
            totals["tax_amount"] = to_float(tax.group(2))
            if tax.group(1):
                totals["tax_rate"] = float(tax.group(1)) / 100
            continue
        ship = _SHIPPING.search(stripped)
        if ship and ":" in stripped:
            totals["shipping"] = to_float(ship.group(1))
            continue
        total = _TOTAL.search(stripped)
        if total and not _ITEM.match(stripped):
            totals["total"] = to_float(total.group(1))  # last one wins: totals sit at the bottom
            continue

        segments = _label_segments(stripped) if ":" in stripped else []
        labelled = [(k, v) for k, v in segments if k in _KNOWN_LABELS]
        if labelled and not re.search(r"\bqty\b", stripped, re.I):
            in_notes = False
            for key, value in labelled:
                for field_name, aliases in _LABELS.items():
                    if key in aliases and value:
                        if field_name == "notes":
                            notes.append(value)
                            in_notes = True
                        elif field_name.startswith("vendor"):
                            if "@" not in value:
                                found.setdefault(field_name, re.split(r"\s{2,}", value)[0])
                                nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
                                if nxt.startswith("(") and nxt.endswith(")"):
                                    vendor_note = nxt.strip("()")
                        else:
                            found.setdefault(field_name, value)
            continue

        inline_no = _INLINE_INVOICE_NO.search(stripped)
        if inline_no:
            found.setdefault("invoice_number", inline_no.group(1))
            continue

        item = _ITEM.match(stripped)
        if item and _label_key(item.group("name")) not in _NOT_ITEM_NAMES:
            description = item.group("name").strip()
            name, paren_note = split_description(description)
            rest = (item.group("rest") or "").strip() or None
            items.append(LineItem(
                description=description,
                item=name,
                quantity=to_float(item.group("qty")),
                unit_price=to_float(item.group("price")),
                amount=to_float(item.group("amount")),
                note=paren_note or rest,
            ))
            in_notes = False
            continue

        if in_notes:
            notes.append(stripped)

    vendor = found.get("vendor_priority") or found.get("vendor_fallback")
    if vendor_note:
        notes.insert(0, f"Vendor annotation: ({vendor_note})")
    due_raw = found.get("due_date")
    return Invoice(
        invoice_number=found.get("invoice_number"),
        vendor_name=vendor.strip() if vendor else None,
        invoice_date=parse_date(found.get("invoice_date")),
        due_date=parse_date(due_raw),
        due_date_raw=due_raw,
        currency=found.get("currency"),
        line_items=items,
        subtotal=totals.get("subtotal"),
        tax_rate=totals.get("tax_rate"),
        tax_amount=totals.get("tax_amount"),
        shipping=totals.get("shipping"),
        total=totals.get("total"),
        payment_terms=found.get("payment_terms"),
        po_number=found.get("po_number"),
        notes=" ".join(notes) if notes else None,
    )
