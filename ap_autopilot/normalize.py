"""Small, pure normalization helpers shared by parsers and validators.

Kept dependency-free and side-effect-free so they are trivially unit tested.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any, Optional

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%b-%Y", "%d %b %Y",
    "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%d %B %Y", "%Y%m%d",
)

_VENDOR_SUFFIXES = re.compile(r"\b(inc|incorporated|llc|ltd|limited|co|corp|corporation|company|gmbh|plc)\b\.?", re.I)


def fix_ocr_digits(text: str) -> str:
    """Replace the letter O used as zero inside numbers ('2O26' -> '2026', '$3,500.O0' -> '$3,500.00')."""
    previous = None
    while previous != text:  # repeat until stable so 'O0O' style runs resolve
        previous = text
        text = re.sub(r"(?<=[\d.,$])[Oo](?=[\d.,]|\b)", "0", text)
        text = re.sub(r"(?<=\b)[Oo](?=\d)", "0", text)
    return text


def to_float(value: Any) -> Optional[float]:
    """Parse money/quantities: '$1,000.00', '1.000', '-5', 'O.50'. Returns None if not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = fix_ocr_digits(str(value)).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^\d.\-]", "", text)
    if text in {"", "-", ".", "-."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


def parse_date(value: Any) -> Optional[date]:
    """Parse the date styles seen in the sample data. 'yesterday' or '' return None on purpose:
    relative dates on an invoice are a red flag, not something to resolve silently."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = fix_ocr_digits(str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_invoice_number(value: Optional[str]) -> Optional[str]:
    """'INV 1012', 'inv-1012', '1012', '#INV-1012' -> 'INV-1012'."""
    if not value:
        return None
    digits = re.findall(r"\d+", fix_ocr_digits(str(value)))
    if not digits:
        return str(value).strip().upper() or None
    return f"INV-{''.join(digits)}"


def normalize_vendor(value: Optional[str]) -> str:
    """Comparison key for vendor names: case, punctuation and legal suffixes removed."""
    if not value:
        return ""
    text = _VENDOR_SUFFIXES.sub("", value.lower())
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_sku(value: Optional[str]) -> str:
    """'Widget A' / 'widget-a' / 'WIDGETA' -> 'widgeta'. Formatting only, never fuzzy guessing."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def split_description(description: str) -> tuple[str, Optional[str]]:
    """'WidgetA (rush order)' -> ('WidgetA', 'rush order')."""
    match = re.match(r"^\s*(.*?)\s*\((.+)\)\s*$", description or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return (description or "").strip(), None


def invoice_key(vendor: Optional[str], number: Optional[str]) -> Optional[str]:
    """Identity of a bill for duplicate detection: same vendor + same invoice number."""
    num = normalize_invoice_number(number)
    if not num:
        return None
    return f"{normalize_vendor(vendor) or 'unknown-vendor'}:{num}"


def content_hash(invoice: Any) -> str:
    """Fingerprint of what we would pay: vendor, number, items and total.
    Same key + same hash = resubmission. Same key + different hash = revision."""
    items = sorted(
        (normalize_sku(li.item or li.description), li.quantity or 0, round(li.unit_price or 0, 2))
        for li in invoice.line_items
    )
    payload = {
        "vendor": normalize_vendor(invoice.vendor_name),
        "number": normalize_invoice_number(invoice.invoice_number),
        "total": round(invoice.total or 0, 2),
        "items": items,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


_PO_RE = re.compile(r"\b(?:P\.?\s?O\.?|purchase\s+order)\s*(?:number|no\.?|#)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})", re.I)


def normalize_po(value: Optional[str]) -> Optional[str]:
    """'PO 4500012', 'po#4500012', 'PO-4500012', '4500012' -> 'PO-4500012'."""
    if not value:
        return None
    text = str(value).strip().upper()
    match = _PO_RE.search(text)
    core = match.group(1) if match else text
    core = re.sub(r"^PO-?", "", core).strip("-")
    return f"PO-{core}" if core and re.search(r"\d", core) else None


def find_po_reference(text: Optional[str]) -> Optional[str]:
    """A PO number mentioned anywhere in free text, e.g. 'Ref PO-20260115'."""
    for match in _PO_RE.finditer(text or ""):
        if re.search(r"\d", match.group(1)):  # a real PO number always has digits
            return normalize_po(match.group(1))
    return None
