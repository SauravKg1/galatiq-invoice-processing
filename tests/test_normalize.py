from datetime import date

import pytest

from ap_autopilot.normalize import (content_hash, fix_ocr_digits, invoice_key, normalize_invoice_number,
                                    normalize_sku, normalize_vendor, parse_date, split_description, to_float)
from ap_autopilot.models import Invoice, LineItem


@pytest.mark.parametrize("raw, fixed", [
    ("26-Jan-2O26", "26-Jan-2026"),
    ("$3,500.O0", "$3,500.00"),
    ("INVOCE", "INVOCE"),            # letters next to letters are left alone
    ("Accounts Payble", "Accounts Payble"),
])
def test_fix_ocr_digits(raw, fixed):
    assert fix_ocr_digits(raw) == fixed


@pytest.mark.parametrize("raw, value", [("$1,000.00", 1000.0), ("-5", -5.0), ("$3,500.O0", 3500.0), ("", None), ("abc", None), (7, 7.0)])
def test_to_float(raw, value):
    assert to_float(raw) == value


@pytest.mark.parametrize("raw, value", [
    ("2026-01-15", date(2026, 1, 15)), ("Jan 30 2026", date(2026, 1, 30)), ("January 27, 2026", date(2026, 1, 27)),
    ("26-Jan-2O26", date(2026, 1, 26)), ("02/28/2026", date(2026, 2, 28)), ("yesterday", None), (None, None),
])
def test_parse_date(raw, value):
    assert parse_date(raw) == value


def test_identity_normalization():
    assert normalize_invoice_number("INV 1012") == normalize_invoice_number("1012") == "INV-1012"
    assert normalize_vendor("Widgets Inc.") == normalize_vendor("WIDGETS, INC") == "widgets"
    assert normalize_sku("Widget A") == normalize_sku("widget-a") == "widgeta"
    assert normalize_sku("WidgetC") != normalize_sku("WidgetA")
    assert split_description("WidgetA (rush order)") == ("WidgetA", "rush order")
    assert invoice_key("Summit Manufacturing Co.", "INV-1011") == invoice_key("Summit Manufacturing Co", "inv 1011")


def test_content_hash_distinguishes_revision_from_resubmission():
    a = Invoice(vendor_name="X", invoice_number="INV-1", total=10, line_items=[LineItem(description="A", item="A", quantity=1, unit_price=10)])
    same = a.model_copy()
    changed = a.model_copy(update={"total": 20})
    assert content_hash(a) == content_hash(same)
    assert content_hash(a) != content_hash(changed)
