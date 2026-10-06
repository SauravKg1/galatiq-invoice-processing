"""Every provided sample must extract correctly, in every format."""

from datetime import date

import pytest

from ap_autopilot.loaders import DocumentLoadError, load_document
from ap_autopilot.parsers import parse_structured, parse_text_heuristic
from conftest import EXTRA, INVOICES


def extract(path):
    doc = load_document(path)
    return parse_structured(doc) if doc.is_structured else parse_text_heuristic(doc.text)


# file, invoice number (normalized digits), vendor, total, line count
CASES = [
    ("invoice_1001.txt", "1001", "Widgets Inc.", 5000.0, 2),
    ("invoice_1002.txt", "1002", "Gadgets Co.", 15000.0, 1),
    ("invoice_1003.txt", "1003", "Fraudster LLC", 100000.0, 1),
    ("invoice_1004.json", "1004", "Precision Parts Ltd.", 1890.0, 2),
    ("invoice_1004_revised.json", "1004", "Precision Parts Ltd.", 5940.0, 3),
    ("invoice_1005.json", "1005", "Global Supply Chain Partners", 15225.0, 3),
    ("invoice_1006.csv", "1006", "Acme Industrial Supplies", 2750.0, 2),
    ("invoice_1007.csv", "1007", "MegaWidgets Corp", 15525.0, 3),
    ("invoice_1008.txt", "1008", "NoProd Industries", 9900.0, 2),
    ("invoice_1009.json", "1009", None, -250.0, 2),
    ("invoice_1010.txt", "1010", "Consolidated Materials Group", 7185.0, 4),
    ("invoice_1011.pdf", "1011", "Summit Manufacturing Co.", 3000.0, 2),
    ("invoice_1011.txt", "1011", "Summit Manufacturing Co.", 3000.0, 2),
    ("invoice_1012.pdf", "1012", "QuickShip Distributers", 9975.0, 3),
    ("invoice_1012.txt", "1012", "QuickShip Distributers", 9975.0, 3),
    ("invoice_1013.json", "1013", "Atlas Industrial Supply", 22562.8, 8),
    ("invoice_1013.pdf", "1013", "Atlas Industrial Supply", 22562.8, 8),
    ("invoice_1014.xml", "1014", "TechParts International", 4125.0, 2),
    ("invoice_1015.csv", "1015", "Reliable Components Inc.", 6500.0, 3),
    ("invoice_1016.json", "1016", "Widgets Inc.", 3233.0, 3),
]


@pytest.mark.parametrize("name, number, vendor, total, n_items", CASES)
def test_sample_extraction(name, number, vendor, total, n_items):
    inv = extract(INVOICES / name)
    assert number in (inv.invoice_number or "")
    assert inv.vendor_name == vendor
    assert inv.total == pytest.approx(total)
    assert len(inv.line_items) == n_items


def test_messy_text_details():
    inv = extract(INVOICES / "invoice_1002.txt")  # 'Vndr', 'Dt', 'Itms', '@ $750 ea'
    assert inv.line_items[0].quantity == 20 and inv.line_items[0].unit_price == 750
    assert inv.invoice_date == date(2026, 1, 30)

    inv = extract(INVOICES / "invoice_1012.pdf")  # OCR letter O, spaced SKUs, vendor annotation
    assert [li.item for li in inv.line_items] == ["Widget A", "WidgetB", "Gadget X"]
    assert inv.invoice_date == date(2026, 1, 26)
    assert "formerly FastShip" in inv.notes

    inv = extract(INVOICES / "invoice_1003.txt")
    assert inv.due_date is None and inv.due_date_raw == "yesterday"
    assert "Wire transfer" in inv.notes

    inv = extract(INVOICES / "invoice_1010.txt")
    assert inv.shipping == 150 and inv.line_items[3].note == "rush order"

    inv = extract(INVOICES / "invoice_1013.pdf")  # two labels on one line
    assert inv.due_date == date(2026, 3, 24)


def test_email_and_eml_bodies():
    inv = extract(INVOICES / "invoice_1008.txt")
    assert inv.vendor_name == "NoProd Industries"  # not the From: address
    inv = extract(EXTRA / "invoice_2004_bank_change.eml")
    assert inv.vendor_name == "Summit Manufacturing Co." and "bank details have changed" in inv.notes


def test_loader_errors(tmp_path):
    with pytest.raises(DocumentLoadError):
        load_document(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(DocumentLoadError):
        load_document(empty)


def test_malformed_json_degrades_to_text(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"invoice_number": "INV-9", "vendor": "Widgets Inc.", total: 10')
    doc = load_document(bad)
    assert not doc.is_structured and doc.warnings
