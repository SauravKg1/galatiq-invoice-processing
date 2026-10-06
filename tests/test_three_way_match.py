"""Three-way match: invoice vs purchase order vs goods receipt."""

from __future__ import annotations

import sqlite3

import pytest

from ap_autopilot.db import Database
from ap_autopilot.loaders import load_document
from ap_autopilot.models import Invoice, LineItem
from ap_autopilot.normalize import find_po_reference, normalize_po
from ap_autopilot.parsers import parse_structured, parse_text_heuristic
from ap_autopilot.rules import check_three_way_match
from ap_autopilot.tools import build_validation_tools
from conftest import EXTRA, INVOICES


def inv(vendor="Summit Manufacturing Co.", po="PO-4500013", items=(("WidgetB", 4, 480.0),), **kw):
    return Invoice(vendor_name=vendor, invoice_number="INV-9", po_number=po, total=1.0,
                   line_items=[LineItem(description=i, item=i, quantity=q, unit_price=p) for i, q, p in items], **kw)


def codes(findings):
    return {f.code for f in findings}


@pytest.mark.parametrize("raw, expected", [("PO 4500012", "PO-4500012"), ("po#4500012", "PO-4500012"),
                                           ("4500012", "PO-4500012"), ("PO-20260115", "PO-20260115"), ("none", None)])
def test_normalize_po(raw, expected):
    assert normalize_po(raw) == expected


def test_po_reference_in_free_text_needs_digits():
    assert find_po_reference("NOTES: Ref PO-20260115. Deliver to dock B.") == "PO-20260115"
    assert find_po_reference("There is no po here") is None


def test_parsers_extract_po_numbers():
    assert parse_text_heuristic(load_document(EXTRA / "invoice_2012_billed_before_delivery.txt").text).po_number == "PO-4500012"
    assert parse_structured(load_document(EXTRA / "invoice_2015_three_way_match.json")).po_number == "PO-4500015"


def test_full_match(db):
    findings, match = check_three_way_match(inv(), db)
    assert codes(findings) == {"THREE_WAY_MATCHED"} and match["lines"][0]["status"] == "match"


@pytest.mark.parametrize("price, flagged", [(489.0, False), (500.0, True)])
def test_price_tolerance_is_two_percent(db, price, flagged):
    findings, _ = check_three_way_match(inv(items=(("WidgetB", 4, price),)), db)
    assert ("PRICE_ABOVE_PO" in codes(findings)) is flagged


def test_billing_ahead_of_delivery(db):
    findings, match = check_three_way_match(inv("Widgets Inc.", "PO-4500012", (("WidgetA", 10, 250.0),)), db)
    assert "QTY_NOT_RECEIVED" in codes(findings) and "QTY_EXCEEDS_PO" not in codes(findings)
    assert match["lines"][0]["received"] == 6


def test_po_found_by_items_when_invoice_cites_none(db):
    findings, match = check_three_way_match(inv("Reliable Components Inc.", None, (("GadgetX", 3, 750.0),)), db)
    assert match["po_number"] == "PO-4500015" and match["inferred"]
    assert {"PO_INFERRED", "THREE_WAY_MATCHED"} <= codes(findings)


def test_unknown_po_and_item_not_on_po(db):
    assert "PO_NOT_FOUND" in codes(check_three_way_match(inv(po="PO-999999"), db)[0])
    assert "ITEM_NOT_ON_PO" in codes(check_three_way_match(inv(items=(("GadgetX", 1, 750.0),)), db)[0])


def test_po_required_vendors_are_held_others_are_informational(db):
    no_po = (("WidgetA", 1, 250.0),)
    assert "PO_REQUIRED_MISSING" in codes(check_three_way_match(inv("MegaWidgets Corp", None, no_po), db)[0])
    assert codes(check_three_way_match(inv("Widgets Inc.", None, (("WidgetB", 1, 500.0),)), db)[0]) == {"NO_PO"}


def test_paying_consumes_the_po_so_it_cannot_be_billed_twice(pipeline, db):
    first = pipeline.process(EXTRA / "invoice_2015_three_way_match.json")
    second = pipeline.process(EXTRA / "invoice_2016_rebilled_po.txt")
    assert first.status == "PAID" and db.get_po("PO-4500015")["status"] == "closed"
    assert {"QTY_EXCEEDS_PO", "PO_CLOSED"} <= second.validation.codes and second.status != "PAID"


def test_human_approval_also_consumes_the_po(pipeline, db):
    pipeline.process(EXTRA / "invoice_2012_billed_before_delivery.txt")
    item = db.review_queue()[0]
    pipeline.human_decision(item["id"], approve=True, reviewer="saurav", note="Remaining 4 delivered today, GR posted late")
    assert db.get_po("PO-4500012")["lines"][0]["qty_invoiced"] == 10


def test_fastship_po_on_quickship_invoice(pipeline):
    out = pipeline.process(INVOICES / "invoice_1012.pdf")
    assert "PO_VENDOR_MISMATCH" in out.validation.codes
    assert out.validation.po_match["po_vendor"] == "FastShip Ltd."


def test_investigator_can_look_up_purchase_orders(db):
    tool = next(t for t in build_validation_tools(db) if t.name == "get_purchase_order")
    assert tool.fn(po_number="po 4500013")["lines"][0]["unit_price"] == 480.0
    assert tool.fn(po_number="PO-1")["found"] is False


def test_old_database_gains_po_required_flags(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE vendors (name TEXT PRIMARY KEY, status TEXT, currency TEXT, address TEXT, notes TEXT, email TEXT)")
    conn.execute("INSERT INTO vendors VALUES ('MegaWidgets Corp', 'approved', 'USD', NULL, NULL, NULL)")
    conn.commit()
    conn.close()
    assert Database(str(path)).ensure().find_vendor("MegaWidgets Corp")["po_required"] == 1
