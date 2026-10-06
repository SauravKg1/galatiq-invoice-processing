"""History-based anomaly detection: amount outliers, price drift, Benford peer comparison."""

from __future__ import annotations

import math

import pytest

from ap_autopilot.anomaly import (BENFORD, MIN_HISTORY, amount_profile, benford, check_amount, check_price_drift,
                                  vendor_benford_table)
from ap_autopilot.history import generate
from ap_autopilot.models import Invoice, LineItem
from ap_autopilot.rules import check_history
from ap_autopilot.tools import build_validation_tools
from conftest import EXTRA, INVOICES


def hist(totals, item="WidgetA", price=250.0):
    return [{"invoice_number": f"H{i}", "invoice_date": "2025-06-01", "total": t,
             "lines": [{"item": item, "quantity": 1, "unit_price": price}]} for i, t in enumerate(totals)]


def test_history_is_reproducible_and_synthetic():
    first, second = generate(), generate()
    assert first == second and len(first) > 400
    assert all(row[1].startswith("H-") for row in first)  # never collides with real invoice numbers


def test_cold_start_vendor_has_no_baseline():
    findings, profile = check_amount(Invoice(vendor_name="New Co", total=999_999), hist([1000] * (MIN_HISTORY - 1)))
    assert profile is None and [f.code for f in findings] == ["HISTORY_TOO_SHORT"]


def test_amount_outlier_uses_robust_statistics():
    past = [1800, 1900, 2000, 2100, 2200] * 4 + [60_000]  # one huge past order must not widen "normal"
    flagged, _ = check_amount(Invoice(vendor_name="V", total=12_000), hist(past))
    normal, profile = check_amount(Invoice(vendor_name="V", total=2_400), hist(past))
    assert [f.code for f in flagged] == ["AMOUNT_ANOMALY"] and normal == []
    assert profile["median"] == pytest.approx(2000, rel=0.01)


def test_unusually_small_invoices_are_not_flagged():
    assert check_amount(Invoice(vendor_name="V", total=50), hist([2000, 2100, 1900] * 6))[0] == []


def test_price_drift_and_its_exclusions():
    past = hist([1000] * 10)
    creep = Invoice(line_items=[LineItem(description="WidgetA", item="WidgetA", quantity=1, unit_price=270)])
    rush = Invoice(line_items=[LineItem(description="WidgetA (rush order)", item="WidgetA", quantity=1, unit_price=300,
                                        note="rush order")])
    within = Invoice(line_items=[LineItem(description="WidgetA", item="WidgetA", quantity=1, unit_price=255)])
    assert [f.code for f in check_price_drift(creep, past)] == ["PRICE_DRIFT"]
    assert check_price_drift(rush, past) == [] and check_price_drift(within, past) == []
    assert check_price_drift(creep, hist([1000] * 3)) == []  # too few price points to judge


def test_benford_maths():
    perfect = [d + 0.5 for d in range(1, 10) for _ in range(round(BENFORD[d] * 1000))]
    assert benford(perfect)["mad"] < 0.002
    uniform = [d * 100 for d in range(1, 10) for _ in range(50)]
    assert benford(uniform)["verdict"] == "nonconformity"
    assert sum(BENFORD.values()) == pytest.approx(1.0)


def test_benford_flags_only_the_fabricated_vendor(db):
    table = vendor_benford_table(db.history_by_vendor())
    assert [r["vendor"] for r in table if r["outlier"]] == ["MegaWidgets Corp"]


def test_detectors_in_the_pipeline(pipeline):
    spike = pipeline.process(EXTRA / "invoice_2017_amount_spike.txt")
    creep = pipeline.process(EXTRA / "invoice_2018_price_creep.txt")
    rush = pipeline.process(INVOICES / "invoice_1010.txt")
    assert "AMOUNT_ANOMALY" in spike.validation.codes and spike.status == "ESCALATED"
    assert spike.validation.history_context["ratio"] > 5
    assert "PRICE_DRIFT" in creep.validation.codes and creep.status == "ESCALATED"
    assert rush.status == "PAID" and "PRICE_DRIFT" not in rush.validation.codes


def test_unknown_vendors_skip_history(db):
    assert check_history(Invoice(vendor_name="Fraudster LLC", total=100_000), db) == ([], None)


def test_investigator_history_tool(db):
    tool = next(t for t in build_validation_tools(db) if t.name == "vendor_history_summary")
    summary = tool.fn(vendor_name="Consolidated Materials Group")
    assert summary["invoices"] == 44 and summary["unit_prices"]["widgeta"]["max"] == 250.0
