"""Rule and policy behaviour on hand-built invoices: one scenario per test."""

from datetime import date

from ap_autopilot.models import Decision, Finding, Invoice, LineItem, Severity, ValidationReport
from ap_autopilot.policy import enforce
from ap_autopilot.rules import (check_bank_change, check_inventory, check_pricing, check_totals, check_vendor,
                                score_fraud)


def inv(**kw):
    base = dict(invoice_number="INV-1", vendor_name="Widgets Inc.", invoice_date=date(2026, 1, 1),
                due_date=date(2026, 2, 1), total=1000.0, line_items=[LineItem(description="WidgetA", item="WidgetA", quantity=4, unit_price=250)])
    base.update(kw)
    return Invoice(**base)


def codes(findings):
    return {f.code for f in findings}


def test_stock_is_checked_against_the_sum_of_split_lines(db):
    lines = [LineItem(description="WidgetA", item="WidgetA", quantity=10, unit_price=250),
             LineItem(description="WidgetA", item="WidgetA", quantity=6, unit_price=240)]
    findings, _ = check_inventory(inv(line_items=lines), db)
    stock = [f for f in findings if f.code == "STOCK_EXCEEDED"]
    assert stock and stock[0].evidence == {"requested": 16, "available": 15, "lines": 2}


def test_unknown_sku_is_never_auto_matched(db):
    findings, matches = check_inventory(inv(line_items=[LineItem(description="WidgetC", item="WidgetC", quantity=1, unit_price=350)]), db)
    assert "UNKNOWN_ITEM" in codes(findings) and matches["WidgetC"] is None


def test_formatting_variants_do_match(db):
    findings, matches = check_inventory(inv(line_items=[LineItem(description="Gadget X", item="Gadget X", quantity=1, unit_price=750)]), db)
    assert matches["Gadget X"] == "GadgetX" and not findings


def test_zero_stock_item_is_critical(db):
    findings, _ = check_inventory(inv(line_items=[LineItem(description="FakeItem", item="FakeItem", quantity=1, unit_price=1)]), db)
    assert any(f.code == "ZERO_STOCK_ITEM" and f.severity == Severity.CRITICAL for f in findings)


def test_total_mismatch_detected():
    findings = check_totals(inv(subtotal=1000, tax_amount=60, total=1110), tol=0.05)
    assert "TOTAL_MISMATCH" in codes(findings)


def test_price_premium_needs_a_reason(db):
    rush = LineItem(description="WidgetA (rush order)", item="WidgetA", quantity=1, unit_price=300, note="rush order")
    silent = LineItem(description="WidgetA", item="WidgetA", quantity=1, unit_price=300)
    assert "PRICE_PREMIUM_EXPLAINED" in codes(check_pricing(inv(line_items=[rush]), db, 0.10))
    assert "PRICE_ABOVE_CATALOG" in codes(check_pricing(inv(line_items=[silent]), db, 0.10))


def test_vendor_rebrand_of_known_vendor(db):
    findings, vendor = check_vendor(inv(vendor_name="QuickShip Distributers", notes="Vendor annotation: (formerly FastShip Ltd.)"), db)
    assert codes(findings) == {"VENDOR_NAME_CHANGE"} and findings[0].evidence["former_in_master"]


def test_bank_change_language():
    assert check_bank_change(inv(notes="Please note our bank details have changed. Remit to new account 123."))
    assert not check_bank_change(inv(notes="Deliver to dock B."))


def test_fraud_score_is_explainable(settings):
    invoice = inv(notes="URGENT pay immediately, wire transfer preferred", payment_terms="Immediate", total=100000)
    findings = [Finding(code="ZERO_STOCK_ITEM", severity=Severity.CRITICAL, message=""),
                Finding(code="UNKNOWN_VENDOR", severity=Severity.WARNING, message="")]
    score, signals, extra = score_fraud(invoice, findings, settings)
    assert score == 100 and "FRAUD_RISK_HIGH" in codes(extra)
    assert any("urgency" in s for s in signals)


# ------------------------------------------------------------------ policy
def report(*items):
    return ValidationReport(findings=[Finding(code=c, severity=s, message=c) for c, s in items])


def test_guardrail_blocks_rogue_approval_of_critical_invoice(settings):
    d, overrides = enforce(Decision(decision="APPROVE", rationale="looks fine"), inv(),
                           report(("ZERO_STOCK_ITEM", Severity.CRITICAL)), settings)
    assert d.decision == "REJECT" and overrides


def test_guardrail_escalates_large_invoice_with_warnings(settings):
    d, _ = enforce(Decision(decision="APPROVE", rationale="benign"), inv(total=15000),
                   report(("STOCK_EXCEEDED", Severity.WARNING)), settings)
    assert d.decision == "ESCALATE"


def test_guardrail_allows_small_invoice_with_explained_warning(settings):
    d, overrides = enforce(Decision(decision="APPROVE", rationale="premium explained"), inv(total=500),
                           report(("PRICE_ABOVE_CATALOG", Severity.WARNING)), settings)
    assert d.decision == "APPROVE" and not overrides


def test_guardrail_blocks_foreign_currency_auto_payment(settings):
    d, _ = enforce(Decision(decision="APPROVE", rationale="ok"), inv(currency="EUR"), report(), settings)
    assert d.decision == "ESCALATE"


def test_agent_cannot_reject_clean_invoice_without_evidence(settings):
    d, _ = enforce(Decision(decision="REJECT", rationale="gut feeling"), inv(), report(), settings)
    assert d.decision == "ESCALATE"
