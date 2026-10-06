"""Business case maths."""

from dataclasses import replace

import pytest

from ap_autopilot.roi import (SCENARIOS, Assumptions, business_case, from_measurements, llm_cost_from_calls,
                              payback_months, reconcile, scenario, sensitivity, today_costs, waterfall)


def test_defaults_reconcile_to_the_briefs_two_million():
    case = business_case(Assumptions())
    assert case["reconciliation"]["verdict"] == "reconciles"
    assert abs(case["today_total"] - 2_000_000) / 2_000_000 <= 0.10


def test_today_costs_by_hand():
    a = Assumptions(annual_invoices=1000, avg_invoice_value=1000, manual_cost_per_invoice=20, error_rate=0.3,
                    cost_per_error=10, leakage_pct_of_spend=0.01, discount_eligible_share=0.5, discount_rate=0.02,
                    manual_discount_capture=0.25)
    assert today_costs(a) == {"labour": 20_000, "rework": 3_000, "leakage": 10_000, "missed_discounts": 7_500}


def test_waterfall_bars_chain_from_today_to_future():
    a = Assumptions()
    steps = waterfall(a)
    case = business_case(a)
    assert steps[0]["value"] == pytest.approx(case["today_total"])
    assert steps[-1]["value"] == pytest.approx(case["future_total"])
    for prev, step in zip(steps[:-2], steps[1:-1]):
        assert step["start"] == pytest.approx(prev["end"])       # each bar starts where the last ended
    assert steps[-2]["end"] == pytest.approx(steps[-1]["value"])  # the chain lands on the final total


def test_payback_accounts_for_the_ramp():
    assert payback_months(120_000, 1_200_000, ramp_months=0) == pytest.approx(1.2)
    assert payback_months(120_000, 1_200_000, ramp_months=3) > 1.2
    assert payback_months(100, 0, 3) is None


def test_full_automation_and_zero_volume_edges():
    full = business_case(replace(Assumptions(), touchless_share=1.0))
    assert full["future"]["labour"] == pytest.approx(50_000 * (0.50 + 0.05))
    empty = business_case(replace(Assumptions(), annual_invoices=0))
    assert empty["annual_savings"] < 0 and empty["payback_months"] is None  # platform cost with nothing to save


def test_every_scenario_is_valid_and_ordered():
    savings = {n: business_case(scenario(n))["annual_savings"] for n in SCENARIOS}
    assert savings["Conservative"] < savings["Base"] < savings["Optimistic"]
    assert savings["Conservative"] > 0


def test_sensitivity_is_sorted_by_swing_and_brackets_base():
    rows = sensitivity(Assumptions())
    assert [r["swing"] for r in rows] == sorted((r["swing"] for r in rows), reverse=True)
    for r in rows:
        assert min(r["savings_at_low"], r["savings_at_high"]) <= r["base"] + 1e-6
        assert max(r["savings_at_low"], r["savings_at_high"]) >= r["base"] - 1e-6


def test_reconcile_verdicts():
    assert reconcile(2_100_000)["verdict"] == "reconciles"
    assert reconcile(1_000_000)["verdict"] == "under-explains"
    assert reconcile(3_000_000)["verdict"] == "over-explains"


def test_measurements_override_defaults():
    a = from_measurements(Assumptions(), touchless_share=1.4, llm_cost_per_invoice=0.08)
    assert a.touchless_share == 1.0 and a.llm_cost_per_invoice == 0.08


def test_llm_cost_from_recorded_token_usage():
    calls = [{"provider": "grok", "prompt_tokens": 2_000_000, "completion_tokens": 500_000},
             {"provider": "offline", "prompt_tokens": None, "completion_tokens": None}]
    assert llm_cost_from_calls(calls, invoices=100, usd_per_m_input=3, usd_per_m_output=15) == pytest.approx(0.135)
    assert llm_cost_from_calls([{"provider": "offline"}], 10, 3, 15) is None
