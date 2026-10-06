"""Business case page: annual AP cost today vs with the system, in dollars."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from ap_autopilot.models import ProcessingOutcome
from ap_autopilot.reporting import summarize
from ap_autopilot.scorecard import load_latest
from ap_autopilot.roi import (BENCHMARKS, SCENARIOS, Assumptions, business_case, llm_cost_from_calls, scenario,
                              sensitivity, waterfall)

# Validated diverging pair (CVD-safe, passes the palette checks) plus neutral ink for totals
SAVING, COST, TOTAL, INK, MUTED = "#2F6FA3", "#C4702B", "#56627A", "#1F2A44", "#5B6578"

# (field, label, min, max, step, format, help) grouped as the page shows them
CONTROLS: dict[str, list[tuple]] = {
    "Volume": [
        ("annual_invoices", "Invoices per year", 5_000, 200_000, 5_000, "%d",
         "Assumption: the brief gives no volume. 50,000 makes the model reconcile to the stated $2M."),
        ("avg_invoice_value", "Average invoice value ($)", 500.0, 20_000.0, 500.0, "%.0f",
         "Assumption. Drives spend, so it scales leakage and early-pay discounts."),
    ],
    "Today": [
        ("manual_cost_per_invoice", "Manual cost per invoice ($)", 5.0, 40.0, 0.5, "%.2f", BENCHMARKS["manual_cost"][0]),
        ("error_rate", "Invoice error rate", 0.0, 60.0, 1.0, "%.0f%%", "From the brief: 30%."),
        ("cost_per_error", "Cost to fix one error ($)", 0.0, 100.0, 1.0, "%.0f",
         "Assumption: rework, vendor calls and re-approval for one wrong invoice."),
        ("leakage_pct_of_spend", "Spend lost to bad payments", 0.0, 1.0, 0.05, "%.2f%%",
         "Assumption: 0.1% paid in error and never recovered. " + BENCHMARKS["leakage"][0]),
        ("discount_eligible_share", "Spend offering early-pay discounts", 0.0, 50.0, 1.0, "%.0f%%",
         "Assumption: share of spend on terms like 2/10 net 30."),
        ("manual_discount_capture", "Discounts captured today", 0.0, 100.0, 5.0, "%.0f%%", BENCHMARKS["discount_capture"][0]),
    ],
    "With AP Autopilot": [
        ("touchless_share", "Decided without a person", 0.0, 100.0, 1.0, "%.0f%%",
         "Paid or rejected with no human touch. Measured 70% on the deliberately adversarial sample; "
         + BENCHMARKS["exceptions"][0] + "."),
        ("catch_rate", "Errors caught before payment", 0.0, 100.0, 1.0, "%.0f%%",
         "Assumption, set below the 100% recall on the labelled eval set on purpose."),
        ("exception_cost_per_invoice", "Cost per exception ($)", 0.0, 20.0, 0.5, "%.2f",
         "A person reviews with the findings already attached. " + BENCHMARKS["automated_cost"][0] + "."),
        ("automated_cost_per_invoice", "Automated cost per invoice ($)", 0.0, 5.0, 0.05, "%.2f", BENCHMARKS["automated_cost"][0]),
        ("llm_cost_per_invoice", "LLM cost per invoice ($)", 0.0, 1.0, 0.01, "%.2f",
         "Assumption until measured: run the inbox on Grok and switch on 'Use measured numbers'."),
        ("platform_cost_per_year", "Platform run cost per year ($)", 0.0, 300_000.0, 5_000.0, "%.0f",
         "Assumption: hosting, monitoring, rule and model upkeep."),
        ("implementation_cost", "One-time implementation ($)", 0.0, 600_000.0, 10_000.0, "%.0f",
         "Assumption: one forward-deployed engagement."),
        ("ramp_months", "Months to full adoption", 0, 12, 1, "%d", "Shadow mode first, then partial, then full automation."),
    ],
}
PERCENT_FIELDS = {"error_rate", "leakage_pct_of_spend", "discount_eligible_share", "manual_discount_capture",
                  "touchless_share", "catch_rate"}


def _money(x: Optional[float], short: bool = False) -> str:
    if x is None:
        return "n/a"
    if short and abs(x) >= 1_000_000:
        return f"{'-' if x < 0 else ''}${abs(x) / 1_000_000:.2f}M"
    if short and abs(x) >= 1_000:
        return f"{'-' if x < 0 else ''}${abs(x) / 1_000:.0f}K"
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def _load_scenario() -> None:
    chosen = scenario(st.session_state["roi_scenario"])
    for f in fields(Assumptions):
        value = getattr(chosen, f.name)
        st.session_state[f"roi_{f.name}"] = value * 100 if f.name in PERCENT_FIELDS else value


def _measured(batch: Optional[list[ProcessingOutcome]], price_in: float, price_out: float) -> dict[str, Any]:
    """Prefer a fresh inbox run; otherwise use the committed Grok scorecard (eval/results/latest.json)."""
    if not batch:
        card = load_latest()
        if not card:
            return {}
        run = card["llm_runs"][0]
        return {"touchless_share": run["touchless_rate"], "invoices": run["invoices"], "source": f"scorecard ({card['llm']})",
                "llm_cost_per_invoice": run["cost_per_invoice_usd"] or None}
    s = summarize(batch)
    calls = [c for o in batch for c in o.llm_calls]
    return {"touchless_share": s["touchless_rate"], "invoices": s["invoices"], "source": "last inbox run",
            "llm_cost_per_invoice": llm_cost_from_calls(calls, s["invoices"], price_in, price_out)}


def waterfall_chart(a: Assumptions) -> alt.Chart:
    df = pd.DataFrame(waterfall(a))
    short = {"Cost today": "Today", "Less handling labour": "Labour", "Fewer errors to fix": "Rework",
             "Bad payments stopped": "Bad payments", "Discounts captured": "Discounts",
             "Platform run cost": "Platform", "Cost with system": "With system"}
    df["label"] = df["label"].map(lambda l: short.get(l, l))
    df["lo"], df["hi"] = df[["start", "end"]].min(axis=1), df[["start", "end"]].max(axis=1)
    df["amount"] = df["value"].map(lambda v: _money(v, short=True))
    df["order"] = range(len(df))
    kind_scale = alt.Scale(domain=["total", "saving", "cost"], range=[TOTAL, SAVING, COST])
    x = alt.X("label:N", sort=alt.SortField("order"), title=None,
              axis=alt.Axis(labelAngle=0, labelLimit=110, labelOverlap=False, labelColor=MUTED, domainColor="#C9D3CC", ticks=False))
    bars = alt.Chart(df).mark_bar(cornerRadius=4, size=46).encode(
        x=x, y=alt.Y("lo:Q", title=None, axis=alt.Axis(format="$~s", gridColor="#E3E8E4", labelColor=MUTED, domain=False)),
        y2="hi:Q",
        color=alt.Color("kind:N", scale=kind_scale, legend=alt.Legend(title=None, orient="top", labelColor=INK,
                        labelExpr="{'total':'Total cost','saving':'Saving','cost':'Added cost'}[datum.label]")),
        tooltip=[alt.Tooltip("label:N", title="Step"), alt.Tooltip("amount:N", title="Amount")],
    )
    labels = alt.Chart(df).mark_text(dy=-8, fontSize=12, color=INK, fontWeight=600).encode(
        x=x, y="hi:Q", text="amount:N")
    return (bars + labels).properties(height=320).configure_view(stroke=None)


def sensitivity_chart(a: Assumptions) -> alt.Chart:
    rows = sensitivity(a)
    df = pd.DataFrame(rows)
    df["lo"], df["hi"] = df[["savings_at_low", "savings_at_high"]].min(axis=1), df[["savings_at_low", "savings_at_high"]].max(axis=1)
    df["range"] = df.apply(lambda r: f"{_money(r['lo'], True)} to {_money(r['hi'], True)}", axis=1)

    def fmt(name: str, v: float) -> str:
        if name in PERCENT_FIELDS:
            return f"{v:.1%}" if name == "leakage_pct_of_spend" else f"{v:.0%}"
        return f"{v:,.0f}" if v >= 100 else f"${v:,.2f}"

    df["inputs"] = df.apply(lambda r: f"{fmt(r['assumption'], r['low_input'])} to {fmt(r['assumption'], r['high_input'])}", axis=1)
    y = alt.Y("label:N", sort=list(df["label"]), title=None, axis=alt.Axis(labelColor=INK, labelLimit=210, ticks=False, domain=False))
    bars = alt.Chart(df).mark_bar(cornerRadius=4, size=16, color=SAVING).encode(
        y=y, x=alt.X("lo:Q", title="Annual savings", axis=alt.Axis(format="$~s", gridColor="#E3E8E4", labelColor=MUTED)),
        x2="hi:Q",
        tooltip=[alt.Tooltip("label:N", title="Assumption"), alt.Tooltip("inputs:N", title="Input range"),
                 alt.Tooltip("range:N", title="Annual savings")],
    )
    base_df = pd.DataFrame({"base": [rows[0]["base"]], "text": [f"Base {_money(rows[0]['base'], True)}"]})
    base = alt.Chart(base_df).mark_rule(color=INK, strokeDash=[4, 3], strokeWidth=1.5).encode(x="base:Q")
    base_label = alt.Chart(base_df).mark_text(align="left", dx=4, dy=-6, fontSize=11, color=INK).encode(
        x="base:Q", y=alt.value(0), text="text:N")
    return (bars + base + base_label).properties(height=320).configure_view(stroke=None)


def render(batch: Optional[list[ProcessingOutcome]]) -> None:
    st.markdown("## Business case")
    st.markdown('<div class="sub">Annual cost of accounts payable today vs with AP Autopilot. Every number is adjustable; '
                'sources and assumptions are listed under each control and at the bottom.</div>', unsafe_allow_html=True)

    if "roi_scenario" not in st.session_state:
        st.session_state["roi_scenario"] = "Base"
        _load_scenario()
    st.segmented_control("Scenario", list(SCENARIOS), key="roi_scenario", on_change=_load_scenario,
                         help="Conservative uses the pessimistic end of each benchmark range.")

    with st.expander("Assumptions", expanded=False):
        cols = st.columns(len(CONTROLS), gap="large")
        for col, (group, controls) in zip(cols, CONTROLS.items()):
            with col:
                st.markdown(f"**{group}**")
                for name, label, lo, hi, step, fmt, help_ in controls:
                    st.slider(label, min_value=lo, max_value=hi, step=step, format=fmt, key=f"roi_{name}", help=help_)
        st.markdown("**Measured by the system**")
        m1, m2, m3 = st.columns([2, 1, 1])
        has_measure = bool(batch) or load_latest() is not None
        use_measured = m1.toggle("Use measured numbers", value=False, disabled=not has_measure,
                                 help="From the last inbox run, or the committed Grok scorecard. Replaces 'Decided without a person' and the LLM cost with what the system actually did.")
        price_in = m2.number_input("Model $ per 1M input tokens", value=3.0, min_value=0.0, step=0.5,
                                   help="Check the current price on xAI's pricing page.")
        price_out = m3.number_input("Model $ per 1M output tokens", value=15.0, min_value=0.0, step=0.5)
        if not has_measure:
            st.caption("Run the inbox, or `python main.py --compare` with a Grok key, to measure touchless share and LLM cost.")

    def widget_value(name: str) -> Any:
        default = getattr(Assumptions(), name)
        if f"roi_{name}" not in st.session_state:
            return default
        value = st.session_state[f"roi_{name}"]
        return value / 100 if name in PERCENT_FIELDS else value

    a = Assumptions(**{f.name: widget_value(f.name) for f in fields(Assumptions)})
    measured = _measured(batch, price_in, price_out)
    if measured and use_measured:
        updates = {"touchless_share": measured["touchless_share"]}
        if measured.get("llm_cost_per_invoice") is not None:
            updates["llm_cost_per_invoice"] = measured["llm_cost_per_invoice"]
        a = replace(a, **updates)
        note = f"Using measured touchless share {measured['touchless_share']:.0%} from {measured['invoices']} invoices ({measured['source']})"
        note += (f" and LLM cost ${measured['llm_cost_per_invoice']:.3f} per invoice." if measured.get("llm_cost_per_invoice") is not None
                 else ". LLM cost stays assumed: that run made no Grok calls.")
        st.info(note)

    case = business_case(a)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual savings", _money(case["annual_savings"], short=True))
    k1.caption(f"{case['savings_pct']:.0%} of today's AP cost")
    k2.metric("Payback", f"{case['payback_months']} months" if case["payback_months"] is not None else "Never")
    k2.caption(f"on {_money(a.implementation_cost, True)} implementation, {a.ramp_months}-month ramp")
    k3.metric("Three-year net", _money(case["three_year_net"], short=True))
    k3.caption("savings minus implementation")
    k4.metric("Staff time freed", f"{case['fte_freed']:.1f} FTE")
    k4.caption(f"{case['hours_freed']:,.0f} hours a year to redeploy")

    r = case["reconciliation"]
    verdict = {"reconciles": "reconcile with", "under-explains": "fall short of", "over-explains": "overshoot"}[r["verdict"]]
    esc = lambda t: t.replace("$", "\\$")  # noqa: E731  (Streamlit treats $...$ as maths)
    st.markdown(f"These assumptions put today's cost at **{esc(_money(r['explained']))}**, which would {verdict} the "
                f"**{esc(_money(r['stated']))}** loss in the brief ({r['gap_pct']:+.0%}). "
                + ("A business case should explain the client's own number before it claims savings."
                   if r["verdict"] != "reconciles" else ""))

    left, right = st.columns([7, 5], gap="large")
    with left:
        st.markdown("#### From today's cost to the new cost")
        st.altair_chart(waterfall_chart(a), width="stretch")
    with right:
        st.markdown("#### What moves the answer")
        st.altair_chart(sensitivity_chart(a), width="stretch")
        top = sensitivity(a)[0]
        st.caption(f"Each bar is the savings range when one assumption moves across its benchmark range, others held. "
                   f"Pin down **{top['label'].lower()}** first in discovery.")

    with st.expander("Cost breakdown table"):
        names = {"labour": "Handling labour", "rework": "Fixing errors", "leakage": "Bad payments",
                 "missed_discounts": "Missed early-pay discounts", "platform": "Platform run cost"}
        rows = [{"Cost": names[k], "Today": _money(case["today"].get(k, 0.0)), "With system": _money(case["future"].get(k, 0.0))}
                for k in names]
        rows.append({"Cost": "Total", "Today": _money(case["today_total"]), "With system": _money(case["future_total"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Labour savings are capacity: they become cash when people move to higher-value work or volume grows without hiring.")

    with st.expander("Sources"):
        for text, url in BENCHMARKS.values():
            st.markdown(f"- {text} ([source]({url}))")
