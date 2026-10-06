"""Vendor insights: what "normal" looks like for each vendor, from 12 months of history.

The history is SYNTHETIC (see ap_autopilot/history.py) and labelled as such on the page.
"""

from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from ap_autopilot.anomaly import BENFORD, amount_profile, benford, price_history, vendor_benford_table
from ap_autopilot.db import Database
from ap_autopilot.models import ProcessingOutcome

# Same validated CVD-safe pair as the business case page, plus neutral ink
HISTORY, CURRENT, INK, MUTED, BAND = "#2F6FA3", "#C4702B", "#1F2A44", "#5B6578", "#E3E8E4"


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _current_for(vendor: str, outcomes: list[ProcessingOutcome]) -> list[dict]:
    """Invoices processed this session for this vendor, to plot against its history."""
    rows = []
    for o in outcomes:
        inv = o.extraction.invoice if o.extraction else None
        if inv and inv.vendor_name and inv.total and inv.invoice_date and \
                inv.vendor_name.lower().rstrip(".") == vendor.lower().rstrip("."):
            rows.append({"invoice_date": str(inv.invoice_date), "total": inv.total,
                         "label": f"{inv.invoice_number or o.source_file} ({o.status.lower()})",
                         "lines": [{"item": li.item or li.description, "price": li.unit_price} for li in inv.line_items
                                   if li.unit_price is not None]})
    return rows


def spend_chart(history: list[dict], profile: dict, current: list[dict]) -> alt.Chart:
    df = pd.DataFrame([{"invoice_date": h["invoice_date"], "total": h["total"], "label": h["invoice_number"]} for h in history])
    band = alt.Chart(pd.DataFrame({"lo": [0], "hi": [profile["normal_high"]]})).mark_rect(color=BAND, opacity=0.7).encode(
        y="lo:Q", y2="hi:Q")
    median = alt.Chart(pd.DataFrame({"m": [profile["median"]]})).mark_rule(color=INK, strokeDash=[4, 3], strokeWidth=1.5).encode(y="m:Q")
    x = alt.X("invoice_date:T", title=None, axis=alt.Axis(format="%b %Y", labelColor=MUTED, grid=False))
    y = alt.Y("total:Q", title="Invoice total", axis=alt.Axis(format="$~s", labelColor=MUTED, gridColor="#EEF1EE"))
    dots = alt.Chart(df).mark_circle(size=64, color=HISTORY, opacity=0.85, stroke="white", strokeWidth=1).encode(
        x=x, y=y, tooltip=[alt.Tooltip("label:N", title="Invoice"), alt.Tooltip("invoice_date:T", title="Date"),
                           alt.Tooltip("total:Q", title="Total", format="$,.0f")])
    layers = [band, median, dots]
    if current:
        cdf = pd.DataFrame(current)
        layers.append(alt.Chart(cdf).mark_point(shape="diamond", size=320, filled=True, color=CURRENT, stroke="white",
                                                strokeWidth=1.5, opacity=1).encode(
            x="invoice_date:T", y="total:Q", tooltip=[alt.Tooltip("label:N", title="Processed now"),
                                                      alt.Tooltip("total:Q", title="Total", format="$,.0f")]))
        layers.append(alt.Chart(cdf).mark_text(align="right", dx=-14, fontSize=12, fontWeight=600, color=INK).encode(
            x="invoice_date:T", y="total:Q", text="label:N"))
    return alt.layer(*layers).properties(height=300).configure_view(stroke=None)


def price_chart(history: list[dict], sku: str, current: list[dict]) -> alt.Chart:
    rows = [{"invoice_date": h["invoice_date"], "price": line["unit_price"], "invoice": h["invoice_number"]}
            for h in history for line in h["lines"] if line["item"].lower() == sku.lower()]
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=40, color=HISTORY), color=HISTORY, strokeWidth=2,
                                   interpolate="step-after").encode(
        x=alt.X("invoice_date:T", title=None, axis=alt.Axis(format="%b %Y", labelColor=MUTED, grid=False)),
        y=alt.Y("price:Q", title="Unit price", scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(format="$,.0f", labelColor=MUTED, gridColor="#EEF1EE")),
        tooltip=[alt.Tooltip("invoice:N", title="Invoice"), alt.Tooltip("price:Q", title="Price", format="$,.2f")],
    )
    now = [c for c in current if c["item"].lower() == sku.lower()]
    if now:
        cdf = pd.DataFrame(now)
        line = alt.layer(line, alt.Chart(cdf).mark_point(shape="diamond", size=320, filled=True, color=CURRENT, stroke="white",
                                                         strokeWidth=1.5, opacity=1).encode(
            x="invoice_date:T", y="price:Q", tooltip=[alt.Tooltip("label:N", title="Processed now"),
                                                      alt.Tooltip("price:Q", title="Price", format="$,.2f")]),
            alt.Chart(cdf).mark_text(align="right", dx=-14, fontSize=12, fontWeight=600, color=INK).encode(
                x="invoice_date:T", y="price:Q", text="label:N"))
    return line.properties(height=220).configure_view(stroke=None)


def benford_chart(observed: dict[int, float]) -> alt.Chart:
    df = pd.DataFrame([{"digit": str(d), "Observed": observed[d], "Benford": BENFORD[d]} for d in range(1, 10)])
    bars = alt.Chart(df).mark_bar(color=HISTORY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=22).encode(
        x=alt.X("digit:N", title="First digit of the amount", axis=alt.Axis(labelAngle=0, labelColor=MUTED)),
        y=alt.Y("Observed:Q", title="Share of invoices", axis=alt.Axis(format="%", labelColor=MUTED, gridColor="#EEF1EE")),
        tooltip=[alt.Tooltip("digit:N", title="Digit"), alt.Tooltip("Observed:Q", format=".1%"),
                 alt.Tooltip("Benford:Q", title="Benford expects", format=".1%")])
    expected = alt.Chart(df).mark_tick(color=INK, thickness=3, size=30).encode(x="digit:N", y="Benford:Q")
    return (bars + expected).properties(height=220).configure_view(stroke=None)


def render(db: Database, outcomes: Optional[list[ProcessingOutcome]]) -> None:
    st.markdown("## Vendor insights")
    st.markdown('<div class="sub">What normal looks like for each vendor, from its past 12 months of invoices. '
                'The anomaly checks compare every new invoice against this. History here is <b>synthetic</b> '
                '(the brief has none); in production it is read from the ERP.</div>', unsafe_allow_html=True)
    by_vendor = db.history_by_vendor()
    if not by_vendor:
        st.info("No history loaded. Reset demo data to generate it.")
        return
    vendors = sorted(by_vendor)
    vendor = st.selectbox("Vendor", vendors, index=vendors.index("Acme Industrial Supplies") if "Acme Industrial Supplies" in vendors else 0)
    history = by_vendor[vendor]
    profile = amount_profile([h["total"] for h in history])
    current = _current_for(vendor, outcomes or [])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Invoices, last 12 months", len(history))
    k2.metric("Typical invoice", _money(profile["median"]) if profile else "n/a")
    k3.metric("Top of normal range", _money(profile["normal_high"]) if profile else "n/a")
    k4.metric("Spend, last 12 months", _money(sum(h["total"] for h in history)))

    st.markdown("#### Invoice amounts over time")
    if profile:
        st.altair_chart(spend_chart(history, profile, current), width="stretch")
        note = ("Shaded: the normal range. Dashed: the typical invoice. "
                + ("Orange diamonds: invoices processed in this session." if current else
                   "Run the inbox to see this session's invoices plotted against the history."))
        st.caption(note)

    prices = price_history(history)
    if prices:
        st.markdown("#### Unit prices charged")
        skus = sorted({line["item"] for h in history for line in h["lines"]})
        sku = st.segmented_control("Item", skus, default=skus[0], key="vi_sku") or skus[0]
        current_prices = [{"invoice_date": c["invoice_date"], "item": ln["item"], "price": ln["price"], "label": c["label"]}
                          for c in current for ln in c["lines"]]
        st.altair_chart(price_chart(history, sku, current_prices), width="stretch")
        st.caption("A new invoice priced more than 3% above the highest price ever charged is held as price drift."
                   + (" Orange diamond: this session's invoice." if any(cp["item"].lower() == sku.lower() for cp in current_prices) else ""))

    st.markdown("#### Benford's law: do the amounts look naturally occurring?")
    left, right = st.columns([5, 4], gap="large")
    table = vendor_benford_table(by_vendor)
    with left:
        scope = st.segmented_control("Show", ["This vendor", "All vendors"], default="This vendor", key="vi_scope") or "This vendor"
        amounts = [h["total"] for h in history] if scope == "This vendor" else [h["total"] for rows in by_vendor.values() for h in rows]
        b = benford(amounts)
        st.altair_chart(benford_chart(b["observed"]), width="stretch")
        st.caption("Bars: how often each first digit appears. Dark ticks: what Benford's law expects. "
                   "Invented numbers tend to cluster on a few digits.")
    with right:
        st.markdown("**Deviation compared with peers**")
        st.dataframe(pd.DataFrame([{"Vendor": r["vendor"], "Invoices": r["invoices"],
                                    "Deviation": f"{r['vs_peers']:.1f}x typical" + ("  (audit)" if r["outlier"] else "")}
                                   for r in table]),
                     hide_index=True, width="stretch",
                     column_config={"Vendor": st.column_config.TextColumn(width="medium"),
                                    "Invoices": st.column_config.NumberColumn(width="small")})
        st.caption("One vendor's invoices rarely span enough orders of magnitude to follow Benford exactly, so each vendor "
                   "is compared with the typical vendor. 1.75x or more is flagged for a periodic audit, never to block an invoice.")
