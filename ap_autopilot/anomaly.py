"""History-based anomaly detection: what is unusual *for this vendor*.

Rules catch what is wrong; history catches what is out of character. Every
detector is explainable in one sentence a reviewer can check by hand.

* Amount outlier: robust z-score of log(total) against the vendor's history,
  using median and MAD (median absolute deviation), so one large past order
  can't distort "normal". Only the high side is flagged: an unusually small
  invoice is not a loss risk.
* Price drift: the billed unit price against everything this vendor has
  charged for the item. Catches creep that stays under the 10% catalog check.
* Benford's law: first-digit distribution of amounts vs the log10(1 + 1/d)
  expectation, scored with Nigrini's MAD. One vendor's invoices rarely span
  enough orders of magnitude to conform in absolute terms, so vendors are
  compared with their peers: a vendor deviating far more than the typical
  vendor is the signal (invented numbers cluster). Vendor-level only: a single
  invoice carries no statistical signal.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any, Optional

from .models import Finding, Invoice, Severity
from .normalize import normalize_sku

MIN_HISTORY = 12             # below this a vendor has no reliable "normal" yet
AMOUNT_Z_THRESHOLD = 3.5     # robust z (Iglewicz and Hoaglin's recommended outlier cut-off)
MIN_PRICE_POINTS = 5
PRICE_DRIFT_PCT = 0.03       # above the highest price ever charged by more than 3%
_PREMIUM = re.compile(r"\b(rush|expedit\w*|express|overnight|priority)\b", re.I)

BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
# Nigrini (2012) first-digit MAD conformity bands
BENFORD_BANDS = [(0.006, "close conformity"), (0.012, "acceptable conformity"),
                 (0.015, "marginal conformity"), (float("inf"), "nonconformity")]


def _money(x: float) -> str:
    return f"${x:,.0f}"


def amount_profile(totals: list[float]) -> Optional[dict[str, Any]]:
    values = [t for t in totals if t and t > 0]
    if len(values) < MIN_HISTORY:
        return None
    logs = [math.log(v) for v in values]
    med = statistics.median(logs)
    mad = statistics.median(abs(x - med) for x in logs) or 1e-9
    return {"n": len(values), "median": math.exp(med), "log_median": med, "log_mad": mad,
            "p95": sorted(values)[int(0.95 * (len(values) - 1))],
            # the range a reviewer can eyeball: totals whose robust z is within the threshold
            "normal_high": math.exp(med + AMOUNT_Z_THRESHOLD * mad / 0.6745)}


def robust_z(total: float, profile: dict[str, Any]) -> float:
    return 0.6745 * (math.log(total) - profile["log_median"]) / profile["log_mad"]


def check_amount(inv: Invoice, history: list[dict[str, Any]]) -> tuple[list[Finding], Optional[dict[str, Any]]]:
    profile = amount_profile([h["total"] for h in history])
    if profile is None:
        return ([Finding(code="HISTORY_TOO_SHORT", severity=Severity.INFO,
                         message=f"Only {len(history)} past invoices from this vendor; no amount baseline yet.")]
                if inv.vendor_name else []), None
    if not inv.total or inv.total <= 0:
        return [], profile
    z = robust_z(inv.total, profile)
    profile = {**profile, "z": round(z, 2), "ratio": inv.total / profile["median"]}
    if z > AMOUNT_Z_THRESHOLD:
        return [Finding(code="AMOUNT_ANOMALY", severity=Severity.WARNING,
                        evidence={"z": round(z, 1), "median": round(profile["median"], 2), "n": profile["n"]},
                        message=f"{_money(inv.total)} is {inv.total / profile['median']:.1f}x this vendor's median of "
                                f"{_money(profile['median'])} over {profile['n']} invoices, well above its normal range "
                                f"(up to {_money(profile['normal_high'])}). Confirm the order with purchasing.")], profile
    return [], profile


def price_history(history: list[dict[str, Any]]) -> dict[str, list[float]]:
    prices: dict[str, list[float]] = {}
    for h in history:
        for line in h["lines"]:
            prices.setdefault(normalize_sku(line["item"]), []).append(float(line["unit_price"]))
    return prices


def check_price_drift(inv: Invoice, history: list[dict[str, Any]]) -> list[Finding]:
    out: list[Finding] = []
    past = price_history(history)
    for li in inv.line_items:
        if li.unit_price is None or _PREMIUM.search(" ".join(filter(None, [li.note, li.description]))):
            continue  # labelled premiums are judged by the catalog rule, not as drift
        prices = past.get(normalize_sku(li.item or li.description), [])
        if len(prices) < MIN_PRICE_POINTS:
            continue
        ceiling = max(prices)
        if li.unit_price > ceiling * (1 + PRICE_DRIFT_PCT):
            usual = statistics.median(prices)
            same = sum(1 for p in prices if abs(p - usual) < 0.005)
            habit = (f"charged {_money(usual)} on all {len(prices)} previous lines" if same == len(prices)
                     else f"has charged at most {_money(ceiling)} across {len(prices)} previous lines")
            out.append(Finding(code="PRICE_DRIFT", severity=Severity.WARNING, item=li.item,
                               evidence={"billed": li.unit_price, "historical_max": ceiling, "n": len(prices)},
                               message=f"{li.item}: billed at {_money(li.unit_price)} ({li.unit_price / usual - 1:+.0%}); "
                                       f"this vendor {habit}. Ask for the price change in writing."))
    return out


def benford(amounts: list[float]) -> Optional[dict[str, Any]]:
    digits = [int(str(f"{a:.2f}").lstrip("0.")[0]) for a in amounts if a and a >= 1]
    if not digits:
        return None
    counts = Counter(digits)
    n = len(digits)
    observed = {d: counts.get(d, 0) / n for d in range(1, 10)}
    mad = sum(abs(observed[d] - BENFORD[d]) for d in range(1, 10)) / 9
    verdict = next(label for limit, label in BENFORD_BANDS if mad <= limit)
    return {"n": n, "observed": observed, "expected": BENFORD, "mad": round(mad, 4), "verdict": verdict,
            "reliable": n >= 100}


PEER_OUTLIER_RATIO = 1.75   # deviation vs the median vendor's deviation
MIN_BENFORD_INVOICES = 30


def vendor_benford_table(history_by_vendor: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Benford deviation per vendor, judged against peers rather than in absolute terms."""
    rows = []
    for vendor, history in history_by_vendor.items():
        b = benford([h["total"] for h in history])
        if b and b["n"] >= MIN_BENFORD_INVOICES:
            rows.append({"vendor": vendor, "invoices": b["n"], "mad": b["mad"], "observed": b["observed"]})
    if not rows:
        return []
    typical = statistics.median(r["mad"] for r in rows)
    for r in rows:
        r["vs_peers"] = round(r["mad"] / typical, 2) if typical else 1.0
        r["outlier"] = r["vs_peers"] >= PEER_OUTLIER_RATIO
    return sorted(rows, key=lambda r: r["mad"], reverse=True)
