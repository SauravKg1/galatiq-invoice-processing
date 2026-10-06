"""Business case: what the system is worth to Acme, in dollars.

Pure functions, no I/O, fully tested. Every default is either a figure from
the case brief, a published benchmark (cited in BENCHMARKS), or a labelled
assumption meant to be replaced with Acme's real numbers in discovery.

Model
-----
Today's annual cost of accounts payable =
    labour        invoices x manual cost per invoice
  + rework        invoices x error rate x cost to fix one error
  + leakage       spend x share lost to duplicate / erroneous payments
  + discounts     early-payment discounts offered but missed

With the system =
    touchless     invoices x touchless share x automated cost (incl. LLM)
  + exceptions    invoices x (1 - touchless share) x assisted cost per exception
  + rework        today's rework x (1 - catch rate)
  + leakage       today's leakage x (1 - catch rate)
  + discounts     missed at the automated capture rate
  + platform      annual run cost (hosting, monitoring, model upkeep)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

STATED_ANNUAL_LOSS = 2_000_000   # from the case brief
STATED_ERROR_RATE = 0.30         # from the case brief
STATED_DELAY_DAYS = 5            # from the case brief
WORK_HOURS_PER_FTE = 1_800

BENCHMARKS = {
    "manual_cost": ("APQC: median cost per invoice $21.40; top quartile $10.18",
                    "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "manual_cost_range": ("Ardent Partners (AP Metrics That Matter 2025): $15 to $40 for primarily manual AP",
                          "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "automated_cost": ("Fully automated under $1 per invoice; semi-automated $3 to $5",
                       "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "minutes": ("Manual handling 8 to 15 minutes per invoice; semi-automated 3 to 6",
                "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "discount_capture": ("Ardent Partners: manual AP captures 20 to 30% of early-payment discounts; automated above 80%",
                         "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "exceptions": ("Industry median: about 22% of invoices need manual intervention",
                   "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
    "leakage": ("PRGX: duplicate and erroneous payments affect 0.8% to over 2% of annual disbursements",
                "https://www.prgx.com/guides/ap-recovery-audit-services-guide/"),
    "staffing": ("IOFM: average AP team processes about 4,200 invoices per FTE a year",
                 "https://www.lido.app/blog/invoice-processing-cost-benchmarks"),
}


@dataclass(frozen=True)
class Assumptions:
    # Volume (assumption: the brief gives none; chosen so the model reconciles to the stated $2M)
    annual_invoices: int = 50_000
    avg_invoice_value: float = 4_000.0
    # Today (brief + benchmarks)
    manual_cost_per_invoice: float = 21.40       # APQC median
    error_rate: float = STATED_ERROR_RATE        # brief
    cost_per_error: float = 25.0                 # assumption: rework, vendor calls, re-approval
    leakage_pct_of_spend: float = 0.001          # assumption: 0.1% paid and never recovered (PRGX: 0.8-2% affected)
    discount_eligible_share: float = 0.10        # assumption: share of spend offered early-pay terms
    discount_rate: float = 0.02                  # "2/10 net 30", the standard early-pay term
    manual_discount_capture: float = 0.25        # Ardent: 20-30%
    manual_minutes_per_invoice: float = 10.0     # benchmark: 8-15
    # With the system
    touchless_share: float = 0.75                # production assumption; measured 70% on the adversarial sample
    automated_cost_per_invoice: float = 0.50     # benchmark: under $1 fully automated
    llm_cost_per_invoice: float = 0.05           # assumption until measured with Grok (tokens x price)
    exception_cost_per_invoice: float = 4.00     # benchmark: $3-5 semi-automated (person reviews with evidence attached)
    exception_minutes_per_invoice: float = 4.0   # benchmark: 3-6
    catch_rate: float = 0.90                     # assumption: below the 100% eval recall on purpose
    automated_discount_capture: float = 0.80     # Ardent: above 80%
    platform_cost_per_year: float = 60_000.0     # assumption: hosting, monitoring, model and rule upkeep
    implementation_cost: float = 150_000.0       # assumption: one forward-deployed engagement
    ramp_months: int = 3                         # assumption: shadow mode, then partial, then full automation

    @property
    def annual_spend(self) -> float:
        return self.annual_invoices * self.avg_invoice_value


LABELS = {
    "annual_invoices": "Invoices per year",
    "avg_invoice_value": "Average invoice value",
    "manual_cost_per_invoice": "Manual cost per invoice",
    "error_rate": "Invoice error rate",
    "cost_per_error": "Cost to fix one error",
    "leakage_pct_of_spend": "Spend lost to bad payments",
    "discount_eligible_share": "Spend with early-pay discounts",
    "touchless_share": "Decided without a person",
    "exception_cost_per_invoice": "Cost per exception",
    "catch_rate": "Errors caught before payment",
    "llm_cost_per_invoice": "LLM cost per invoice",
    "platform_cost_per_year": "Platform run cost",
}


def today_costs(a: Assumptions) -> dict[str, float]:
    discounts_available = a.annual_spend * a.discount_eligible_share * a.discount_rate
    return {
        "labour": a.annual_invoices * a.manual_cost_per_invoice,
        "rework": a.annual_invoices * a.error_rate * a.cost_per_error,
        "leakage": a.annual_spend * a.leakage_pct_of_spend,
        "missed_discounts": discounts_available * (1 - a.manual_discount_capture),
    }


def future_costs(a: Assumptions) -> dict[str, float]:
    today = today_costs(a)
    discounts_available = a.annual_spend * a.discount_eligible_share * a.discount_rate
    touchless = a.annual_invoices * a.touchless_share
    exceptions = a.annual_invoices - touchless
    return {
        "labour": touchless * (a.automated_cost_per_invoice + a.llm_cost_per_invoice)
                  + exceptions * (a.exception_cost_per_invoice + a.llm_cost_per_invoice),
        "rework": today["rework"] * (1 - a.catch_rate),
        "leakage": today["leakage"] * (1 - a.catch_rate),
        "missed_discounts": discounts_available * (1 - a.automated_discount_capture),
        "platform": a.platform_cost_per_year,
    }


def business_case(a: Assumptions) -> dict[str, Any]:
    today, future = today_costs(a), future_costs(a)
    today_total, future_total = sum(today.values()), sum(future.values())
    savings = today_total - future_total
    hours_today = a.annual_invoices * a.manual_minutes_per_invoice / 60
    hours_future = a.annual_invoices * (1 - a.touchless_share) * a.exception_minutes_per_invoice / 60
    return {
        "today": today,
        "future": future,
        "today_total": today_total,
        "future_total": future_total,
        "annual_savings": savings,
        "savings_pct": savings / today_total if today_total else 0.0,
        "payback_months": payback_months(a.implementation_cost, savings, a.ramp_months),
        "three_year_net": 3 * savings - a.implementation_cost,
        "hours_freed": hours_today - hours_future,
        "fte_freed": (hours_today - hours_future) / WORK_HOURS_PER_FTE,
        "reconciliation": reconcile(today_total),
    }


def payback_months(implementation_cost: float, annual_savings: float, ramp_months: int) -> float | None:
    """Months until cumulative savings cover the implementation, with savings ramping
    linearly to the full run rate over `ramp_months` (nobody saves 100% on day one)."""
    if annual_savings <= 0:
        return None
    monthly, cumulative = annual_savings / 12, 0.0
    for month in range(1, 121):
        share = min(1.0, month / ramp_months) if ramp_months > 0 else 1.0
        before = cumulative
        cumulative += monthly * share
        if cumulative >= implementation_cost:
            return round(month - 1 + (implementation_cost - before) / (monthly * share), 1)
    return None


# Three scenarios, so the case never rests on one set of numbers. Conservative sits
# at the pessimistic end of each benchmark range; optimistic at the favourable end.
SCENARIOS: dict[str, dict[str, Any]] = {
    "Conservative": dict(manual_cost_per_invoice=15.0, cost_per_error=15.0, leakage_pct_of_spend=0.0005,
                         touchless_share=0.55, catch_rate=0.70, automated_cost_per_invoice=1.0,
                         exception_cost_per_invoice=8.0, llm_cost_per_invoice=0.15, automated_discount_capture=0.60,
                         platform_cost_per_year=120_000.0, implementation_cost=300_000.0, ramp_months=6),
    "Base": {},
    "Optimistic": dict(touchless_share=0.85, catch_rate=0.95, exception_cost_per_invoice=3.0,
                       automated_cost_per_invoice=0.30, platform_cost_per_year=40_000.0, ramp_months=2),
}


def scenario(name: str, base: Assumptions | None = None) -> Assumptions:
    return replace(base or Assumptions(), **SCENARIOS[name])


def reconcile(today_total: float, stated: float = STATED_ANNUAL_LOSS) -> dict[str, Any]:
    """Do the assumptions actually explain the brief's $2M? A business case that
    doesn't reconcile to the client's own number won't survive the first question."""
    gap = today_total - stated
    pct = gap / stated
    if abs(pct) <= 0.10:
        verdict = "reconciles"
    elif pct < 0:
        verdict = "under-explains"
    else:
        verdict = "over-explains"
    return {"stated": stated, "explained": today_total, "gap": gap, "gap_pct": pct, "verdict": verdict}


def waterfall(a: Assumptions) -> list[dict[str, Any]]:
    """Today's cost -> each saving -> added cost -> cost with the system."""
    today, future = today_costs(a), future_costs(a)
    steps = [{"label": "Cost today", "kind": "total", "value": sum(today.values())}]
    names = {"labour": "Less handling labour", "rework": "Fewer errors to fix",
             "leakage": "Bad payments stopped", "missed_discounts": "Discounts captured"}
    for key, label in names.items():
        steps.append({"label": label, "kind": "saving" if today[key] >= future[key] else "cost",
                      "value": future[key] - today[key]})
    steps.append({"label": "Platform run cost", "kind": "cost", "value": future["platform"]})
    steps.append({"label": "Cost with system", "kind": "total", "value": sum(future.values())})
    running = 0.0
    for s in steps:  # bar spans for plotting
        if s["kind"] == "total":
            s["start"], s["end"], running = 0.0, s["value"], s["value"]
        else:
            s["start"], s["end"] = running, running + s["value"]
            running = s["end"]
    return steps


SENSITIVITY_RANGES = {   # (low, high) multipliers or absolute bounds, chosen to span the benchmark ranges
    "annual_invoices": (0.5, 1.5),
    "manual_cost_per_invoice": (10.18 / 21.40, 40 / 21.40),   # APQC top quartile .. Ardent manual high end
    "touchless_share": (0.55 / 0.75, 0.90 / 0.75),
    "catch_rate": (0.70 / 0.90, 1.0 / 0.90),
    "cost_per_error": (0.4, 2.0),
    "leakage_pct_of_spend": (0.5, 3.0),
    "exception_cost_per_invoice": (0.75, 2.0),
    "platform_cost_per_year": (0.5, 2.0),
}


def sensitivity(a: Assumptions) -> list[dict[str, Any]]:
    """Annual savings when each assumption moves across its plausible range, others held.
    Sorted by swing: the top rows are the assumptions to pin down in discovery."""
    base = business_case(a)["annual_savings"]
    rows = []
    for name, (lo, hi) in SENSITIVITY_RANGES.items():
        value = getattr(a, name)
        lo_v, hi_v = value * lo, value * hi
        if name in {"touchless_share", "catch_rate"}:
            lo_v, hi_v = min(lo_v, 1.0), min(hi_v, 1.0)
        if isinstance(value, int):
            lo_v, hi_v = int(lo_v), int(hi_v)
        s_lo = business_case(replace(a, **{name: lo_v}))["annual_savings"]
        s_hi = business_case(replace(a, **{name: hi_v}))["annual_savings"]
        rows.append({"assumption": name, "label": LABELS.get(name, name), "low_input": lo_v, "high_input": hi_v,
                     "savings_at_low": s_lo, "savings_at_high": s_hi, "base": base,
                     "swing": abs(s_hi - s_lo)})
    return sorted(rows, key=lambda r: r["swing"], reverse=True)


def from_measurements(a: Assumptions, touchless_share: float | None = None,
                      llm_cost_per_invoice: float | None = None) -> Assumptions:
    """Overlay numbers measured by the running system on the defaults."""
    updates: dict[str, Any] = {}
    if touchless_share is not None:
        updates["touchless_share"] = max(0.0, min(1.0, touchless_share))
    if llm_cost_per_invoice is not None:
        updates["llm_cost_per_invoice"] = max(0.0, llm_cost_per_invoice)
    return replace(a, **updates)


def llm_cost_from_calls(calls: list[dict[str, Any]], invoices: int, usd_per_m_input: float,
                        usd_per_m_output: float) -> float | None:
    """Measured LLM cost per invoice from recorded token usage. None if no real model calls were made."""
    tokens_in = sum(c.get("prompt_tokens") or 0 for c in calls if c.get("provider") != "offline")
    tokens_out = sum(c.get("completion_tokens") or 0 for c in calls if c.get("provider") != "offline")
    if not invoices or not (tokens_in or tokens_out):
        return None
    return (tokens_in * usd_per_m_input + tokens_out * usd_per_m_output) / 1_000_000 / invoices


def as_dict(a: Assumptions) -> dict[str, Any]:
    return asdict(a)
