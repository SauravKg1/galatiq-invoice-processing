"""Grok vs rules scorecard: what the LLM adds, what it costs, how stable it is.

Runs the labelled eval in rules-only mode and in LLM mode (optionally several
times), on fresh databases, then reports quality, LLM contributions, cost,
latency, stability and every invoice where the two modes disagree. Results are
written to eval/results/ as JSON and to eval/SCORECARD.md for the repo.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import PROJECT_ROOT, Settings
from .evaluation import run_eval
from .llm import LLMClient, OfflineLLM
from .models import ProcessingOutcome
from .reporting import summarize

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REPORT_PATH = PROJECT_ROOT / "eval" / "SCORECARD.md"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 1)


def mode_metrics(result: dict[str, Any], price_in: float, price_out: float) -> dict[str, Any]:
    """Quality metrics from the eval plus what the LLM actually did, from recorded calls."""
    outcomes: list[ProcessingOutcome] = result["outcomes"]
    calls = [c for o in outcomes for c in o.llm_calls if c.get("provider") != "offline"]
    tokens_in = sum(c.get("prompt_tokens") or 0 for c in calls)
    tokens_out = sum(c.get("completion_tokens") or 0 for c in calls)
    n = len(outcomes) or 1
    llm_findings = Counter(f.code for o in outcomes if o.validation for f in o.validation.findings if f.source == "llm")
    latencies = [o.total_ms for o in outcomes]
    return {
        **result["metrics"],
        "touchless_rate": summarize(outcomes)["touchless_rate"],
        "llm_calls": len(calls),
        "llm_calls_failed": sum(1 for c in calls if not c.get("ok")),
        "degraded_to_rules": sum(1 for c in calls if "DEGRADED" in (c.get("note") or "")),
        "schema_repairs": sum(1 for c in calls if "schema repair" in (c.get("note") or "")),
        "self_corrections": sum(1 for o in outcomes if o.extraction and o.extraction.attempts > 1),
        "critique_disagreements": sum(1 for o in outcomes if o.approval
                                      for r in o.approval.rounds if not r.critique.agrees),
        "guardrail_overrides": sum(len(o.approval.overrides) for o in outcomes if o.approval),
        "llm_findings": dict(llm_findings.most_common()),
        "tool_calls": sum(len(o.validation.tool_calls) for o in outcomes if o.validation),
        "tokens_in_per_invoice": round(tokens_in / n),
        "tokens_out_per_invoice": round(tokens_out / n),
        "cost_per_invoice_usd": round((tokens_in * price_in + tokens_out * price_out) / 1_000_000 / n, 4),
        "cost_per_run_usd": round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 3),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
    }


def stability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of invoices that got the same final status in every run."""
    if len(runs) < 2:
        return {"runs": len(runs), "consistent_share": None, "unstable": []}
    by_file: dict[str, list[str]] = {}
    for run in runs:
        for row in run["rows"]:
            by_file.setdefault(row["file"], []).append(row["actual"])
    unstable = [{"file": f, "statuses": s} for f, s in by_file.items() if len(set(s)) > 1]
    return {"runs": len(runs), "consistent_share": round(1 - len(unstable) / len(by_file), 3), "unstable": unstable}


def disagreements(rules: dict[str, Any], llm: dict[str, Any]) -> list[dict[str, Any]]:
    """Invoices where rules-only and LLM mode reached different outcomes, with the LLM's reasoning."""
    out = []
    for r_row, l_row, l_out in zip(rules["rows"], llm["rows"], llm["outcomes"]):
        if r_row["actual"] == l_row["actual"]:
            continue
        llm_codes = sorted({f.code for f in l_out.validation.findings if f.source == "llm"}) if l_out.validation else []
        out.append({
            "file": r_row["file"],
            "expected": r_row["expected"],
            "rules": r_row["actual"],
            "llm": l_row["actual"],
            "llm_correct": l_row["status_ok"],
            "rules_correct": r_row["status_ok"],
            "llm_findings": llm_codes,
            "llm_rationale": (l_out.approval.final.rationale if l_out.approval else l_out.error or "")[:400],
        })
    return out


def build_scorecard(settings: Settings, llm_factory, runs: int = 1, price_in: float = 3.0,
                    price_out: float = 15.0, rules_llm: Optional[LLMClient] = None) -> dict[str, Any]:
    """`llm_factory()` returns a fresh LLM client per run (so call logs don't mix)."""
    rules = run_eval(settings, llm=rules_llm or OfflineLLM())
    llm_runs = [run_eval(settings, llm=llm_factory()) for _ in range(max(1, runs))]
    sample_llm = llm_runs[0]["outcomes"][0].llm_mode if llm_runs[0]["outcomes"] else "unknown"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm": sample_llm,
        "runs": len(llm_runs),
        "prices_usd_per_m_tokens": {"input": price_in, "output": price_out},
        "rules": mode_metrics(rules, price_in, price_out),
        "llm_runs": [mode_metrics(r, price_in, price_out) for r in llm_runs],
        "stability": stability(llm_runs),
        "disagreements": disagreements(rules, llm_runs[0]),
        "rows": [{"file": r["file"], "expected": r["expected"], "rules": r["actual"],
                  **{f"llm_run_{i + 1}": run["rows"][j]["actual"] for i, run in enumerate(llm_runs)}}
                 for j, r in enumerate(rules["rows"])],
    }


# --------------------------------------------------------------------- report
def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(card: dict[str, Any]) -> str:
    rules, llm = card["rules"], card["llm_runs"][0]
    provider = card["llm"].split(":")[0]
    name = "Grok" if "grok" in provider.lower() else provider
    stab = card["stability"]
    lines = [
        f"# Scorecard: {name} vs rules-only",
        "",
        f"Generated {card['generated_at']} with `{card['llm']}` on the labelled eval set "
        f"({rules['invoices']} invoices scored), {card['runs']} LLM run(s). "
        f"Prices assumed: ${card['prices_usd_per_m_tokens']['input']}/M input, "
        f"${card['prices_usd_per_m_tokens']['output']}/M output tokens. Reproduce with `python main.py --compare`.",
        "",
        "## Quality",
        "",
        f"| Metric | Rules only | {name} |",
        "|---|---|---|",
        f"| Correct final outcome | {_pct(rules['status_accuracy'])} | {_pct(llm['status_accuracy'])} |",
        f"| Known issues flagged | {_pct(rules['flag_recall'])} | {_pct(llm['flag_recall'])} |",
        f"| Wrongful payments | {rules['wrongful_payments']} | {llm['wrongful_payments']} |",
        f"| Clean invoices blocked | {rules['missed_payments']} | {llm['missed_payments']} |",
        f"| Decided without a person | {_pct(rules['touchless_rate'])} | {_pct(llm['touchless_rate'])} |",
        "",
        f"## What {name} did",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| LLM calls (failed) | {llm['llm_calls']} ({llm['llm_calls_failed']}) |",
        f"| Investigator tool calls | {llm['tool_calls']} |",
        f"| Extractions that self-corrected | {llm['self_corrections']} |",
        f"| Schema repairs | {llm['schema_repairs']} |",
        f"| Auditor disagreed with the VP agent | {llm['critique_disagreements']} time{'s' if llm['critique_disagreements'] != 1 else ''} |",
        f"| Guardrail overrides of agent decisions | {llm['guardrail_overrides']} |",
        f"| Calls degraded to rules (outages, bad output) | {llm['degraded_to_rules']} |",
        f"| Findings only the investigator raised | {', '.join(f'{k} ({v})' for k, v in llm['llm_findings'].items()) or 'none'} |",
        "",
        "## Cost and speed",
        "",
        f"| Metric | Rules only | {name} |",
        "|---|---|---|",
        f"| Tokens per invoice (in / out) | 0 | {llm['tokens_in_per_invoice']:,} / {llm['tokens_out_per_invoice']:,} |",
        f"| Cost per invoice | $0 | ${llm['cost_per_invoice_usd']:.4f} |",
        f"| Cost of this eval run | $0 | ${llm['cost_per_run_usd']:.2f} |",
        f"| Time per invoice, median / 95th percentile | {rules['latency_ms_p50']:,.0f} / {rules['latency_ms_p95']:,.0f} ms "
        f"| {llm['latency_ms_p50']:,.0f} / {llm['latency_ms_p95']:,.0f} ms |",
        "",
        "## Stability",
        "",
    ]
    if stab["consistent_share"] is None:
        lines.append("Single LLM run. Run `python main.py --compare --runs 3` to measure repeatability.")
    else:
        lines.append(f"Across {stab['runs']} runs, **{_pct(stab['consistent_share'])}** of invoices got the same outcome every time.")
        for u in stab["unstable"]:
            lines.append(f"- `{u['file']}`: {' / '.join(u['statuses'])}")
    lines += ["", f"## Where {name} and the rules disagree", ""]
    if not card["disagreements"]:
        lines.append("No disagreements: both modes reached the same outcome on every invoice.")
    else:
        lines += [f"| Invoice | Expected | Rules | {name} | {name}'s reasoning |", "|---|---|---|---|---|"]
        for d in card["disagreements"]:
            mark = lambda ok: "" if ok else " (wrong)"  # noqa: E731
            reason = d["llm_rationale"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {d['file']} | {d['expected']} | {d['rules']}{mark(d['rules_correct'])} "
                         f"| {d['llm']}{mark(d['llm_correct'])} | {reason} |")
    lines += ["", "## Every invoice", "", "| Invoice | Expected | Rules | " +
              " | ".join(f"{name} run {i + 1}" for i in range(card["runs"])) + " |",
              "|---|---|---|" + "---|" * card["runs"]]
    for r in card["rows"]:
        lines.append(f"| {r['file']} | {r['expected']} | {r['rules']} | " +
                     " | ".join(r[f"llm_run_{i + 1}"] for i in range(card["runs"])) + " |")
    return "\n".join(lines) + "\n"


def write_report(card: dict[str, Any], results_dir: Path = RESULTS_DIR, report_path: Path = REPORT_PATH) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = card["generated_at"].replace(":", "").replace("-", "")[:15]
    run_file = results_dir / f"scorecard_{stamp}.json"
    payload = json.dumps(card, indent=2, default=str)
    run_file.write_text(payload)
    (results_dir / "latest.json").write_text(payload)
    report_path.write_text(render_markdown(card))
    return {"json": run_file, "latest": results_dir / "latest.json", "markdown": report_path}


def load_latest(results_dir: Path = RESULTS_DIR) -> Optional[dict[str, Any]]:
    path = results_dir / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
