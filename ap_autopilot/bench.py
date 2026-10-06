"""Before/after benchmark for the throughput optimisations.

    python main.py --bench            # simulated Grok latencies (no key needed)
    python main.py --bench --live     # real Grok calls, real numbers (needs XAI_API_KEY)

Four configurations run the labelled eval set on a fresh database each time:

  baseline         one invoice at a time, every call on the reasoning model, full VP + auditor loop
  + routing        reading and drafting on the fast model
  + fast path      clean structured invoices skip the auditor round
  + parallel reads up to N files read ahead while decisions stay in arrival order

Each run reports LLM calls, tokens, cost and wall time, and checks that every
invoice got the same decision as the baseline. An optimisation that changes a
decision is a bug, not a speed-up.

Simulated mode: the model returns the same deterministic answers as offline mode, and
each call sleeps for a latency taken from the ASSUMPTIONS below, scaled down so the
bench finishes in seconds. Times are scaled back up for the report. Tokens are
estimated from prompt length. Every number in simulated mode is an estimate built
on those assumptions, and the report says so.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from .config import PROJECT_ROOT, Settings
from .evaluation import run_eval
from .llm import LLMClient, LLMError, build_llm, is_fast_role

BENCH_PATH = PROJECT_ROOT / "eval" / "BENCH.md"

# Assumed per-call behaviour of each model tier. Edit to match your measurements.
# Prices are USD per 1M tokens; check docs.x.ai for current list prices.
ASSUMPTIONS: dict[str, dict[str, float]] = {
    "reasoning": {"first_token_s": 1.5, "tokens_per_s": 80, "output_tokens": 500, "price_in": 3.00, "price_out": 15.00},
    "fast": {"first_token_s": 0.4, "tokens_per_s": 150, "output_tokens": 250, "price_in": 0.20, "price_out": 0.50},
}
INVESTIGATOR_TURNS = 3          # typical: two database lookups, then submit_result
TOOL_RESULT_TOKENS = 250        # each lookup result appended to the conversation


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens


class SimulatedLLM(LLMClient):
    """Behaves like a networked model (offline=False, so every LLM step runs), answers with the
    deterministic fallbacks, and waits as long as the assumed model would."""

    provider = "simulated"
    offline = False
    supports_vision = False

    def __init__(self, routed: bool, scale: float = 0.005):
        super().__init__()
        self.routed, self.scale = routed, scale
        self.model = "sim-reasoning"
        self.main_wait_s = 0.0          # simulated waiting on the decision thread (scaled)
        self._lock = threading.Lock()

    def model_for(self, role: str) -> str:
        return "sim-fast" if self.routed and is_fast_role(role) else "sim-reasoning"

    def _call(self, role: str, prompt_chars: int) -> None:
        tier = "fast" if self.model_for(role) == "sim-fast" else "reasoning"
        a = ASSUMPTIONS[tier]
        seconds = a["first_token_s"] + a["output_tokens"] / a["tokens_per_s"]
        start = time.perf_counter()
        time.sleep(seconds * self.scale)
        waited = time.perf_counter() - start
        if threading.current_thread() is threading.main_thread():
            with self._lock:
                self.main_wait_s += waited
        self._record(role, ok=True, latency_ms=seconds * 1000, usage=_Usage(prompt_chars // 4, int(a["output_tokens"])),
                     model=self.model_for(role), note="simulated")

    def structured(self, *, role, system, user, schema, fallback=None, images=None):
        if fallback is None:
            raise LLMError(f"{role}: simulated mode needs a deterministic fallback")
        self._call(role, len(system) + len(user))
        return fallback()

    def tool_loop(self, *, role, system, user, tools, final_schema, fallback=None, max_steps=8):
        if fallback is None:
            raise LLMError(f"{role}: simulated mode needs a deterministic fallback")
        chars = len(system) + len(user) + 600 * len(tools)   # tool schemas are part of every prompt
        for turn in range(INVESTIGATOR_TURNS):
            self._call(role, chars + turn * 4 * (TOOL_RESULT_TOKENS + 150))
        return fallback(), []


CONFIGS = [
    ("Baseline", dict(routed=False, fast_path=False, workers=1)),
    ("+ model routing", dict(routed=True, fast_path=False, workers=1)),
    ("+ fast path", dict(routed=True, fast_path=True, workers=1)),
    ("+ parallel reads", dict(routed=True, fast_path=True, workers=4)),
]


def _cost(calls: list[dict[str, Any]], routed: bool, live_prices: Optional[dict] = None) -> tuple[float, int, int]:
    usd, tin, tout = 0.0, 0, 0
    for c in calls:
        p_in, p_out = c.get("prompt_tokens") or 0, c.get("completion_tokens") or 0
        tier = "fast" if routed and is_fast_role(c["role"]) else "reasoning"
        prices = (live_prices or ASSUMPTIONS)[tier]
        usd += p_in / 1e6 * prices["price_in"] + p_out / 1e6 * prices["price_out"]
        tin, tout = tin + p_in, tout + p_out
    return usd, tin, tout


def run_bench(settings: Optional[Settings] = None, live: bool = False, scale: float = 0.005,
              workers: int = 4, live_prices: Optional[dict] = None) -> dict[str, Any]:
    base = settings or Settings()
    results, baseline_decisions = [], None
    for name, cfg in CONFIGS:
        s = replace(base, fast_path=cfg["fast_path"], read_workers=workers if cfg["workers"] > 1 else 1)
        if live:
            if not cfg["routed"]:   # baseline: one model for everything
                s = replace(s, xai_model_fast=s.xai_model_reasoning or s.xai_model,
                            openai_model_fast=s.openai_model_reasoning or s.openai_model)
            llm = build_llm(s)
            if llm.offline:
                raise LLMError("--bench --live needs an LLM key (XAI_API_KEY); run without --live for the simulation")
            eff_scale = 1.0
        else:
            llm = SimulatedLLM(routed=cfg["routed"], scale=scale)
            eff_scale = scale
        report = run_eval(s, llm=llm)
        calls = [c for o in report["outcomes"] for c in o.llm_calls if c.get("ok")]
        usd, tin, tout = _cost(calls, cfg["routed"], live_prices if live else None)
        if live:
            wall = report["wall_s"]
        else:   # scale the waiting back up; real compute time stays as measured
            waiting = llm.main_wait_s + report["batch_wait_s"]
            wall = report["wall_s"] + waiting * (1 / eff_scale - 1)
        decisions = {r["file"]: r["actual"] for r in report["rows"]}
        baseline_decisions = baseline_decisions or decisions
        same = sum(decisions[f] == baseline_decisions.get(f) for f in decisions)
        results.append({
            "name": name, **cfg, "invoices": len(decisions), "calls": len(calls),
            "reasoning_calls": sum(1 for c in calls if not (cfg["routed"] and is_fast_role(c["role"]))),
            "fast_calls": sum(1 for c in calls if cfg["routed"] and is_fast_role(c["role"])),
            "auditor_calls": sum(1 for c in calls if c["role"] == "approval.critic"),
            "fast_path_invoices": sum(1 for o in report["outcomes"] if o.approval and o.approval.fast_path),
            "tokens_in": tin, "tokens_out": tout, "cost_usd": usd, "wall_s": wall,
            "same_decisions": same, "metrics": report["metrics"],
        })
    return {"live": live, "scale": scale, "workers": workers, "results": results}


def render_markdown(bench: dict[str, Any]) -> str:
    r0, r1 = bench["results"][0], bench["results"][-1]
    mode = "live Grok calls" if bench["live"] else "simulated latencies (assumptions below)"
    lines = [
        "# Throughput benchmark",
        "",
        f"Labelled eval set, {r0['invoices']} invoices, fresh database per run, {mode}. "
        "Generated by `python main.py --bench" + (" --live" if bench["live"] else "") + "`.",
        "",
        "| Configuration | LLM calls | Reasoning / fast | Auditor calls | Fast-path invoices | Tokens in / out | Cost | Wall time | Same decisions | Wrongful payments |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in bench["results"]:
        lines.append(
            f"| {r['name']} | {r['calls']} | {r['reasoning_calls']} / {r['fast_calls']} | {r['auditor_calls']} | "
            f"{r['fast_path_invoices']} | {r['tokens_in']:,} / {r['tokens_out']:,} | ${r['cost_usd']:.2f} | "
            f"{_duration(r['wall_s'])} | {r['same_decisions']}/{r['invoices']} | {r['metrics']['wrongful_payments']} |")
    lines += [
        "",
        f"**Net effect:** wall time {_duration(r0['wall_s'])} to {_duration(r1['wall_s'])} "
        f"({1 - r1['wall_s'] / r0['wall_s']:.0%} less), cost ${r0['cost_usd']:.2f} to ${r1['cost_usd']:.2f} "
        f"({1 - r1['cost_usd'] / r0['cost_usd']:.0%} less), reasoning-model calls {r0['reasoning_calls']} to "
        f"{r1['reasoning_calls']}, with {r1['same_decisions']}/{r1['invoices']} decisions unchanged.",
        "",
        "## What each optimisation does",
        "",
        "- **Model routing.** Reading text invoices and drafting vendor emails are pattern work, so they go to the fast "
        "model (`XAI_MODEL_FAST`). The investigator, VP and auditor judge risk and money, so they stay on the reasoning "
        "model (`XAI_MODEL_REASONING`). Both default to `XAI_MODEL`, so routing is opt-in.",
        "- **Fast path.** A structured file with no warnings, zero fraud risk, low investigator risk, under the "
        "approval limit and in USD skips the auditor round when the VP approves. The investigator still runs and "
        "the policy guardrails still apply. Anything else, or a VP decision other than approve, gets the full debate. "
        "`AP_FAST_PATH=0` turns it off.",
        f"- **Parallel reads.** Up to {bench['workers']} files are read ahead in parallel (`AP_READ_WORKERS`). "
        "Checking, deciding and paying stay strictly in arrival order, because duplicate detection, PO consumption and "
        "paid-once-ever depend on every earlier invoice. Parallel decisions would risk paying twice.",
        "",
        "## Where the remaining time goes",
        "",
        "The investigator's tool loop (several reasoning calls per invoice) is now the largest cost. Next levers: "
        "skip it for structured files the rules cleared, or give it a fast model for the lookups and the reasoning "
        "model only for the final verdict. Both trade some catch rate for speed, so they should be measured with "
        "`--compare` on real Grok before being switched on.",
    ]
    if not bench["live"]:
        a, f = ASSUMPTIONS["reasoning"], ASSUMPTIONS["fast"]
        lines += [
            "",
            "## Assumptions (simulated mode)",
            "",
            "The simulated model gives the same deterministic answers as offline mode, so this run measures call counts "
            "and waiting time, not answer quality. It does show that the optimisations never change a decision. "
            "Run `--bench --live` with a key for real latencies and token counts.",
            "",
            f"- Reasoning model: {a['first_token_s']}s to first token, {a['tokens_per_s']:.0f} tokens/s, "
            f"{a['output_tokens']:.0f} output tokens per call (including reasoning), ${a['price_in']:.2f} / ${a['price_out']:.2f} per 1M tokens in / out.",
            f"- Fast model: {f['first_token_s']}s to first token, {f['tokens_per_s']:.0f} tokens/s, "
            f"{f['output_tokens']:.0f} output tokens per call, ${f['price_in']:.2f} / ${f['price_out']:.2f} per 1M tokens in / out.",
            f"- Investigator: {INVESTIGATOR_TURNS} turns per invoice. Input tokens are prompt characters / 4.",
            f"- Sleeps run at {bench['scale']} of real time and are scaled back up; compute time is reported as measured.",
            "- Prices are assumptions; check docs.x.ai for current list prices.",
        ]
    return "\n".join(lines) + "\n"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f} min"


def write_markdown(bench: dict[str, Any], path: Path = BENCH_PATH) -> Path:
    path.write_text(render_markdown(bench))
    return path
