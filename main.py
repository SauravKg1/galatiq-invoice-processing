#!/usr/bin/env python3
"""Acme Corp AP Autopilot: multi-agent invoice processing.

  python main.py --invoice_path=data/invoices/invoice_1001.txt     # one invoice (the brief's command)
  python main.py --batch data/invoices                              # a folder, in arrival order
  python main.py --review                                           # human approval queue
  python main.py --eval                                             # scorecard on the labelled set
  python main.py --reset-db                                         # fresh inventory.db

Options: --llm auto|grok|openai|offline  --verbose  --json  --log-events  --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ap_autopilot.config import PROJECT_ROOT, Settings
from ap_autopilot.db import Database
from ap_autopilot.graph import InvoicePipeline
from ap_autopilot.llm import LLMError, build_llm
from ap_autopilot.loaders import SUPPORTED_EXTENSIONS
from ap_autopilot.reporting import render_batch, render_outcome, summarize

SUPPORTED = SUPPORTED_EXTENSIONS
console = Console()


def interactive_reviewer(ctx: dict) -> tuple[str | None, str]:
    approval = ctx["approval"]
    console.print(f"\n[bold yellow]Human review needed[/] for {Path(ctx['file']).name}: {approval.final.rationale}")
    choice = Prompt.ask("Approve, reject, or leave in queue?", choices=["a", "r", "q"], default="q")
    if choice == "q":
        return None, ""
    note = Prompt.ask("Note for the audit log", default="reviewed in CLI")
    return ("approve" if choice == "a" else "reject"), note


def collect(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED)


def cmd_review(pipeline: InvoicePipeline) -> None:
    queue = pipeline.db.review_queue("pending")
    if not queue:
        console.print("[green]Review queue is empty.[/]")
        return
    table = Table(title=f"Pending human review ({len(queue)})", header_style="bold")
    for col in ("ID", "File", "Vendor", "Amount", "Reason"):
        table.add_column(col, overflow="fold")
    for q in queue:
        table.add_row(str(q["id"]), q["source_file"] or "", q["vendor"] or "?",
                      f"{q['currency']} {q['amount']:,.2f}" if q["amount"] is not None else "n/a", (q["reason"] or "")[:160])
    console.print(table)
    while True:
        choice = Prompt.ask("Item ID to decide (blank to quit)", default="")
        if not choice:
            return
        if not choice.isdigit() or not pipeline.db.get_review(int(choice)):
            console.print("[red]No such item[/]")
            continue
        verdict = Prompt.ask("Approve or reject?", choices=["a", "r"])
        note = Prompt.ask("Reason (goes to the audit log)", default="")
        reviewer = Prompt.ask("Your name", default="ap-reviewer")
        try:
            result = pipeline.human_decision(int(choice), verdict == "a", reviewer, note)
            console.print(f"[bold]{result.status}[/]: {result.message}")
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")


def cmd_outbox(pipeline: InvoicePipeline) -> None:
    drafts = pipeline.db.outbox("draft")
    if not drafts:
        console.print("[green]Outbox is empty.[/]")
        return
    table = Table(title=f"Drafts waiting to be sent ({len(drafts)}). Nothing is sent automatically.", header_style="bold")
    for col in ("ID", "To", "Kind", "Subject", "Written by"):
        table.add_column(col, overflow="fold")
    for d in drafts:
        table.add_row(str(d["id"]), d["to_addr"], d["kind"].replace("_", " "), d["subject"], "Grok" if d["author"] == "llm" else "template")
    console.print(table)
    while True:
        choice = Prompt.ask("Draft ID to read (blank to quit)", default="")
        if not choice:
            return
        draft = pipeline.db.get_draft(int(choice)) if choice.isdigit() else None
        if not draft or draft["status"] != "draft":
            console.print("[red]No such draft[/]")
            continue
        console.print(f"[bold]To:[/] {draft['to_addr']}\n[bold]Subject:[/] {draft['subject']}\n\n{draft['body']}\n")
        action = Prompt.ask("Send, discard, or keep?", choices=["s", "d", "k"], default="k")
        if action != "k":
            who = Prompt.ask("Your name", default="ap-reviewer")
            pipeline.db.set_draft_status(draft["id"], "sent" if action == "s" else "discarded", who)
            pipeline.db.audit(draft["run_id"], draft["invoice_key"], "communicate", "sent" if action == "s" else "discarded",
                              {"draft_id": draft["id"], "to": draft["to_addr"], "by": who})
            console.print("[green]Marked as sent (mock: no real email leaves this machine).[/]" if action == "s" else "Discarded.")


def cmd_eval(settings: Settings, provider: str) -> int:
    from ap_autopilot.evaluation import run_eval

    result = run_eval(settings, llm=build_llm(settings, provider))
    table = Table(title="Evaluation on labelled invoices (fresh DB, arrival order)", header_style="bold")
    for col in ("File", "Expected", "Actual", "OK", "Missing flags", "Trap"):
        table.add_column(col, overflow="fold")
    for r in result["rows"]:
        ok = "[green]yes[/]" if r["status_ok"] and not r["missing_flags"] else "[red]NO[/]"
        if r["skipped"]:
            ok = "[dim]needs vision/OCR[/]"
        if r["wrongful_payment"]:
            ok = "[bold red]WRONGFUL PAYMENT[/]"
        table.add_row(r["file"], r["expected"], r["actual"], ok, ", ".join(r["missing_flags"]), r["why"])
    console.print(table)
    m = result["metrics"]
    console.print(f"status accuracy [bold]{m['status_accuracy']:.0%}[/] | flag recall [bold]{m['flag_recall']:.0%}[/] | "
                  f"wrongful payments [bold]{m['wrongful_payments']}[/] | missed payments [bold]{m['missed_payments']}[/] | "
                  f"avg {m['avg_ms']:.0f} ms/invoice")
    if m["needs_vision_skipped"]:
        console.print(f"[dim]{m['needs_vision_skipped']} scanned invoice(s) not scored: set XAI_API_KEY for Grok vision "
                      f"or install Tesseract OCR. They were still processed, and none was paid.[/]")
    return 0 if m["wrongful_payments"] == 0 else 1


def cmd_bench(settings: Settings, live: bool, workers: int, price_in: float, price_out: float) -> int:
    from ap_autopilot.bench import ASSUMPTIONS, run_bench, write_markdown

    console.print("[bold]Throughput benchmark[/] " + ("(live LLM calls)" if live else "(simulated latencies, no key needed)"))
    prices = None
    if live:
        prices = {"reasoning": {"price_in": price_in, "price_out": price_out},
                  "fast": {"price_in": ASSUMPTIONS["fast"]["price_in"], "price_out": ASSUMPTIONS["fast"]["price_out"]}}
    try:
        bench = run_bench(settings, live=live, workers=workers, live_prices=prices)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    table = Table(box=box.SIMPLE_HEAD)
    for col in ("Config", "Calls", "Deep/fast", "Auditor", "Cost", "Wall", "Same"):
        table.add_column(col, justify="left" if col == "Config" else "right", no_wrap=True)
    for r in bench["results"]:
        table.add_row(r["name"], str(r["calls"]), f"{r['reasoning_calls']} / {r['fast_calls']}", str(r["auditor_calls"]),
                      f"${r['cost_usd']:.2f}",
                      f"{r['wall_s'] / 60:.1f}m" if r["wall_s"] >= 90 else f"{r['wall_s']:.0f}s",
                      f"{r['same_decisions']}/{r['invoices']}")
    console.print(table)
    path = write_markdown(bench)
    console.print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    bad = [r for r in bench["results"] if r["same_decisions"] != r["invoices"] or r["metrics"]["wrongful_payments"]]
    if bad and not live:
        console.print("[red]An optimisation changed a decision.[/]")
        return 1
    return 0


def cmd_compare(settings: Settings, provider: str, runs: int, price_in: float, price_out: float) -> int:
    from ap_autopilot.scorecard import build_scorecard, write_report

    try:
        probe = build_llm(settings, provider)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    if probe.offline:
        console.print("[red]--compare needs an LLM to compare against. Set XAI_API_KEY in .env (see .env.example).[/]")
        return 2
    console.print(f"Running the eval rules-only, then {runs}x on {probe.provider}:{probe.model}. "
                  f"This makes real API calls and takes a few minutes.")
    card = build_scorecard(settings, lambda: build_llm(settings, provider), runs=runs,
                           price_in=price_in, price_out=price_out)
    paths = write_report(card)
    rules, llm = card["rules"], card["llm_runs"][0]
    table = Table(title="Scorecard: rules-only vs LLM", header_style="bold")
    for col in ("Metric", "Rules only", card["llm"]):
        table.add_column(col)
    for label, key, fmt in [("Correct outcome", "status_accuracy", "{:.0%}"), ("Issues flagged", "flag_recall", "{:.0%}"),
                            ("Wrongful payments", "wrongful_payments", "{}"), ("Clean invoices blocked", "missed_payments", "{}"),
                            ("Decided without a person", "touchless_rate", "{:.0%}"), ("Cost per invoice", "cost_per_invoice_usd", "${:.4f}"),
                            ("Median ms per invoice", "latency_ms_p50", "{:,.0f}")]:
        table.add_row(label, fmt.format(rules[key]), fmt.format(llm[key]))
    console.print(table)
    if card["stability"]["consistent_share"] is not None:
        console.print(f"Stability: {card['stability']['consistent_share']:.0%} of invoices got the same outcome in all {runs} runs.")
    console.print(f"{len(card['disagreements'])} invoice(s) where the modes disagree. Full report: {paths['markdown']}")
    return 0 if all(r["wrongful_payments"] == 0 for r in card["llm_runs"]) else 1


def cmd_roi() -> int:
    from ap_autopilot.roi import BENCHMARKS, SCENARIOS, business_case, scenario, sensitivity

    table = Table(title="Business case: annual AP cost, today vs with AP Autopilot", header_style="bold")
    for col in ("Scenario", "Cost today", "With system", "Annual savings", "Payback", "3-year net", "Staff time freed"):
        table.add_column(col, justify="right" if col != "Scenario" else "left")
    for name in SCENARIOS:
        c = business_case(scenario(name))
        table.add_row(name, f"${c['today_total']:,.0f}", f"${c['future_total']:,.0f}",
                      f"${c['annual_savings']:,.0f} ({c['savings_pct']:.0%})", f"{c['payback_months']} months",
                      f"${c['three_year_net']:,.0f}", f"{c['fte_freed']:.1f} FTE")
    console.print(table)
    base = business_case(scenario("Base"))
    r = base["reconciliation"]
    console.print(f"Base case explains ${r['explained']:,.0f} of the brief's ${r['stated']:,.0f} loss "
                  f"({r['gap_pct']:+.0%}): {r['verdict']}.")
    top = sensitivity(scenario("Base"))[:3]
    console.print("Assumptions that move the answer most: " + "; ".join(
        f"{t['label']} (${t['savings_at_low']:,.0f} to ${t['savings_at_high']:,.0f})" for t in top))
    console.print("[dim]Sources: " + " | ".join(f"{v[0]}" for v in list(BENCHMARKS.values())[:3]) + " ...[/]")
    return 0


def cmd_check_llm(settings: Settings, provider: str, vision: bool = False) -> int:
    """Prove the key, model name and structured-output path work before a demo."""
    from ap_autopilot.models import Decision

    try:
        llm = build_llm(settings, provider)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    if llm.offline:
        console.print("[yellow]Offline mode: no LLM configured. Set XAI_API_KEY in .env (see .env.example).[/]")
        return 1
    try:
        result = llm.structured(role="check", system="You approve test invoices.", schema=Decision,
                                user="A $10 test invoice from an approved vendor passed every check. Decide.")
    except Exception as exc:
        _explain_llm_failure(llm, exc)
        return 2
    call = llm.calls[-1] if llm.calls else {}
    console.print(f"[green]OK[/] {llm.provider}:{llm.model} replied {result.decision} "
                  f"in {call.get('latency_ms')} ms ({call.get('prompt_tokens')} prompt tokens)")
    if vision:
        from ap_autopilot.loaders import load_document
        from ap_autopilot.models import VisionExtraction

        sample = Path(__file__).parent / "data" / "invoices_extra" / "invoice_2011_photo.jpg"
        try:
            read = llm.structured(role="check.vision", system="Read this invoice image. Transcribe it and extract the invoice.",
                                  user="Attached: one invoice photo.", schema=VisionExtraction,
                                  images=load_document(sample).images)
        except Exception as exc:
            _explain_llm_failure(llm, exc, vision=True)
            return 1
        call = llm.calls[-1]
        ok = (read.invoice.vendor_name or "").startswith("Reliable") and read.invoice.total == 2000
        console.print(f"[{'green' if ok else 'yellow'}]{'OK' if ok else 'CHECK'}[/] vision via {call.get('model')}: "
                      f"read vendor '{read.invoice.vendor_name}', total {read.invoice.total} "
                      f"(expected Reliable Components Inc., 2000.0) in {call.get('latency_ms')} ms")
        if not ok:
            console.print("[yellow]If the model can't take images, set XAI_VISION_MODEL in .env to a vision-capable Grok model.[/]")
            return 1
    return 0


def _explain_llm_failure(llm, exc: Exception, vision: bool = False) -> None:
    """Turn a failed check call into a next step instead of a traceback."""
    detail = next((c["note"] for c in reversed(llm.calls) if c.get("note")), str(exc))
    text = detail.lower()
    if "401" in text or "403" in text or "auth" in text or "api key" in text:
        hint = "The key was refused: re-copy XAI_API_KEY from console.x.ai and check Billing has credits."
    elif "model" in text and ("not found" in text or "does not exist" in text or "404" in text or "invalid" in text):
        hint = "Model name not recognised: copy an exact model ID from console.x.ai Models into XAI_MODEL."
    elif vision:
        hint = "This model may not accept images: set XAI_VISION_MODEL in .env to a vision-capable model."
    elif "connect" in text or "timeout" in text:
        hint = "Could not reach the API: check your internet connection and XAI_BASE_URL."
    else:
        hint = "See the error above; docs/LOCAL_SETUP.md has a troubleshooting table."
    console.print(f"[red]FAILED[/] {llm.provider}:{llm.model}{' (vision)' if vision else ''}: {detail[:300]}")
    console.print(f"[yellow]{hint}[/]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acme Corp multi-agent invoice processing")
    parser.add_argument("--invoice_path", "--invoice-path", dest="invoice_path", help="process one invoice file")
    parser.add_argument("--batch", help="process every invoice in a folder (sorted = arrival order)")
    parser.add_argument("--review", action="store_true", help="work the human approval queue")
    parser.add_argument("--outbox", action="store_true", help="read and send the drafted messages")
    parser.add_argument("--eval", action="store_true", help="score the system on eval/expected_outcomes.yaml")
    parser.add_argument("--roi", action="store_true", help="print the business case (three scenarios)")
    parser.add_argument("--compare", action="store_true", help="scorecard: rules-only vs LLM on the labelled eval set")
    parser.add_argument("--runs", type=int, default=1, help="with --compare: LLM runs, to measure stability (try 3)")
    parser.add_argument("--price-in", type=float, default=3.0, help="with --compare: USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=15.0, help="with --compare: USD per 1M output tokens")
    parser.add_argument("--bench", action="store_true", help="before/after throughput benchmark (simulated unless --live)")
    parser.add_argument("--live", action="store_true", help="with --bench: real LLM calls (needs a key)")
    parser.add_argument("--workers", type=int, default=None, help="parallel reads in a batch (default AP_READ_WORKERS=4)")
    parser.add_argument("--check-llm", action="store_true", help="send one test request to the configured LLM")
    parser.add_argument("--vision", action="store_true", help="with --check-llm: also test reading an invoice image")
    parser.add_argument("--reset-db", action="store_true", help="recreate inventory.db with seed data")
    parser.add_argument("--llm", default=None, choices=["auto", "grok", "openai", "offline"], help="LLM provider")
    parser.add_argument("--interactive", action="store_true", help="ask a human inline when an invoice is escalated")
    parser.add_argument("--verbose", "-v", action="store_true", help="show info findings and critique rounds")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of panels")
    parser.add_argument("--log-events", action="store_true", help="stream structured JSON events to stderr as each stage runs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    settings.log_to_console = settings.log_to_console or args.log_events
    provider = args.llm or settings.llm_provider

    if args.reset_db:
        Database(settings.db_path).init(reset=True)
        if not args.json:
            console.print(f"[green]Database reset:[/] {settings.db_path}")
        if not (args.invoice_path or args.batch or args.review or args.eval):
            return 0
    if args.eval:
        return cmd_eval(settings, provider)
    if args.roi:
        return cmd_roi()
    if args.workers is not None:
        settings.read_workers = max(1, args.workers)
    if args.bench:
        return cmd_bench(settings, args.live, settings.read_workers if settings.read_workers > 1 else 4,
                         args.price_in, args.price_out)
    if args.compare:
        return cmd_compare(settings, provider, args.runs, args.price_in, args.price_out)
    if args.check_llm:
        return cmd_check_llm(settings, provider, vision=args.vision)

    try:
        llm = build_llm(settings, provider)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    pipeline = InvoicePipeline(settings, llm=llm, reviewer=interactive_reviewer if args.interactive else None)

    if args.review:
        cmd_review(pipeline)
        return 0
    if args.outbox:
        cmd_outbox(pipeline)
        return 0
    target = args.invoice_path or args.batch
    if not target:
        parser.print_help()
        return 1
    files = collect(target)
    if not files:
        console.print(f"[red]No invoice files found at {target}[/]")
        return 1

    if not args.json:
        console.print(f"[dim]LLM: {llm.provider}:{llm.model} | DB: {settings.db_path}[/]")
    outcomes = []
    for outcome in pipeline.iter_batch(files):   # parallel reads, decisions in arrival order
        outcomes.append(outcome)
        if not args.json:
            render_outcome(console, outcome, verbose=args.verbose)
    if args.json:
        payload = [json.loads(o.model_dump_json(exclude={"extraction": {"raw_text"}})) for o in outcomes]
        print(json.dumps({"results": payload, "summary": summarize(outcomes)}, indent=2, default=str))
    elif len(outcomes) > 1:
        render_batch(console, outcomes)
    return 0 if all(o.status != "FAILED" for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
