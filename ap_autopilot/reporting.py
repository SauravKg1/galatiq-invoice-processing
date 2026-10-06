"""Console rendering and batch metrics (shared by the CLI, eval and UI)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rich.console import Console
from rich.markup import escape as esc
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import ProcessingOutcome, Severity

STATUS_STYLE = {"PAID": "bold green", "REJECTED": "bold red", "ESCALATED": "bold yellow", "FAILED": "bold magenta"}
SEVERITY_STYLE = {Severity.CRITICAL: "red", Severity.WARNING: "yellow", Severity.INFO: "dim"}


def _money(x: Any, currency: str = "USD") -> str:
    if x is None:
        return "n/a"
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{'-' if x < 0 else ''}{symbol}{abs(x):,.2f}"


def render_outcome(console: Console, o: ProcessingOutcome, verbose: bool = False) -> None:
    inv = o.extraction.invoice if o.extraction else None
    head = Text()
    head.append(f"{o.source_file}  ", style="bold")
    head.append(o.status, style=STATUS_STYLE.get(o.status, "bold"))
    if inv:
        head.append(f"   {inv.vendor_name or 'unknown vendor'} | {inv.invoice_number or 'no number'} | "
                    f"{_money(inv.total, inv.currency or 'USD')}")
    lines: list[Text | str] = [head]

    if o.extraction:
        e = o.extraction
        lines.append(f"[cyan]1 Ingest[/]    {e.source_format.upper()} via {e.method}, {len(inv.line_items)} line items, "
                     f"confidence {e.confidence:.0%}, attempts {e.attempts}")
        for c in e.corrections[: 6 if verbose else 2]:
            lines.append(f"            [dim]self-correction: {esc(c)}[/]")
    if o.validation:
        v = o.validation
        lines.append(f"[cyan]2 Validate[/]  {len(v.critical)} critical, {len(v.warnings)} warning{'s' if len(v.warnings) != 1 else ''}, "
                     f"fraud risk {v.risk_score}/100" + (f", investigator {v.llm_risk}/100" if v.llm_risk is not None else ""))
        shown = [f for f in v.findings if verbose or f.severity != Severity.INFO]
        for f in shown:
            src = " (llm)" if f.source == "llm" else ""
            lines.append(Text.assemble("            ", (f"{f.severity.value.upper():8}", SEVERITY_STYLE[f.severity]),
                                       f" {f.code}{src}: ", (f.message, "default")))
        if v.tool_calls:
            lines.append(f"            [dim]investigator tool calls: {', '.join(c['tool'] for c in v.tool_calls)}[/]")
        if v.po_match and v.po_match.get("found"):
            m = v.po_match
            bad = [ln for ln in m["lines"] if ln["status"] != "match"]
            how = " (matched by items)" if m.get("inferred") else ""
            verdict = "[green]all lines match[/]" if not bad else "[yellow]" + esc("; ".join(f"{ln['item']}: {ln['status']}" for ln in bad)) + "[/]"
            lines.append(f"            3-way match vs {m['po_number']}{how}: {verdict}")
    if o.approval:
        a = o.approval
        lines.append(f"[cyan]3 Approve[/]   {a.final.decision} by {a.decided_by} ({a.policy_tier}), "
                     + ("fast path, auditor skipped (clean invoice)" if a.fast_path else f"{len(a.rounds)} critique round(s)"))
        if verbose:
            for r in a.rounds:
                lines.append(f"            [dim]round {r.round}: VP {r.decision.decision}, auditor "
                             f"{'agrees' if r.critique.agrees else 'disagrees: ' + esc('; '.join(r.critique.issues))}[/]")
        lines.append(f"            [italic]{esc(a.final.rationale)}[/]")
        for ov in a.overrides:
            lines.append(f"            [magenta]guardrail: {esc(ov)}[/]")
    if o.payment:
        lines.append(f"[cyan]4 Payment[/]   {o.payment.status}: {esc(o.payment.message)}"
                     + (f" ({o.payment.reference})" if o.payment.reference else ""))
    if o.drafts:
        summary = ", ".join(f"{d.kind.replace('_', ' ')} to {d.to}" + (" (Grok)" if d.author == "llm" else "") for d in o.drafts)
        lines.append(f"[cyan]5 Tell[/]      {len(o.drafts)} draft(s) in the outbox: {esc(summary)}")
        if verbose:
            for d in o.drafts:
                lines.append(f"            [dim]--- {d.audience} | {esc(d.subject)}[/]")
                for body_line in d.body.splitlines():
                    lines.append(f"            [dim]{esc(body_line)}[/]")
                if d.note:
                    lines.append(f"            [magenta]{esc(d.note)}[/]")
    if o.error:
        lines.append(f"[magenta]error: {esc(o.error)}[/]")
    llm_ok = sum(1 for c in o.llm_calls if c.get("ok") and c.get("provider") != "offline")
    lines.append(f"[dim]run {o.run_id} | {o.total_ms:.0f} ms | {o.llm_mode} | {llm_ok} LLM calls[/]")
    console.print(Panel("\n".join(str(x) if isinstance(x, str) else x.markup for x in lines),
                        border_style=STATUS_STYLE.get(o.status, "white").split()[-1], expand=True))


def summarize(outcomes: list[ProcessingOutcome]) -> dict[str, Any]:
    status = Counter(o.status for o in outcomes)

    def total(s: str) -> float:
        return round(sum((o.amount or 0) for o in outcomes if o.status == s and (o.amount or 0) > 0), 2)

    touchless = sum(1 for o in outcomes if o.status in {"PAID", "REJECTED"})
    return {
        "invoices": len(outcomes),
        "paid": status.get("PAID", 0),
        "rejected": status.get("REJECTED", 0),
        "escalated": status.get("ESCALATED", 0),
        "failed": status.get("FAILED", 0),
        "amount_paid": total("PAID"),
        "amount_blocked": total("REJECTED"),
        "amount_held_for_review": total("ESCALATED"),
        "touchless_rate": round(touchless / len(outcomes), 3) if outcomes else 0.0,
        "avg_ms": round(sum(o.total_ms for o in outcomes) / len(outcomes), 1) if outcomes else 0.0,
        "llm_calls": sum(1 for o in outcomes for c in o.llm_calls if c.get("provider") != "offline"),
    }


def render_batch(console: Console, outcomes: list[ProcessingOutcome]) -> None:
    table = Table(title="Batch results", show_lines=False, header_style="bold")
    for col in ("File", "Vendor", "Amount", "Status", "Risk", "Top reason"):
        table.add_column(col, overflow="fold")
    for o in outcomes:
        inv = o.extraction.invoice if o.extraction else None
        v = o.validation
        top = (v.critical or v.warnings)[0].code if v and (v.critical or v.warnings) else ("clean" if v else (o.error or ""))
        table.add_row(o.source_file, (inv.vendor_name if inv else None) or "?",
                      _money(inv.total if inv else None, (inv.currency if inv else None) or "USD"),
                      Text(o.status, style=STATUS_STYLE.get(o.status, "")), str(v.risk_score if v else "-"), top)
    console.print(table)
    s = summarize(outcomes)
    console.print(
        f"[bold]{s['invoices']}[/] invoices | [green]{s['paid']} paid ({_money(s['amount_paid'])})[/] | "
        f"[red]{s['rejected']} rejected ({_money(s['amount_blocked'])} blocked)[/] | "
        f"[yellow]{s['escalated']} held for review ({_money(s['amount_held_for_review'])})[/] | "
        f"touchless {s['touchless_rate']:.0%} | avg {s['avg_ms']:.0f} ms/invoice"
    )
