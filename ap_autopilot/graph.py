"""Orchestration: a LangGraph StateGraph over the four agents.

    ingest -> validate -> approve --APPROVE--> pay ------\
                                  --REJECT---> reject ---+--> communicate -> finalize -> END
                                  --ESCALATE-> escalate -/
    (any node error) ----------------------------------> fail -> communicate

Agents never call each other. They read and write typed fields on a shared
state and the graph routes on those fields, so every hop is inspectable,
replayable and testable in isolation. Every node is wrapped so an exception
becomes a FAILED outcome in the human queue, never a silently dropped invoice.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.approval import ApprovalAgent
from .agents.communication import CommunicationAgent
from .agents.ingestion import IngestionAgent
from .agents.payment import PaymentAgent
from .agents.validation import ValidationAgent
from .config import Settings
from .db import Database
from .llm import LLMClient, build_llm
from .models import (ApprovalResult, Decision, Draft, ExtractionResult, Invoice, PaymentResult, ProcessingOutcome,
                     ValidationReport)
from .normalize import content_hash, invoice_key
from .observability import Tracer, new_run_id
from .policy import fast_path_reason

# reviewer(outcome_so_far) -> ("approve" | "reject" | None, note)
Reviewer = Callable[[dict[str, Any]], tuple[Optional[str], str]]


class APState(TypedDict, total=False):
    run_id: str
    path: str
    invoice_key: Optional[str]
    extraction: ExtractionResult
    validation: ValidationReport
    approval: ApprovalResult
    payment: PaymentResult
    status: str
    error: str
    drafts: list[Draft]


class InvoicePipeline:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLMClient] = None,
                 db: Optional[Database] = None, reviewer: Optional[Reviewer] = None):
        self.settings = settings or Settings()
        self.db = (db or Database(self.settings.db_path)).ensure()
        self.llm = llm or build_llm(self.settings)
        self.reviewer = reviewer
        self.ingestion = IngestionAgent(self.llm, self.db, self.settings)
        self.validation = ValidationAgent(self.llm, self.db, self.settings)
        self.approval = ApprovalAgent(self.llm, self.settings)
        self.payment = PaymentAgent(self.db, self.settings)
        self.communication = CommunicationAgent(self.llm, self.db, self.settings)
        self._tracer: Optional[Tracer] = None
        self._prefetched: dict[str, tuple[Optional[ExtractionResult], Optional[BaseException], list[dict], float]] = {}
        self._prefetch_calls: dict[str, list[dict]] = {}
        self.batch_wait_s = 0.0   # time the decision stage spent waiting for a parallel read (bench uses it)
        self.graph = self._build_graph()

    # ------------------------------------------------------------ public
    def process(self, path: str | Path) -> ProcessingOutcome:
        """Run one invoice end to end. Same code path as `process_stream`, so they can't diverge."""
        outcome: Optional[ProcessingOutcome] = None
        for event in self.process_stream(path):
            if event["type"] == "outcome":
                outcome = event["outcome"]
        assert outcome is not None
        return outcome

    def process_stream(self, path: str | Path) -> Iterator[dict[str, Any]]:
        """Yield one event per graph step as it finishes (LangGraph `stream_mode="updates"`),
        then a final {"type": "outcome"} event. Used by the UI's live trace."""
        run_id = new_run_id()
        self._tracer = Tracer(run_id, self.db, self.settings.log_dir, echo=self.settings.log_to_console)
        calls_before = len(self.llm.calls)
        self._tracer.event("pipeline", "received", file=str(path), llm=f"{self.llm.provider}:{self.llm.model}")
        state: dict[str, Any] = {"run_id": run_id, "path": str(path)}
        yield {"type": "start", "run_id": run_id, "file": Path(path).name}
        for update in self.graph.stream(dict(state), stream_mode="updates"):
            for node, change in update.items():
                state.update(change or {})
                if node == "finalize":
                    continue
                yield {"type": "node", "node": node, "ms": self._tracer.timings_ms.get(node, 0.0),
                       "summary": describe_step(node, state), "state": dict(state)}
        outcome = ProcessingOutcome(
            run_id=run_id, source_file=Path(path).name, invoice_key=state.get("invoice_key"),
            status=state.get("status", "FAILED"), extraction=state.get("extraction"),
            validation=state.get("validation"), approval=state.get("approval"), payment=state.get("payment"),
            error=state.get("error"), llm_mode=f"{self.llm.provider}:{self.llm.model}",
            timings_ms=dict(self._tracer.timings_ms),
            llm_calls=self._prefetch_calls.pop(str(path), []) + [c for c in self.llm.calls[calls_before:] if not c.get("tag")],
            drafts=state.get("drafts", []),
        )
        self._persist(outcome)
        self._tracer.event("pipeline", "completed", status=outcome.status, total_ms=outcome.total_ms)
        yield {"type": "outcome", "outcome": outcome}

    def process_batch(self, paths: list[str | Path], workers: Optional[int] = None) -> list[ProcessingOutcome]:
        return list(self.iter_batch(paths, workers))

    def iter_batch(self, paths: list[str | Path], workers: Optional[int] = None) -> Iterator[ProcessingOutcome]:
        """Process invoices in arrival order, yielding each outcome as it is decided.

        Reading is independent per file (it only looks up the catalog), so up to `workers`
        files are read in parallel ahead of the decision stage. Checking, deciding and
        paying stay strictly sequential: duplicate detection, PO consumption and
        "paid once, ever" depend on every earlier invoice, so they must see the ledger
        in arrival order. Same outcomes as one-at-a-time processing, less waiting.
        """
        workers = self.settings.read_workers if workers is None else workers
        paths = [str(p) for p in paths]
        if workers <= 1 or len(paths) <= 1:
            for p in paths:
                yield self.process(p)
            return

        def read(path: str):
            start = time.perf_counter()
            with self.llm.tagged(path):
                try:
                    result, error = self.ingestion.run(path), None
                except Exception as exc:   # surfaced by the ingest node, which fails safe as usual
                    result, error = None, exc
            calls = [c for c in list(self.llm.calls) if c.get("tag") == path]
            return result, error, calls, (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="read") as pool:
            futures = [(p, pool.submit(read, p)) for p in paths]
            for p, future in futures:
                waited = time.perf_counter()
                self._prefetched[p] = future.result()
                self.batch_wait_s += time.perf_counter() - waited
                yield self.process(p)

    def human_decision(self, review_id: int, approve: bool, reviewer: str = "human", note: str = "") -> PaymentResult:
        """Resolve a queued escalation. Approval runs the same idempotent payment step."""
        item = self.db.get_review(review_id)
        if not item or item["status"] != "pending":
            raise ValueError(f"review item {review_id} is not pending")
        tracer = Tracer(item["run_id"], self.db, self.settings.log_dir, echo=self.settings.log_to_console)
        tracer.invoice_key = item["invoice_key"]
        record = next((r for r in self.db.invoice_history(item["invoice_key"]) if r["run_id"] == item["run_id"]), None)
        if approve:
            stored = ProcessingOutcome.model_validate_json(self._stored_report(item["run_id"]))
            if not stored.extraction:
                raise ValueError("this invoice could not be read; reject it or re-process a readable copy")
            result = self.payment.pay(stored.extraction.invoice, item["invoice_key"], item["run_id"])
            status = "PAID" if result.status == "paid" else ("REJECTED" if result.status == "skipped_duplicate" else "ESCALATED")
            if result.status == "paid":
                self._record_po_invoiced(stored.validation)
        else:
            result = PaymentResult(status="rejected", vendor=item["vendor"], amount=item["amount"],
                                   currency=item["currency"], message=f"Rejected by {reviewer}: {note}")
            status = "REJECTED"
        self.db.resolve_review(review_id, "approved" if approve else "rejected", reviewer, note)
        if approve and result.status == "paid":
            try:
                for d in self.communication.run("PAID", stored.extraction, stored.validation, stored.approval, result,
                                                source_file=item["source_file"] or ""):
                    self.db.add_draft(run_id=item["run_id"], invoice_key=item["invoice_key"], source_file=item["source_file"],
                                      audience=d.audience, kind=d.kind, to_addr=d.to, subject=d.subject, body=d.body,
                                      author=d.author, note=d.note)
            except Exception as exc:  # a failed draft never undoes a payment
                tracer.event("communicate", "error", error=str(exc))
        if record:
            self.db.update_invoice_status(item["invoice_key"], item["run_id"], status, "human", f"{reviewer}: {note}")
        tracer.event("human_review", "decided", review_id=review_id, approve=approve, reviewer=reviewer,
                     note=note, payment_status=result.status)
        return result

    # ------------------------------------------------------------- graph
    def _build_graph(self):
        g = StateGraph(APState)
        g.add_node("ingest", self._guard("ingest", self._ingest))
        g.add_node("validate", self._guard("validate", self._validate))
        g.add_node("approve", self._guard("approve", self._approve))
        g.add_node("pay", self._guard("pay", self._pay))
        g.add_node("reject", self._guard("reject", self._reject))
        g.add_node("escalate", self._guard("escalate", self._escalate))
        g.add_node("fail", self._fail)
        g.add_node("communicate", self._communicate)
        g.add_node("finalize", lambda state: {})

        g.add_edge(START, "ingest")
        g.add_conditional_edges("ingest", self._ok_or_fail("validate"), ["validate", "fail"])
        g.add_conditional_edges("validate", self._ok_or_fail("approve"), ["approve", "fail"])
        g.add_conditional_edges("approve", self._route_decision, ["pay", "reject", "escalate", "fail"])
        for node in ("pay", "reject", "escalate"):
            g.add_conditional_edges(node, self._ok_or_fail("communicate"), ["communicate", "fail"])
        g.add_edge("fail", "communicate")
        g.add_edge("communicate", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    @staticmethod
    def _ok_or_fail(next_node: str) -> Callable[[APState], str]:
        return lambda state: "fail" if state.get("error") else next_node

    @staticmethod
    def _route_decision(state: APState) -> str:
        if state.get("error"):
            return "fail"
        return {"APPROVE": "pay", "REJECT": "reject", "ESCALATE": "escalate"}[state["approval"].final.decision]

    def _guard(self, name: str, fn: Callable[[APState], dict[str, Any]]) -> Callable[[APState], dict[str, Any]]:
        def node(state: APState) -> dict[str, Any]:
            try:
                with self._tracer.stage(name):
                    return fn(state)
            except Exception as exc:
                self._tracer.event(name, "exception", traceback=traceback.format_exc(limit=4))
                return {"error": f"{name}: {type(exc).__name__}: {exc}"}
        return node

    # ------------------------------------------------------------- nodes
    def _ingest(self, state: APState) -> dict[str, Any]:
        pre = self._prefetched.pop(state["path"], None)
        if pre is not None:   # read in parallel by iter_batch
            extraction, error, calls, read_ms = pre
            self._prefetch_calls[state["path"]] = calls
            self._tracer.event("ingest", "prefetched", read_ms=round(read_ms, 1))
            if error is not None:
                raise error
        else:
            extraction = self.ingestion.run(state["path"])
        inv = extraction.invoice
        key = invoice_key(inv.vendor_name, inv.invoice_number)
        self._tracer.invoice_key = key
        self._tracer.event("ingest", "extracted", method=extraction.method, attempts=extraction.attempts,
                           confidence=extraction.confidence, vendor=inv.vendor_name, total=inv.total,
                           items=len(inv.line_items), corrections=extraction.corrections)
        return {"extraction": extraction, "invoice_key": key}

    def _validate(self, state: APState) -> dict[str, Any]:
        report = self.validation.run(state["extraction"])
        self._tracer.event("validate", "report", risk_score=report.risk_score,
                           critical=[f.code for f in report.critical], warnings=[f.code for f in report.warnings],
                           llm_findings=[f.code for f in report.findings if f.source == "llm"],
                           tool_calls=len(report.tool_calls))
        return {"validation": report}

    def _approve(self, state: APState) -> dict[str, Any]:
        fast = fast_path_reason(state["extraction"], state["validation"], self.settings)
        result = self.approval.run(state["extraction"].invoice, state["validation"], fast_path=fast is not None)
        self._tracer.event("approve", "decision", decision=result.final.decision, decided_by=result.decided_by,
                           tier=result.policy_tier, rounds=len(result.rounds), overrides=result.overrides,
                           fast_path=fast if result.fast_path else None)
        return {"approval": result}

    def _pay(self, state: APState) -> dict[str, Any]:
        result = self.payment.pay(state["extraction"].invoice, state.get("invoice_key"), state["run_id"])
        self._tracer.event("pay", result.status, amount=result.amount, vendor=result.vendor, reference=result.reference)
        if result.status == "paid":
            self._record_po_invoiced(state.get("validation"))
            return {"payment": result, "status": "PAID"}
        if result.status == "skipped_duplicate":
            return {"payment": result, "status": "REJECTED"}
        return self._escalate({**state, "payment": result}) | {"payment": result}

    def _reject(self, state: APState) -> dict[str, Any]:
        result = self.payment.reject(state["extraction"].invoice, state["approval"])
        self._tracer.event("reject", "logged", reason=state["approval"].final.rationale)
        return {"payment": result, "status": "REJECTED"}

    def _escalate(self, state: APState) -> dict[str, Any]:
        approval = state.get("approval") or ApprovalResult(
            final=Decision(decision="ESCALATE", rationale=state.get("error", "manual handling")),
            policy_tier="standard", decided_by="policy")
        invoice = state["extraction"].invoice if state.get("extraction") else Invoice()
        queued = self.payment.escalate(invoice, approval, state.get("invoice_key"), state["run_id"],
                                       Path(state["path"]).name)
        self._tracer.event("escalate", "queued", reference=queued.reference, reason=approval.final.rationale)
        if self.reviewer and state.get("extraction"):
            verdict, note = self.reviewer({"invoice": invoice, "approval": approval,
                                           "validation": state.get("validation"), "file": state["path"]})
            if verdict in {"approve", "reject"}:
                review_id = int(queued.reference.split("-")[1])
                self._persist_partial(state, approval)  # human decision needs the stored record
                result = self.human_decision(review_id, verdict == "approve", "interactive-reviewer", note)
                return {"payment": result, "status": "PAID" if result.status == "paid" else "REJECTED",
                        "approval": approval.model_copy(update={"decided_by": "human"})}
        return {"payment": queued, "status": "ESCALATED"}

    def _record_po_invoiced(self, report: Optional[ValidationReport]) -> None:
        """After payment, consume the PO's open quantity so a later invoice can't bill it again."""
        match = report.po_match if report else None
        if not match or not match.get("found"):
            return
        for line in match["lines"]:
            if line.get("po_qty") is not None and line["invoiced_qty"]:
                self.db.add_po_invoiced(match["po_number"], line["item"], line["invoiced_qty"])

    def _communicate(self, state: APState) -> dict[str, Any]:
        """Draft the messages this outcome should trigger. Drafting problems are logged, never
        allowed to change the outcome: an invoice is never un-paid because an email failed."""
        start = time.perf_counter()
        try:
            drafts = self.communication.run(state.get("status", "FAILED"), state.get("extraction"), state.get("validation"),
                                            state.get("approval"), state.get("payment"), state.get("error"),
                                            Path(state["path"]).name)
            saved = []
            for d in drafts:
                draft_id = self.db.add_draft(run_id=state["run_id"], invoice_key=state.get("invoice_key"),
                                             source_file=Path(state["path"]).name, audience=d.audience, kind=d.kind,
                                             to_addr=d.to, subject=d.subject, body=d.body, author=d.author, note=d.note)
                saved.append(d.model_copy(update={"id": draft_id}))
            self._tracer.event("communicate", "drafted", drafts=[f"{d.audience}:{d.kind}:{d.author}" for d in saved])
            return {"drafts": saved}
        except Exception as exc:
            self._tracer.event("communicate", "error", error=f"{type(exc).__name__}: {exc}")
            return {"drafts": []}
        finally:
            self._tracer.timings_ms["communicate"] = round((time.perf_counter() - start) * 1000, 1)

    def _fail(self, state: APState) -> dict[str, Any]:
        self._tracer.event("pipeline", "failed", error=state.get("error"))
        try:
            invoice = state["extraction"].invoice if state.get("extraction") else Invoice()
            approval = ApprovalResult(final=Decision(decision="ESCALATE", rationale=f"System error: {state.get('error')}"),
                                      policy_tier="standard", decided_by="policy")
            queued = self.payment.escalate(invoice, approval, state.get("invoice_key"), state["run_id"],
                                           Path(state["path"]).name)
            return {"status": "FAILED", "payment": queued}
        except Exception:
            return {"status": "FAILED"}

    # --------------------------------------------------------- persistence
    def _persist(self, outcome: ProcessingOutcome) -> None:
        if any(r["run_id"] == outcome.run_id for r in self.db.invoice_history(outcome.invoice_key)):
            return  # already written by the interactive-review path
        inv = outcome.extraction.invoice if outcome.extraction else None
        self.db.record_invoice(
            run_id=outcome.run_id, invoice_key=outcome.invoice_key,
            invoice_number=inv.invoice_number if inv else None, vendor=inv.vendor_name if inv else None,
            invoice_date=str(inv.invoice_date) if inv and inv.invoice_date else None,
            total=inv.total if inv else None, currency=(inv.currency if inv else None) or self.settings.base_currency,
            content_hash=content_hash(inv) if inv else None, source_file=outcome.source_file, status=outcome.status,
            risk_score=outcome.validation.risk_score if outcome.validation else None,
            decided_by=outcome.approval.decided_by if outcome.approval else None,
            rationale=outcome.approval.final.rationale if outcome.approval else outcome.error,
            report_json=outcome.model_dump_json(),
        )

    def _persist_partial(self, state: APState, approval: ApprovalResult) -> None:
        partial = ProcessingOutcome(run_id=state["run_id"], source_file=Path(state["path"]).name,
                                    invoice_key=state.get("invoice_key"), status="ESCALATED",
                                    extraction=state.get("extraction"), validation=state.get("validation"),
                                    approval=approval, llm_mode=f"{self.llm.provider}:{self.llm.model}")
        self._persist(partial)

    def _stored_report(self, run_id: str) -> str:
        with self.db.connect() as conn:
            row = conn.execute("SELECT report_json FROM invoices WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            raise ValueError(f"no stored record for run {run_id}")
        return row["report_json"]


# ------------------------------------------------------------------ trace text
NODE_ORDER = ["ingest", "validate", "approve", "pay", "reject", "escalate", "fail", "communicate"]
NODE_LABELS = {"ingest": "Read", "validate": "Check", "approve": "Decide", "pay": "Pay", "reject": "Reject",
               "escalate": "Hold for a person", "fail": "Fail safe", "communicate": "Tell"}


def _money(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def describe_step(node: str, state: dict[str, Any]) -> list[str]:
    """Plain-English lines describing what a step just did, from the state it produced."""
    if state.get("error") and node in {"ingest", "validate", "approve", "pay", "reject", "escalate"}:
        return [f"Error: {state['error']}"]
    lines: list[str] = []
    if node == "ingest" and state.get("extraction"):
        x = state["extraction"]
        how = {"structured": "parsed as structured data, no LLM needed", "llm": "extracted by the LLM",
               "heuristic": "parsed by the text rules", "vision": "read from the image by the vision model",
               "ocr": "read from the image by local OCR"}[x.method]
        lines.append(f"{x.source_format.upper()} {how}: {len(x.invoice.line_items)} line item(s), "
                     f"{x.invoice.vendor_name or 'no vendor'}, total {_money(x.invoice.total)}.")
        for c in x.corrections:
            lines.append(f"Self-correction: {c}")
        if x.attempts > 1:
            lines.append(f"Re-read {x.attempts - 1} time(s) after verification.")
    elif node == "validate" and state.get("validation"):
        v = state["validation"]
        lines.append(f"{len(v.critical)} critical, {len(v.warnings)} warning(s), fraud risk {v.risk_score}/100.")
        seen: dict[str, str] = {}
        for f in v.critical + v.warnings:
            seen.setdefault(f.code, f.source)
        for code, source in list(seen.items())[:5]:
            lines.append(f"Flag: {code}" + (" (raised by the investigator)" if source == "llm" else ""))
        for call in v.tool_calls:
            lines.append(f"Investigator called {call['tool']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")
        if v.po_match and v.po_match.get("found"):
            bad = [ln for ln in v.po_match["lines"] if ln["status"] != "match"]
            lines.append(f"3-way match vs {v.po_match['po_number']}: " + ("all lines match" if not bad else f"{len(bad)} line issue(s)"))
    elif node == "approve" and state.get("approval"):
        a = state["approval"]
        if a.fast_path:
            lines.append("Fast path: clean structured invoice under the limit; VP approved, auditor round skipped.")
        for r in a.rounds:
            lines.append(f"Round {r.round}: VP proposed {r.decision.decision}; auditor "
                         + ("agreed." if r.critique.agrees else f"disagreed ({'; '.join(r.critique.issues)[:120]})."))
        for o in a.overrides:
            lines.append(f"Guardrail: {o}")
        lines.append(f"Decision: {a.final.decision} by {a.decided_by}.")
    elif node in {"pay", "reject", "escalate", "fail"} and state.get("payment"):
        lines.append(state["payment"].message)
    elif node == "communicate":
        drafts = state.get("drafts") or []
        lines.append(f"{len(drafts)} draft(s): " + ", ".join(f"{d.kind.replace('_', ' ')} to {d.to}" for d in drafts)
                     if drafts else "No message needed.")
    return lines or ["Done."]


def flow_dot(visited: list[str], current: Optional[str] = None, status: Optional[str] = None) -> str:
    """Graphviz DOT of the agent graph: the path taken is highlighted, untaken branches fade."""
    blue, orange, ink, muted, pale = "#2F6FA3", "#C4702B", "#1F2A44", "#9AA3B2", "#E6EAE7"
    done = set(visited)
    terminal = {"pay", "reject", "escalate", "fail"}
    branch_taken = next((n for n in visited if n in terminal), None)

    def node(name: str) -> str:
        label = NODE_LABELS[name]
        if name == current:
            return f'{name} [label="{label}", style="filled,bold", fillcolor="{orange}", fontcolor="white", color="{orange}"]'
        if name in done:
            return f'{name} [label="{label}", style=filled, fillcolor="{blue}", fontcolor="white", color="{blue}"]'
        faded = branch_taken is not None and name in terminal
        return (f'{name} [label="{label}", style="filled,dashed", fillcolor="{pale}", fontcolor="{muted}", color="{muted}"]'
                if faded else f'{name} [label="{label}", style=filled, fillcolor="{pale}", fontcolor="{ink}", color="{muted}"]')

    edges = [("ingest", "validate"), ("validate", "approve"), ("approve", "pay"), ("approve", "reject"),
             ("approve", "escalate"), ("pay", "communicate"), ("reject", "communicate"), ("escalate", "communicate")]
    if "fail" in done:
        edges += [(visited[visited.index("fail") - 1] if visited.index("fail") else "ingest", "fail"), ("fail", "communicate")]
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;',
             'node [shape=box, style=filled, fontname="Helvetica", fontsize=11, margin="0.18,0.08", penwidth=1.4];',
             'edge [arrowsize=0.6, penwidth=1.2];']
    shown = NODE_ORDER if "fail" in done else [n for n in NODE_ORDER if n != "fail"]
    lines += [node(n) for n in shown]
    for a, b in edges:
        on_path = a in done and b in done and (b not in terminal or b == branch_taken) and (a not in terminal or a == branch_taken)
        style = f'color="{blue}", penwidth=2.2' if on_path else f'color="{muted}", style=dashed'
        lines.append(f"{a} -> {b} [{style}];")
    lines.append("}")
    return "\n".join(lines)
