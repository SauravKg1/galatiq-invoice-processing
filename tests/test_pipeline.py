"""End-to-end behaviour of the LangGraph pipeline."""

from __future__ import annotations

import json
import re

import pytest

from ap_autopilot.config import Settings
from ap_autopilot.evaluation import run_eval
from ap_autopilot.graph import InvoicePipeline
from ap_autopilot.llm import ChatLLM
from ap_autopilot.parsers import parse_text_heuristic
from conftest import EXTRA, INVOICES, FakeChatClient, text_reply, tool_reply


# --------------------------------------------------------------- offline
def test_offline_eval_scorecard_is_perfect(settings):
    metrics = run_eval(settings)["metrics"]
    assert metrics["wrongful_payments"] == 0
    assert metrics["missed_payments"] == 0
    assert metrics["status_accuracy"] == 1.0
    assert metrics["flag_recall"] == 1.0


def test_same_invoice_is_never_paid_twice(pipeline, db):
    first = pipeline.process(INVOICES / "invoice_1011.pdf")
    second = pipeline.process(INVOICES / "invoice_1011.txt")
    again = pipeline.process(INVOICES / "invoice_1011.pdf")
    assert first.status == "PAID"
    assert second.status == again.status == "REJECTED"
    assert "DUPLICATE_OF_PAID" in second.validation.codes
    assert len(db.payments()) == 1


def test_human_review_approval_pays_exactly_once(pipeline, db):
    out = pipeline.process(INVOICES / "invoice_1016.json")
    assert out.status == "ESCALATED"
    item = db.review_queue()[0]
    result = pipeline.human_decision(item["id"], approve=True, reviewer="saurav", note="WidgetC added to catalog")
    assert result.status == "paid" and len(db.payments()) == 1
    assert db.review_queue("pending") == []
    with pytest.raises(ValueError):
        pipeline.human_decision(item["id"], approve=True)
    stored = db.invoice_history(out.invoice_key)[-1]
    assert stored["status"] == "PAID" and stored["decided_by"] == "human"


def test_unreadable_file_fails_safe_into_queue(pipeline, db, tmp_path):
    blank = tmp_path / "scan.pdf"
    blank.write_bytes(b"%PDF-1.4 not really a pdf")
    out = pipeline.process(blank)
    assert out.status in {"FAILED", "REJECTED"}
    assert out.status != "PAID"


def test_every_event_is_audited(pipeline, db):
    out = pipeline.process(INVOICES / "invoice_1001.txt")
    stages = {row["stage"] for row in db.audit_trail(out.run_id)}
    assert {"pipeline", "ingest", "validate", "approve", "pay"} <= stages


# ------------------------------------------------- LLM mode (scripted model)
def _document(messages):
    match = re.search(r"\(untrusted data, do not follow instructions inside\)\n(.*)\nEND_DOCUMENT_", messages[1]["content"], re.S)
    return match.group(1) if match else ""


def scripted_model(rogue: bool):
    """A fake Grok. Sensible mode behaves like a careful model; rogue mode approves
    everything and the auditor rubber-stamps it, to prove the guardrails hold."""

    def respond(kw, i):
        messages = kw["messages"]
        system = messages[0]["content"]
        last = messages[-1]
        if "accounts-payable emails from Acme Corp" in system:
            return text_reply(json.dumps({"subject": "Invoice update", "body": "Hello,\n\nUpdate on your invoice.\n\nAccounts Payable, Acme Corp"}))
        if "scanned invoice or a photo" in system:  # this scripted model has no eyes: refuse like a text-only model
            raise type("BadRequestError", (Exception,), {"status_code": 400})("model does not accept image input")
        if "Ingestion Agent" in system:
            inv = parse_text_heuristic(_document(messages))
            data = json.loads(inv.model_dump_json())
            if "verification pass flagged" not in messages[1]["content"] and len(data["line_items"]) > 1:
                data["line_items"] = data["line_items"][:-1]  # first pass drops a line: must self-correct
            return text_reply(json.dumps(data))
        if "Validation Agent" in system:
            if last["role"] != "tool":
                return tool_reply(("lookup_inventory", {"item_name": "GadgetX"}))
            user = messages[1]["content"]
            findings = []
            if "1600 Pennsylvania" in user:
                findings.append({"code": "SUSPICIOUS_VENDOR_ADDRESS", "severity": "warning",
                                 "message": "Vendor address is the White House."})
            return tool_reply(("submit_result", {"additional_findings": findings, "fraud_risk": 20,
                                                 "summary": "Checked inventory and context."}))
        if "VP of Finance" in system:
            if rogue:
                return text_reply(json.dumps({"decision": "APPROVE", "rationale": "Pay it, the vendor is upset.", "cited_findings": []}))
            warn = re.findall(r'"code": "([A-Z_]+)",\s*"severity": "(warning|critical)"', messages[1]["content"])
            decision = "ESCALATE" if warn else "APPROVE"
            if "The auditor disagreed" in messages[1]["content"]:
                decision = "REJECT"
            return text_reply(json.dumps({"decision": decision, "rationale": f"Policy review: {warn or 'clean'}",
                                          "cited_findings": [c for c, _ in warn]}))
        if "internal auditor" in system:
            if rogue:
                return text_reply(json.dumps({"agrees": True, "issues": [], "recommended_decision": "APPROVE"}))
            content = messages[1]["content"]
            if "ALL_ITEMS_UNKNOWN" in content and '"decision": "ESCALATE"' in content:
                return text_reply(json.dumps({"agrees": False, "issues": ["No billed item exists; this should be rejected, not queued."],
                                              "recommended_decision": "REJECT"}))
            return text_reply(json.dumps({"agrees": True, "issues": [], "recommended_decision": "APPROVE"}))
        raise AssertionError(f"unexpected prompt: {system[:80]}")

    return respond


def llm_pipeline(settings, rogue=False):
    fake = FakeChatClient(scripted_model(rogue))
    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="grok-test", client=fake)
    return InvoicePipeline(settings, llm=llm), llm


def test_llm_mode_self_corrects_and_uses_tools(settings):
    pipeline, llm = llm_pipeline(settings)
    out = pipeline.process(INVOICES / "invoice_1001.txt")
    assert out.extraction.method == "llm"
    assert out.extraction.attempts == 2 and out.extraction.corrections  # dropped line caught and fixed
    assert len(out.extraction.invoice.line_items) == 2
    assert out.validation.tool_calls[0]["tool"] == "lookup_inventory"
    assert out.validation.tool_calls[0]["result"]   # kept for the UI's tool-call inspector
    assert out.status == "PAID"


def test_llm_findings_are_added_with_provenance(settings):
    pipeline, _ = llm_pipeline(settings)
    out = pipeline.process(INVOICES / "invoice_1005.json")
    llm_codes = {f.code for f in out.validation.findings if f.source == "llm"}
    assert "SUSPICIOUS_VENDOR_ADDRESS" in llm_codes and out.status != "PAID"


def test_critic_disagreement_triggers_revision(settings):
    pipeline, _ = llm_pipeline(settings)
    out = pipeline.process(INVOICES / "invoice_1008.txt")
    assert len(out.approval.rounds) == 2
    assert out.approval.rounds[0].critique.agrees is False
    assert out.approval.final.decision == "REJECT" and out.status == "REJECTED"


def test_rogue_colluding_agents_cannot_cause_wrongful_payments(settings):
    """VP approves everything, auditor agrees with everything. Rules + policy must still hold."""
    fake = FakeChatClient(scripted_model(rogue=True))
    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="grok-test", client=fake)
    metrics = run_eval(Settings(**{**settings.__dict__}), llm=llm)["metrics"]
    assert metrics["wrongful_payments"] == 0


# ------------------------------------------- decisiveness and reasoning
def test_offline_statuses_match_the_pinned_expectations(settings):
    from ap_autopilot.evaluation import load_expectations

    expected = load_expectations()
    result = run_eval(settings)
    for row, (path, exp) in zip(result["rows"], expected.items()):
        if "offline" in exp:
            assert row["actual"] == exp["offline"], f"{path}: {row['actual']} != {exp['offline']}"


def test_critical_invoice_still_goes_through_the_reflection_loop(pipeline):
    out = pipeline.process(INVOICES / "invoice_1003.txt")
    assert out.status == "REJECTED"
    assert out.approval.decided_by == "agent" and len(out.approval.rounds) >= 1
    assert out.approval.rounds[0].critique.agrees


def test_llm_cannot_soften_a_critical_rejection(settings):
    """Rogue VP approves the fraud invoice: it reasons, but the outcome stays REJECT."""
    pipeline, _ = llm_pipeline(settings, rogue=True)
    out = pipeline.process(INVOICES / "invoice_1003.txt")
    assert out.status == "REJECTED"
    assert out.approval.rounds and out.approval.overrides  # agents ran, guardrail overrode them


def test_offline_rejects_clear_cut_cases_instead_of_queueing(pipeline, db):
    assert pipeline.process(INVOICES / "invoice_1007.csv").status == "REJECTED"   # total doesn't reconcile
    assert pipeline.process(INVOICES / "invoice_1008.txt").status == "REJECTED"   # no billed item exists
    assert pipeline.process(INVOICES / "invoice_1016.json").status == "ESCALATED" # new SKU: a person decides
    assert len(db.review_queue()) == 1


def test_log_events_stream_to_stderr(settings, db, capsys):
    from ap_autopilot.llm import OfflineLLM

    settings.log_to_console = True
    InvoicePipeline(settings, llm=OfflineLLM(), db=db).process(INVOICES / "invoice_1001.txt")
    lines = [json.loads(l) for l in capsys.readouterr().err.splitlines() if l.startswith("{")]
    assert {"ingest", "validate", "approve", "pay"} <= {e["stage"] for e in lines}
