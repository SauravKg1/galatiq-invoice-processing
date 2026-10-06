"""Throughput optimisations must never change a decision."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from ap_autopilot.bench import SimulatedLLM, render_markdown, run_bench
from ap_autopilot.config import Settings
from ap_autopilot.graph import InvoicePipeline
from ap_autopilot.llm import ChatLLM, OfflineLLM, build_llm, is_fast_role
from ap_autopilot.models import Decision
from conftest import EXTRA, INVOICES, FakeChatClient, text_reply
from test_pipeline import llm_pipeline

ALL = sorted(p for d in (INVOICES, EXTRA) for p in d.iterdir() if p.suffix.lower() in {".txt", ".json", ".csv", ".xml", ".pdf", ".eml"})


def _pipeline(tmp_path: Path, name: str, **overrides) -> InvoicePipeline:
    s = Settings(db_path=str(tmp_path / f"{name}.db"), log_dir=str(tmp_path / "logs"), llm_provider="offline", **overrides)
    p = InvoicePipeline(s, llm=OfflineLLM())
    p.db.init(reset=True)
    return p


# ------------------------------------------------------------ parallel reads
def test_parallel_batch_matches_one_at_a_time(tmp_path):
    seq = [o.status for o in _pipeline(tmp_path, "seq").process_batch(ALL, workers=1)]
    par = [o.status for o in _pipeline(tmp_path, "par").process_batch(ALL, workers=4)]
    assert par == seq            # duplicates, revisions and PO consumption all depend on order


def test_out_of_order_reads_are_decided_in_arrival_order(tmp_path):
    pipeline = _pipeline(tmp_path, "ooo")
    files = [INVOICES / "invoice_1004.json", INVOICES / "invoice_1004_revised.json", INVOICES / "invoice_1001.txt"]
    real = pipeline.ingestion.run
    delays = {str(files[0]): 0.3, str(files[1]): 0.0, str(files[2]): 0.1}   # first file finishes reading last

    def slow(path):
        time.sleep(delays[str(path)])
        return real(path)

    pipeline.ingestion.run = slow
    outcomes = pipeline.process_batch(files, workers=3)
    assert [o.source_file for o in outcomes] == [f.name for f in files]
    reference = _pipeline(tmp_path, "ref").process_batch(files, workers=1)
    assert [o.status for o in outcomes] == [o.status for o in reference]


def test_parallel_read_failure_still_fails_safe(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    outcomes = _pipeline(tmp_path, "fail").process_batch([INVOICES / "invoice_1001.txt", empty], workers=2)
    assert [o.status for o in outcomes] == ["PAID", "FAILED"]


def test_each_invoice_keeps_its_own_llm_calls_in_parallel(tmp_path):
    s = Settings(db_path=str(tmp_path / "sim.db"), log_dir=str(tmp_path / "logs"))
    pipeline = InvoicePipeline(s, llm=SimulatedLLM(routed=True, scale=0.0001))
    pipeline.db.init(reset=True)
    files = [INVOICES / n for n in ("invoice_1001.txt", "invoice_1002.txt", "invoice_1003.txt")]
    for o in pipeline.process_batch(files, workers=3):
        reads = [c for c in o.llm_calls if c["role"].startswith("ingestion.")]
        assert reads and all(Path(c["tag"]).name == o.source_file for c in reads)
        assert any(c["role"] == "approval.vp" for c in o.llm_calls)


# --------------------------------------------------------------- fast path
def test_clean_structured_invoice_skips_the_auditor(tmp_path):
    out = _pipeline(tmp_path, "fp").process(INVOICES / "invoice_1004.json")
    assert out.status == "PAID" and out.approval.fast_path and out.approval.rounds == []
    assert not any(c["role"] == "approval.critic" for c in out.llm_calls)


def test_fast_path_can_be_switched_off(tmp_path):
    out = _pipeline(tmp_path, "nofp", fast_path=False).process(INVOICES / "invoice_1004.json")
    assert out.status == "PAID" and not out.approval.fast_path and len(out.approval.rounds) == 1


def test_fast_path_never_used_for_flagged_or_unstructured_invoices(tmp_path):
    pipeline = _pipeline(tmp_path, "mix")
    for out in pipeline.process_batch(ALL):
        if not out.approval or not out.approval.fast_path:
            continue
        assert out.extraction.method == "structured"
        assert not out.validation.warnings and not out.validation.critical and out.validation.risk_score == 0
        assert out.status == "PAID"


def test_vp_surprise_on_clean_invoice_still_gets_audited(tmp_path):
    pipeline = _pipeline(tmp_path, "surprise")
    pipeline.approval._propose = lambda *a, **k: Decision(decision="ESCALATE", rationale="Unusual, want a second look.")
    out = pipeline.process(INVOICES / "invoice_1004.json")
    assert not out.approval.fast_path and out.approval.rounds


def test_high_investigator_risk_disables_fast_path(settings):
    pipeline, _ = llm_pipeline(settings)        # scripted investigator reports risk 20
    out = pipeline.process(INVOICES / "invoice_1004.json")
    assert not out.approval.fast_path and out.approval.rounds


# ----------------------------------------------------------- model routing
def test_roles_route_to_the_right_tier():
    assert is_fast_role("ingestion.extract") and is_fast_role("ingestion.self_correct") and is_fast_role("communication.vendor")
    for role in ("validation.investigate", "approval.vp", "approval.vp_revise", "approval.critic", "ingestion.vision"):
        assert not is_fast_role(role)


def test_chat_client_sends_each_role_to_its_model():
    fake = FakeChatClient(lambda kw, i: text_reply('{"decision": "APPROVE", "rationale": "ok"}'))
    llm = ChatLLM(provider="fake", api_key="x", base_url="http://fake", model="default", client=fake,
                  fast_model="quick", reasoning_model="deep")
    llm.structured(role="communication.vendor", system="s", user="u", schema=Decision)
    llm.structured(role="approval.vp", system="s", user="u", schema=Decision)
    assert [r["model"] for r in fake.requests] == ["quick", "deep"]
    assert [c["model"] for c in llm.calls] == ["quick", "deep"]


def test_routing_defaults_to_the_single_model(monkeypatch):
    s = Settings(xai_api_key="k", xai_model="grok-x", xai_model_fast="", xai_model_reasoning="")
    llm = build_llm(s, "grok")
    assert llm.model_for("ingestion.extract") == llm.model_for("approval.vp") == "grok-x"
    llm = build_llm(replace(s, xai_model_fast="grok-fast"), "grok")
    assert llm.model_for("ingestion.extract") == "grok-fast" and llm.model_for("approval.critic") == "grok-x"


# ------------------------------------------------------------------- bench
def test_bench_keeps_every_decision_and_cuts_reasoning_calls(tmp_path):
    bench = run_bench(Settings(db_path=str(tmp_path / "b.db"), log_dir=str(tmp_path / "logs")), scale=0.0001)
    first, last = bench["results"][0], bench["results"][-1]
    for r in bench["results"]:
        assert r["same_decisions"] == r["invoices"] and r["metrics"]["wrongful_payments"] == 0
    assert last["reasoning_calls"] < first["reasoning_calls"]
    assert last["auditor_calls"] < first["auditor_calls"]
    assert last["cost_usd"] < first["cost_usd"]
    text = render_markdown(bench)
    assert "Assumptions (simulated mode)" in text and "\u2014" not in text
