"""Live trace: the streamed run must match the plain run, step for step."""

from __future__ import annotations

import pytest

from ap_autopilot.config import Settings
from ap_autopilot.graph import InvoicePipeline, describe_step, flow_dot
from ap_autopilot.llm import OfflineLLM
from conftest import INVOICES

CASES = [
    ("invoice_1001.txt", "PAID", ["ingest", "validate", "approve", "pay", "communicate"]),
    ("invoice_1008.txt", "REJECTED", ["ingest", "validate", "approve", "reject", "communicate"]),
]


def _pipeline(tmp_path, name="t"):
    p = InvoicePipeline(Settings(db_path=str(tmp_path / f"{name}.db"), log_dir=str(tmp_path / "logs")), llm=OfflineLLM())
    p.db.init(reset=True)
    return p


def _nodes(pipeline, path):
    events = list(pipeline.process_stream(path))
    return events, [ev["node"] for ev in events if ev["type"] == "node"]


@pytest.mark.parametrize("name,status,path", CASES)
def test_stream_visits_expected_nodes(tmp_path, name, status, path):
    events, nodes = _nodes(_pipeline(tmp_path), INVOICES / name)
    assert nodes == path
    assert events[0]["type"] == "start" and events[-1]["type"] == "outcome"
    assert events[-1]["outcome"].status == status
    assert all(ev["summary"] for ev in events if ev["type"] == "node")


def test_held_invoice_goes_through_escalate(tmp_path):
    pipeline = _pipeline(tmp_path)
    for name in sorted(p.name for p in INVOICES.iterdir()):
        events, nodes = _nodes(pipeline, INVOICES / name)
        if events[-1]["outcome"].status == "ESCALATED":
            assert "escalate" in nodes and "pay" not in nodes
            return
    pytest.fail("no sample invoice was held for review")


def test_unreadable_file_takes_fail_path(tmp_path):
    bad = tmp_path / "empty.txt"
    bad.write_text("")
    events, nodes = _nodes(_pipeline(tmp_path), bad)
    assert "fail" in nodes and nodes[-1] == "communicate"
    assert events[-1]["outcome"].status == "FAILED"


@pytest.mark.parametrize("name,status,path", CASES)
def test_stream_outcome_matches_process(tmp_path, name, status, path):
    streamed = [ev for ev in _pipeline(tmp_path, "a").process_stream(INVOICES / name)][-1]["outcome"]
    plain = _pipeline(tmp_path, "b").process(INVOICES / name)
    for field in ("status", "invoice_key", "extraction", "validation"):
        assert getattr(streamed, field) == getattr(plain, field), field
    # the payment reference embeds the run id, so compare everything else
    assert streamed.payment.model_dump(exclude={"reference", "message"}) == plain.payment.model_dump(exclude={"reference", "message"})
    assert streamed.approval.final == plain.approval.final
    assert set(streamed.timings_ms) == set(plain.timings_ms)


def test_flow_dot_highlights_path_and_fades_untaken_branches():
    dot = flow_dot(["ingest", "validate", "approve", "reject"], current="reject")
    assert 'reject [label="Reject", style="filled,bold", fillcolor="#C4702B"' in dot
    assert 'ingest [label="Read", style=filled, fillcolor="#2F6FA3"' in dot
    assert 'pay [label="Pay", style="filled,dashed"' in dot
    assert "approve -> reject [color=\"#2F6FA3\"" in dot
    assert "approve -> pay [color=\"#9AA3B2\", style=dashed]" in dot
    assert "fail" not in dot   # the fail-safe node only appears when it was used


def test_flow_dot_shows_fail_node_when_used():
    dot = flow_dot(["ingest", "fail", "communicate"])
    assert "ingest -> fail [color=\"#2F6FA3\"" in dot


def test_describe_step_reports_errors():
    assert describe_step("ingest", {"error": "could not parse"}) == ["Error: could not parse"]


def test_mermaid_export_lists_every_node():
    import importlib.util
    from conftest import ROOT
    spec = importlib.util.spec_from_file_location("export_graph", ROOT / "scripts" / "export_graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.mermaid()
    for node in ("ingest", "validate", "approve", "pay", "reject", "escalate", "fail", "communicate"):
        assert node in text
