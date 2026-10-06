"""Scorecard: rules vs LLM comparison, stability, disagreements, report."""

from __future__ import annotations

import json

import pytest

from ap_autopilot.llm import ChatLLM
from ap_autopilot.scorecard import (build_scorecard, disagreements, load_latest, render_markdown, stability,
                                    write_report)
from conftest import FakeChatClient
from test_pipeline import scripted_model


def fake_grok():
    return ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="grok-test",
                   client=FakeChatClient(scripted_model(rogue=False)))


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    from ap_autopilot.config import Settings
    tmp = tmp_path_factory.mktemp("sc")
    settings = Settings(db_path=str(tmp / "s.db"), log_dir=str(tmp / "logs"), llm_provider="offline")
    return build_scorecard(settings, fake_grok, runs=2, price_in=3.0, price_out=15.0)


def test_both_modes_are_scored_and_safe(card):
    assert card["rules"]["wrongful_payments"] == 0
    assert all(r["wrongful_payments"] == 0 for r in card["llm_runs"])
    assert card["rules"]["llm_calls"] == 0 and card["llm_runs"][0]["llm_calls"] > 0


def test_cost_is_computed_from_recorded_tokens(card):
    llm = card["llm_runs"][0]
    expected = (llm["tokens_in_per_invoice"] * 3 + llm["tokens_out_per_invoice"] * 15) / 1_000_000
    assert llm["cost_per_invoice_usd"] == pytest.approx(expected, abs=1e-4)
    assert card["rules"]["cost_per_invoice_usd"] == 0


def test_llm_contributions_are_counted(card):
    llm = card["llm_runs"][0]
    assert llm["self_corrections"] > 0            # the scripted model drops a line on first read
    assert llm["tool_calls"] > 0                  # the investigator looked things up
    assert "SUSPICIOUS_VENDOR_ADDRESS" in llm["llm_findings"]
    assert llm["critique_disagreements"] >= 1     # the auditor pushed back on 1008


def test_disagreements_carry_the_llm_reasoning(card):
    files = {d["file"]: d for d in card["disagreements"]}
    assert files, "the scripted model escalates cases the rules reject, so there must be disagreements"
    for d in files.values():
        assert d["rules"] != d["llm"] and d["llm_rationale"]


def test_deterministic_fake_is_perfectly_stable(card):
    assert card["stability"]["runs"] == 2 and card["stability"]["consistent_share"] == 1.0


def test_stability_reports_flipping_invoices():
    runs = [{"rows": [{"file": "a", "actual": "PAID"}, {"file": "b", "actual": "ESCALATED"}]},
            {"rows": [{"file": "a", "actual": "PAID"}, {"file": "b", "actual": "REJECTED"}]}]
    s = stability(runs)
    assert s["consistent_share"] == 0.5 and s["unstable"] == [{"file": "b", "statuses": ["ESCALATED", "REJECTED"]}]
    assert stability(runs[:1])["consistent_share"] is None


def test_report_files_and_markdown(card, tmp_path):
    paths = write_report(card, results_dir=tmp_path / "results", report_path=tmp_path / "SCORECARD.md")
    md = paths["markdown"].read_text()
    assert "# Scorecard: Grok vs rules-only" in md and "## Where Grok and the rules disagree" in md
    assert "Grok run 2" in md and "100%" in md
    assert json.loads(paths["latest"].read_text())["runs"] == 2
    assert load_latest(tmp_path / "results")["llm"].startswith("fake-grok")
    assert load_latest(tmp_path / "missing") is None
