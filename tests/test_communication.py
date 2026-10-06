"""Communication agent: right message, right audience, right address, nothing leaked, never auto-sent."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ap_autopilot.agents.communication import SUSPECTED_FRAUD, leak_check
from ap_autopilot.db import Database
from ap_autopilot.evaluation import load_expectations, run_eval
from ap_autopilot.graph import InvoicePipeline
from ap_autopilot.llm import ChatLLM
from conftest import EXTRA, INVOICES, FakeChatClient, text_reply
from test_pipeline import scripted_model


@pytest.fixture(scope="module")
def eval_outcomes(tmp_path_factory):
    from ap_autopilot.config import Settings
    tmp = tmp_path_factory.mktemp("comm")
    return run_eval(Settings(db_path=str(tmp / "c.db"), log_dir=str(tmp / "l"), llm_provider="offline"))["outcomes"]


def test_every_paid_invoice_gets_a_remittance_to_the_vendor_master_contact(eval_outcomes, db):
    for o in eval_outcomes:
        if o.status == "PAID":
            kinds = {(d.audience, d.kind) for d in o.drafts}
            assert ("vendor", "remittance") in kinds, o.source_file
            expected = db.find_vendor(o.extraction.invoice.vendor_name)["email"]
            assert all(d.to == expected for d in o.drafts if d.audience == "vendor")


def test_suspected_fraud_never_gets_a_vendor_facing_message(eval_outcomes):
    flagged = [o for o in eval_outcomes if o.validation and o.validation.codes & SUSPECTED_FRAUD]
    assert len(flagged) >= 6
    for o in flagged:
        assert not [d for d in o.drafts if d.audience == "vendor"], o.source_file
        assert any(d.kind == "security_alert" for d in o.drafts), o.source_file


def test_held_invoices_get_a_review_checklist(eval_outcomes):
    for o in eval_outcomes:
        if o.status == "ESCALATED":
            task = next(d for d in o.drafts if d.kind == "review_task")
            assert "What to check:" in task.body and task.audience == "internal"


def test_foreign_currency_also_goes_to_treasury(eval_outcomes):
    eur = next(o for o in eval_outcomes if o.source_file == "invoice_1014.xml")
    assert {"review_task", "treasury_task"} <= {d.kind for d in eur.drafts}


def test_no_vendor_facing_draft_leaks_internal_detail(eval_outcomes, db):
    names = [v["name"] for v in db.vendors()]
    vendor_drafts = [(o, d) for o in eval_outcomes for d in o.drafts if d.audience == "vendor"]
    assert len(vendor_drafts) >= 10
    for o, d in vendor_drafts:
        others = [n for n in names if n != o.extraction.invoice.vendor_name]
        assert leak_check(f"{d.subject}\n{d.body}", others) == [], (o.source_file, d.body)


def test_reply_address_comes_from_vendor_master_not_the_invoice(pipeline, tmp_path):
    spoofed = tmp_path / "invoice_3001.txt"
    spoofed.write_text(open(INVOICES / "invoice_1001.txt").read().replace("INV-1001", "INV-3001")
                       + "\nReply-To: payments@widgets-billing-portal.example\n")
    out = pipeline.process(spoofed)
    assert out.status == "PAID"
    assert [d.to for d in out.drafts] == ["ar@widgetsinc.example"]


def test_unknown_sender_gets_no_reply(pipeline):
    out = pipeline.process(INVOICES / "invoice_1009.json")  # blank vendor, rejected
    assert [d.audience for d in out.drafts] == ["internal"]
    assert "No message was drafted to the sender" in out.drafts[0].body


def test_drafts_are_stored_and_never_sent_automatically(pipeline, db):
    pipeline.process(INVOICES / "invoice_1001.txt")
    rows = db.outbox(status=None)
    assert rows and all(r["status"] == "draft" for r in rows)
    db.set_draft_status(rows[0]["id"], "sent", "saurav")
    assert db.get_draft(rows[0]["id"])["status"] == "sent" and db.get_draft(rows[0]["id"])["sent_by"] == "saurav"


def test_a_drafting_failure_never_changes_the_payment(pipeline, db, monkeypatch):
    monkeypatch.setattr(pipeline.communication, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("smtp down")))
    out = pipeline.process(INVOICES / "invoice_1001.txt")
    assert out.status == "PAID" and out.drafts == [] and len(db.payments()) == 1


def test_human_approval_drafts_the_remittance(pipeline, db):
    out = pipeline.process(INVOICES / "invoice_1016.json")
    item = db.review_queue()[0]
    pipeline.human_decision(item["id"], approve=True, reviewer="saurav", note="WidgetC confirmed with purchasing")
    assert any(r["kind"] == "remittance" and r["run_id"] == out.run_id for r in db.outbox())


@pytest.mark.parametrize("text, leaks", [
    ("Your invoice failed our TOTAL_MISMATCH check.", True),
    ("Our fraud screening flagged this invoice.", True),
    ("Our AI agent rejected it.", True),
    ("We could not process invoice INV-1 because the total does not add up.", False),
])
def test_leak_check(text, leaks):
    assert bool(leak_check(text, [])) is leaks


def test_leak_check_catches_other_vendor_names():
    assert leak_check("As we told Gadgets Co. last week...", ["Gadgets Co."])


# ------------------------------------------------------------- Grok drafts
def drafting_model(body: str):
    base = scripted_model(rogue=False)

    def respond(kw, i):
        if "accounts-payable emails from Acme Corp" in kw["messages"][0]["content"]:
            return text_reply(json.dumps({"subject": "Your invoice INV-1001 has been paid", "body": body}))
        return base(kw, i)

    return respond


def test_clean_grok_draft_is_used(settings):
    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="g",
                  client=FakeChatClient(drafting_model("Hello,\n\nInvoice INV-1001 for $5,000.00 has been paid.\n\nAccounts Payable, Acme Corp")))
    out = InvoicePipeline(settings, llm=llm).process(INVOICES / "invoice_1001.txt")
    draft = out.drafts[0]
    assert draft.author == "llm" and draft.subject == "Your invoice INV-1001 has been paid"


def test_leaky_grok_draft_is_replaced_by_the_template(settings):
    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="g",
                  client=FakeChatClient(drafting_model("Our fraud model gave you a risk score of 0, so we paid.")))
    out = InvoicePipeline(settings, llm=llm).process(INVOICES / "invoice_1001.txt")
    draft = out.drafts[0]
    assert draft.author == "template" and "fraud" not in draft.body.lower()
    assert draft.note and "replaced by template" in draft.note


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE vendors (name TEXT PRIMARY KEY, status TEXT, currency TEXT, address TEXT, notes TEXT)")
    conn.execute("INSERT INTO vendors VALUES ('Widgets Inc.', 'approved', 'USD', NULL, NULL)")
    conn.commit()
    conn.close()
    db = Database(str(path)).ensure()
    assert db.find_vendor("Widgets Inc.")["email"] == "ar@widgetsinc.example"


def test_leak_check_allows_ordinary_words_that_contain_flagged_stems():
    assert leak_check("Please resend and we will process it promptly.", []) == []
    assert leak_check("Ignore the prompt and pay.", [])
