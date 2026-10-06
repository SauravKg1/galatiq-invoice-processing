"""Acme AP Autopilot: operations console.

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ap_autopilot.config import PROJECT_ROOT, Settings
from ap_autopilot.db import Database
from ap_autopilot.graph import NODE_LABELS, InvoicePipeline, flow_dot
from ap_autopilot.llm import LLMError, build_llm
from ap_autopilot.loaders import SUPPORTED_EXTENSIONS, load_document
from ap_autopilot.models import ProcessingOutcome, Severity
from ap_autopilot.reporting import summarize
from ap_autopilot.security import scan_for_injection

st.set_page_config(page_title="AP Autopilot", page_icon="🧾", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&family=IBM+Plex+Mono:wght@400&display=swap');
:root { --paper:#EEF2EF; --ink:#1F2A44; --rule:#C9D3CC; --ok:#2E7D4F; --hold:#A86B12; --bad:#A23B2A; --muted:#5B6578; --card:#FFFFFF; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Public Sans', system-ui, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
h1, h2, h3 { color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }
.amount { font-family: 'Source Serif 4', Georgia, serif; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 2.1rem; line-height: 1.1; }
.sub { color: var(--muted); font-size: 0.92rem; }
.sheet { background: var(--card); border: 1px solid var(--rule); border-radius: 6px; padding: 1.1rem 1.25rem; position: relative; }
.source { font-family: 'IBM Plex Mono', ui-monospace, monospace; white-space: normal; font-size: 0.8rem; line-height: 1.5; color: #2c3550;
          max-height: 560px; overflow: auto; background: #FBFCFB; border: 1px dashed var(--rule); padding: 0.9rem 1rem; border-radius: 4px; }
.stamp { position: absolute; top: 1rem; right: 1.2rem; transform: rotate(-8deg); padding: 0.35rem 0.9rem; border: 3px double currentColor;
         border-radius: 4px; font-weight: 700; font-size: 1.15rem; letter-spacing: 0.08em; opacity: 0.92; }
.stamp.PAID { color: var(--ok); } .stamp.REJECTED { color: var(--bad); } .stamp.ESCALATED, .stamp.FAILED { color: var(--hold); }
.stage { border-left: 3px solid var(--rule); padding: 0.25rem 0 0.6rem 0.9rem; margin: 0.35rem 0 0.6rem 0.2rem; }
.stage h4 { margin: 0 0 0.25rem 0; font-size: 1rem; }
.finding { display: flex; gap: 0.6rem; padding: 0.42rem 0.6rem; margin: 0.25rem 0; background: #F7F9F7; border-radius: 4px; font-size: 0.9rem; }
.finding .sev { flex: 0 0 auto; font-weight: 600; font-size: 0.78rem; padding: 0.05rem 0.45rem; border-radius: 3px; color: white; height: fit-content; }
.sev.critical { background: var(--bad); } .sev.warning { background: var(--hold); } .sev.info { background: #8A94A6; }
mark.inj { background: #F6D5CF; color: #7A1F12; border-bottom: 2px solid var(--bad); padding: 0 1px; }
mark.soc { background: #F5E3C2; color: #6B4508; border-bottom: 2px solid var(--hold); padding: 0 1px; }
.finding code { background: transparent; color: var(--ink); font-weight: 600; padding: 0; }
.rationale { font-style: italic; color: #2c3550; }
.guard { color: #6B3FA0; font-size: 0.88rem; }
.step { border-left: 3px solid #2F6FA3; padding: 0.15rem 0 0.35rem 0.8rem; margin: 0 0 0.45rem 0.1rem; font-size: 0.88rem; }
.step.latest { border-left-color: #C4702B; }
.step b { color: var(--ink); } .step .ms { color: var(--muted); font-size: 0.78rem; margin-left: 0.4rem; }
.step ul { margin: 0.15rem 0 0 0; padding-left: 1.1rem; } .step li { margin: 0; color: #2c3550; }
@media (max-width: 640px) { .stamp { position: static; display: inline-block; margin-bottom: 0.6rem; } .amount { font-size: 1.6rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SAMPLE_DIRS = [PROJECT_ROOT / "data" / "invoices", PROJECT_ROOT / "data" / "invoices_extra"]
SUPPORTED = SUPPORTED_EXTENSIONS
STATUS_LABEL = {"PAID": "PAID", "REJECTED": "REJECTED", "ESCALATED": "HELD FOR REVIEW", "FAILED": "NEEDS MANUAL HANDLING"}
e = html.escape


def sample_files() -> list[Path]:
    return [f for d in SAMPLE_DIRS for f in sorted(d.iterdir()) if f.suffix.lower() in SUPPORTED]


def money(x, currency="USD") -> str:
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}" if currency in (None, "USD") else f"{sign}{currency} {abs(x):,.2f}"


# ------------------------------------------------------------------ setup
settings = Settings()
with st.sidebar:
    st.markdown("### AP Autopilot")
    st.markdown('<div class="sub">Acme Corp accounts payable</div>', unsafe_allow_html=True)
    view = st.radio("View", ["Process an invoice", "Run the inbox", "Review queue", "Vendor insights", "Business case", "Audit trail"],
                    label_visibility="collapsed")
    st.divider()
    detected = settings.resolved_provider()
    options = ["offline"] + [p for p in ("grok", "openai") if (p == "grok" and settings.xai_api_key) or (p == "openai" and settings.openai_api_key)]
    provider = st.selectbox("Reasoning engine", options, index=options.index(detected) if detected in options else 0,
                            help="Grok needs XAI_API_KEY in .env. Offline runs the same pipeline with deterministic rules.")
    if provider == "offline" and not settings.xai_api_key:
        st.caption("Add XAI_API_KEY to .env to run the agents on Grok.")
    if st.button("Reset demo data", help="Recreate inventory.db: clears payments, queue and history"):
        Database(settings.db_path).init(reset=True)
        st.cache_resource.clear()
        st.session_state.pop("outcome", None)
        st.session_state.pop("batch", None)
        st.session_state.pop("trace", None)
        st.toast("Demo data reset")


@st.cache_resource
def get_pipeline(provider_name: str) -> InvoicePipeline:
    return InvoicePipeline(settings, llm=build_llm(settings, provider_name))


try:
    pipeline = get_pipeline(provider)
except LLMError as exc:
    st.error(str(exc))
    st.stop()


# --------------------------------------------------------------- render
@st.cache_data(show_spinner=False)
def scan_pages(path: str) -> list[bytes]:
    return load_document(path).images


def source_html(raw: str) -> str:
    """Render document text faithfully, marking any prompt-injection spans."""
    if not raw:
        return "No text could be read."

    def fmt(chunk: str) -> str:
        return e(chunk).replace(" ", "&nbsp;").replace("\n", "<br>")

    out, pos = [], 0
    for h in sorted(scan_for_injection(raw), key=lambda h: h.start):
        # extend the mark to the end of the sentence so the whole instruction is visible
        end = h.end
        stop = min([i for i in (raw.find(".", h.end), raw.find("\n", h.end)) if i != -1] or [len(raw)])
        end = max(end, stop + 1 if stop < len(raw) and raw[stop] == "." else stop)
        if h.start < pos:
            continue
        out.append(fmt(raw[pos:h.start]))
        css = "inj" if h.tier == "blatant" else "soc"
        out.append(f'<mark class="{css}" title="{e(h.reason)}">{fmt(raw[h.start:end])}</mark>')
        pos = end
    out.append(fmt(raw[pos:]))
    return "".join(out)


def render_findings(outcome: ProcessingOutcome, show_info: bool) -> str:
    rows = []
    for f in outcome.validation.findings:
        if f.severity == Severity.INFO and not show_info:
            continue
        origin = " (raised by the investigator)" if f.source == "llm" else ""
        rows.append(f'<div class="finding"><span class="sev {f.severity.value}">{f.severity.value}</span>'
                    f'<span><code>{e(f.code)}</code>{origin}: {e(f.message)}</span></div>')
    return "".join(rows) or '<div class="sub">No issues found.</div>'


AUDIENCE_LABEL = {"vendor": "To the vendor", "internal": "Internal"}


def render_po_match(m: dict) -> None:
    """Invoice vs purchase order vs goods receipt, line by line (SAP invoice-verification style)."""
    how = ", matched by items (no PO number on the invoice)" if m.get("inferred") else ""
    st.markdown(f"**Three-way match** against {e(m['po_number'])} ({e(m.get('po_vendor') or '')}, {e(m.get('po_status') or '')}{how})")
    def num(x, money=False):
        if x is None:
            return "n/a"
        return f"${x:,.2f}" if money else f"{x:g}"
    rows = [{"Item": ln["item"],
             "Result": "Match" if ln["status"] == "match" else ln["status"].capitalize(),
             "Billed": num(ln["invoiced_qty"]), "Ordered": num(ln["po_qty"]), "Received": num(ln["received"]),
             "Billed before": num(ln["already_invoiced"]),
             "Price vs PO": f"{num(ln['invoiced_price'], True)} vs {num(ln['po_price'], True)}"} for ln in m["lines"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_drafts(outcome: ProcessingOutcome) -> None:
    if not outcome.drafts:
        return
    st.markdown('<div class="stage"><h4>5. Tell the people involved</h4><div class="sub">Drafts only. '
                'Nothing is sent until a person sends it.</div></div>', unsafe_allow_html=True)
    for d in outcome.drafts:
        row = pipeline.db.get_draft(d.id) if d.id else None
        status = row["status"] if row else "draft"
        author = ", written by Grok" if d.author == "llm" else ""
        with st.expander(f"{AUDIENCE_LABEL[d.audience]}: {d.subject}  ({status})", expanded=False):
            st.caption(f"To {d.to}{author}")
            if d.note:
                st.caption(d.note)
            if status != "draft" or not row:
                st.text(row["body"] if row else d.body)
                if row and row["sent_by"]:
                    st.caption(f"{status.capitalize()} by {row['sent_by']} at {row['sent_at']}")
                continue
            subject = st.text_input("Subject", value=row["subject"], key=f"subj-{d.id}")
            body = st.text_area("Message", value=row["body"], height=220, key=f"body-{d.id}")
            send, discard, _ = st.columns([1, 1, 4])
            if send.button("Send", key=f"send-{d.id}", type="primary"):
                pipeline.db.update_draft(d.id, subject, body)
                pipeline.db.set_draft_status(d.id, "sent", "console-reviewer")
                pipeline.db.audit(outcome.run_id, outcome.invoice_key, "communicate", "sent", {"draft_id": d.id, "to": d.to})
                st.toast(f"Sent to {d.to} (mock: no real email leaves this machine)")
                st.rerun()
            if discard.button("Discard", key=f"discard-{d.id}"):
                pipeline.db.set_draft_status(d.id, "discarded", "console-reviewer")
                st.toast("Draft discarded")
                st.rerun()


def render_debate(a) -> None:
    """The approval loop as a conversation: the VP agent proposes, the auditor challenges."""
    for r in a.rounds:
        with st.chat_message("VP", avatar=":material/person:"):
            st.markdown(f"**VP agent, round {r.round}: {r.decision.decision}**")
            st.markdown(r.decision.rationale.replace("$", "\\$"))
            if r.decision.cited_findings:
                st.caption("Cites: " + ", ".join(r.decision.cited_findings))
            if r.decision.conditions:
                st.caption("Conditions: " + "; ".join(r.decision.conditions))
        with st.chat_message("Auditor", avatar=":material/fact_check:"):
            if r.critique.agrees:
                st.markdown(f"**Auditor:** agreed with {r.decision.decision}.")
            else:
                st.markdown(f"**Auditor:** disagreed, recommends {r.critique.recommended_decision}.")
                for issue in r.critique.issues:
                    st.markdown(f"- {issue}".replace("$", "\\$"))
    for o in a.overrides:
        with st.chat_message("Guardrail", avatar=":material/shield:"):
            st.markdown(f"**Policy guardrail:** {o}".replace("$", "\\$"))


def render_trace(events: list[dict], flow_slot, steps_slot, live: bool = False) -> None:
    """Draw the agent graph with the path taken, and the step-by-step timeline."""
    visited = [ev["node"] for ev in events]
    latest = visited[-1] if live and visited else None
    flow_slot.graphviz_chart(flow_dot(visited, current=latest), width="stretch")
    rows = []
    for ev in events:
        cls = "step latest" if ev["node"] == latest else "step"
        items = "".join(f"<li>{e(line)}</li>" for line in ev["summary"])
        rows.append(f'<div class="{cls}"><b>{e(NODE_LABELS[ev["node"]])}</b><span class="ms">{ev["ms"]:.0f} ms</span><ul>{items}</ul></div>')
    steps_slot.markdown("".join(rows) or '<div class="sub">Starting.</div>', unsafe_allow_html=True)


def run_live(target: Path, pace: float) -> None:
    """Process one invoice, redrawing the graph and timeline as each agent finishes."""
    st.markdown("#### Agent trace")
    left, right = st.columns([7, 5], gap="large")
    flow_slot, steps_slot = left.empty(), right.empty()
    note = left.empty()
    note.caption("Blue: steps taken. Orange: the step that just finished. Dashed: branches not taken.")
    events: list[dict] = []
    render_trace(events, flow_slot, steps_slot, live=True)
    for ev in pipeline.process_stream(target):
        if ev["type"] == "node":
            events.append({k: ev[k] for k in ("node", "ms", "summary")})
            render_trace(events, flow_slot, steps_slot, live=True)
            if pace:
                time.sleep(pace)
        elif ev["type"] == "outcome":
            st.session_state["outcome"] = ev["outcome"]
    st.session_state["trace"] = events
    render_trace(events, flow_slot, steps_slot)
    note.caption("Blue: steps taken. Dashed: branches not taken.")


def render_outcome(outcome: ProcessingOutcome) -> None:
    inv = outcome.extraction.invoice if outcome.extraction else None
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("#### Source document")
        raw = outcome.extraction.raw_text if outcome.extraction else ""
        x = outcome.extraction
        if x and x.method in {"vision", "ocr"} and x.source_path and Path(x.source_path).exists():
            pages = scan_pages(x.source_path)
            for page in pages:
                st.image(page, width="stretch")
            reader = "Grok vision transcription" if x.method == "vision" else "local OCR text"
            st.caption(f"Scanned document: no text layer. Below is the {reader} the agents worked from.")
        st.markdown(f'<div class="source">{source_html(raw)}</div>', unsafe_allow_html=True)
        if any(h.tier == "blatant" for h in scan_for_injection(raw)):
            st.caption("Highlighted: instructions aimed at the AP system. In a PDF this text may be invisible to a person.")
        st.caption(f"{outcome.source_file}")
    with right:
        currency = (inv.currency if inv else None) or "USD"
        st.markdown(
            f'<div class="sheet"><div class="stamp {outcome.status}">{STATUS_LABEL[outcome.status]}</div>'
            f'<div class="sub">{e(inv.invoice_number or "No invoice number") if inv else ""}</div>'
            f'<h3 style="margin:0.15rem 0 0.3rem 0">{e(inv.vendor_name or "Unknown vendor") if inv else "Unreadable"}</h3>'
            f'<div class="amount">{money(inv.total if inv else None, currency)}</div>'
            f'<div class="sub">Due {inv.due_date or e(inv.due_date_raw or "date missing") if inv else "n/a"} '
            f'&nbsp;|&nbsp; {outcome.total_ms:.0f} ms &nbsp;|&nbsp; {e(outcome.llm_mode)}</div></div>',
            unsafe_allow_html=True)
        if outcome.error:
            st.error(outcome.error)

        if outcome.extraction:
            x = outcome.extraction
            st.markdown(f'<div class="stage"><h4>1. Read the invoice</h4><div class="sub">{x.source_format.upper()} via {x.method}, '
                        f'{len(inv.line_items)} line item{'s' if len(inv.line_items) != 1 else ''}, confidence {x.confidence:.0%}, {x.attempts} attempt(s)</div></div>',
                        unsafe_allow_html=True)
            table = pd.DataFrame([{"Item as written": li.description, "Matched SKU": outcome.validation.item_matches.get(li.description) if outcome.validation else None,
                                   "Qty": li.quantity, "Unit price": li.unit_price, "Note": li.note or ""} for li in inv.line_items]).fillna("")
            st.dataframe(table, hide_index=True, width="stretch")
            if x.corrections:
                with st.expander(f"Self-corrections ({len(x.corrections)})"):
                    for c in x.corrections:
                        st.write(f"- {c}")
        if outcome.validation:
            v = outcome.validation
            st.markdown(f'<div class="stage"><h4>2. Check it against our records</h4><div class="sub">{len(v.critical)} critical, '
                        f'{len(v.warnings)} warning{"s" if len(v.warnings) != 1 else ""}, fraud risk {v.risk_score}/100'
                        f'{f", investigator {v.llm_risk}/100" if v.llm_risk is not None else ""}</div></div>', unsafe_allow_html=True)
            show_info = st.toggle("Show informational notes", value=False, key=f"info-{outcome.run_id}")
            st.markdown(render_findings(outcome, show_info), unsafe_allow_html=True)
            if v.llm_summary:
                st.caption(f"Investigator: {v.llm_summary}")
            if v.tool_calls:
                with st.expander(f"Investigator tool calls ({len(v.tool_calls)})"):
                    st.caption("Read-only database lookups the investigator chose to make, with what each returned.")
                    for i, c in enumerate(v.tool_calls, 1):
                        args = ", ".join(f"{k}={val!r}" for k, val in c["args"].items())
                        st.markdown(f"**{i}. `{c['tool']}({e(args)})`**")
                        result = c.get("result")
                        if result:
                            try:
                                st.code(json.dumps(json.loads(result), indent=2), language="json")
                            except ValueError:
                                st.code(result)
            if v.po_match and v.po_match.get("found"):
                render_po_match(v.po_match)
            if v.history_context and inv.total:
                h = v.history_context
                st.caption(f"Against this vendor's history ({h['n']} invoices, synthetic): typical invoice "
                           f"\\${h['median']:,.0f}, normal range up to \\${h['normal_high']:,.0f}; this one is {h['ratio']:.1f}x typical. "
                           "See Vendor insights.")
        if outcome.approval:
            a = outcome.approval
            who = {"policy": "policy rule", "agent": "VP agent + auditor", "human": "human reviewer"}[a.decided_by]
            if a.fast_path:
                who = "VP agent (fast path: clean structured invoice, auditor round skipped)"
            st.markdown(f'<div class="stage"><h4>3. Decide</h4><div class="sub">{a.final.decision} by {who}, '
                        f'{"over" if a.policy_tier == "vp_scrutiny" else "under"} the ${settings.approval_threshold:,.0f} limit</div>'
                        f'<p class="rationale">{e(a.final.rationale)}</p>'
                        + "".join(f'<div class="guard">Guardrail: {e(o)}</div>' for o in a.overrides) + "</div>",
                        unsafe_allow_html=True)
            if a.rounds:
                with st.expander(f"VP and auditor discussion ({len(a.rounds)} round{'s' if len(a.rounds) != 1 else ''})",
                                 expanded=len(a.rounds) > 1 or bool(a.overrides)):
                    render_debate(a)
        if outcome.payment:
            p = outcome.payment
            st.markdown(f'<div class="stage"><h4>4. Pay, reject or hold</h4><div class="sub">{e(p.status)}: {e(p.message)}'
                        f'{" (" + e(p.reference) + ")" if p.reference else ""}</div></div>', unsafe_allow_html=True)
        render_drafts(outcome)


# ----------------------------------------------------------------- views
if view == "Process an invoice":
    st.markdown("## Process an invoice")
    st.markdown('<div class="sub">Pick a sample or upload a PDF (text or scanned), photo, text, CSV, JSON, XML or email file. '
                'Processing order matters: the system remembers what it has already paid.</div>', unsafe_allow_html=True)
    files = sample_files()
    c1, c2 = st.columns([3, 2])
    with c1:
        choice = st.selectbox("Sample invoice", files, format_func=lambda p: f"{p.parent.name}/{p.name}")
    with c2:
        upload = st.file_uploader("Or upload your own", type=[s.lstrip(".") for s in SUPPORTED])
    b1, b2 = st.columns([1, 4], vertical_alignment="center")
    go = b1.button("Process invoice", type="primary")
    pace = b2.toggle("Demo pace", value=True, help="Pause briefly after each agent so the flow is easy to follow. "
                     "Display only: processing speed is unchanged.")
    if go:
        target = choice
        if upload is not None:
            tmp = Path(tempfile.mkdtemp()) / upload.name
            tmp.write_bytes(upload.getvalue())
            target = tmp
        run_live(target, 0.6 if pace else 0.0)
    elif st.session_state.get("trace") and "outcome" in st.session_state:
        st.markdown("#### Agent trace")
        left, right = st.columns([7, 5], gap="large")
        render_trace(st.session_state["trace"], left.empty(), right.empty())
        left.caption("Blue: steps taken. Dashed: branches not taken.")
    if "outcome" in st.session_state:
        st.divider()
        render_outcome(st.session_state["outcome"])

elif view == "Run the inbox":
    st.markdown("## Run the inbox")
    st.markdown('<div class="sub">Processes every sample invoice in arrival order, the way a month of AP mail would land.</div>',
                unsafe_allow_html=True)
    fresh = st.checkbox("Start from empty payment history", value=True)
    if st.button("Run the inbox", type="primary"):
        if fresh:
            Database(settings.db_path).init(reset=True)
            st.cache_resource.clear()
            pipeline = get_pipeline(provider)
        bar = st.progress(0.0)
        outcomes = []
        files = sample_files()
        # files are read in parallel ahead; decisions are made one at a time in arrival order
        for i, outcome in enumerate(pipeline.iter_batch(files), 1):
            outcomes.append(outcome)
            bar.progress(i / len(files), text=outcome.source_file)
        bar.empty()
        st.session_state["batch"] = outcomes
    if "batch" in st.session_state:
        outcomes = st.session_state["batch"]
        s = summarize(outcomes)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Paid automatically", f"{s['paid']}")
        m1.caption(money(s["amount_paid"]) + " sent")
        m2.metric("Rejected", f"{s['rejected']}")
        m2.caption(money(s["amount_blocked"]) + " blocked")
        m3.metric("Held for a person", f"{s['escalated']}")
        m3.caption(money(s["amount_held_for_review"]) + " awaiting review")
        m4.metric("Decided without a person", f"{s['touchless_rate']:.0%}")
        m4.caption(f"{s['avg_ms']:.0f} ms per invoice")
        rows = []
        for o in outcomes:
            inv = o.extraction.invoice if o.extraction else None
            v = o.validation
            reasons = (v.critical or v.warnings) if v else []
            rows.append({"File": o.source_file, "Vendor": (inv.vendor_name if inv else None) or "(blank)",
                         "Amount": money(inv.total, inv.currency) if inv else "n/a", "Outcome": STATUS_LABEL[o.status],
                         "Risk": v.risk_score if v else None, "Main reason": reasons[0].message if reasons else "All checks passed"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Main reason": st.column_config.TextColumn(width="large")})
        pick = st.selectbox("Open an invoice", range(len(outcomes)), format_func=lambda i: outcomes[i].source_file)
        render_outcome(outcomes[pick])

elif view == "Vendor insights":
    from app_pages.vendor_insights import render as render_vendor_insights

    batch = st.session_state.get("batch") or []
    single = [st.session_state["outcome"]] if "outcome" in st.session_state else []
    render_vendor_insights(pipeline.db, batch + single)

elif view == "Business case":
    from app_pages.business_case import render as render_business_case

    render_business_case(st.session_state.get("batch"))

elif view == "Review queue":
    st.markdown("## Review queue")
    queue = pipeline.db.review_queue("pending")
    if not queue:
        st.markdown('<div class="sub">Nothing is waiting for review. Escalated invoices appear here.</div>', unsafe_allow_html=True)
    for item in queue:
        with st.container(border=True):
            st.markdown(f"**{e(item['vendor'] or 'Unknown vendor')}** &nbsp; {money(item['amount'], item['currency'])} "
                        f"&nbsp; <span class='sub'>{e(item['source_file'] or '')}</span>", unsafe_allow_html=True)
            st.markdown(f"<p class='rationale'>{e(item['reason'] or '')}</p>", unsafe_allow_html=True)
            note = st.text_input("Reason for your decision", key=f"note-{item['id']}",
                                 placeholder="e.g. Confirmed bank details by phone with the number on file")
            a, r, _ = st.columns([1, 1, 4])
            if a.button("Approve and pay", key=f"a-{item['id']}", type="primary", disabled=not note):
                result = pipeline.human_decision(item["id"], True, "console-reviewer", note)
                st.toast(f"{result.status}: {result.message}")
                st.rerun()
            if r.button("Reject", key=f"r-{item['id']}", disabled=not note):
                pipeline.human_decision(item["id"], False, "console-reviewer", note)
                st.toast("Rejected")
                st.rerun()
    if queue:
        st.caption("A reason is required for every decision; it is written to the audit trail.")
    drafts = pipeline.db.outbox("draft")
    st.markdown(f"#### Messages waiting to be sent ({len(drafts)})")
    if drafts:
        st.dataframe(pd.DataFrame([{"To": d["to_addr"], "Type": d["kind"].replace("_", " "), "Subject": d["subject"],
                                    "Invoice": d["source_file"], "Written by": "Grok" if d["author"] == "llm" else "template"}
                                   for d in drafts]), hide_index=True, width="stretch")
        st.caption("Open the invoice to edit and send, or use `python main.py --outbox`. Nothing is sent automatically.")
    else:
        st.markdown('<div class="sub">No drafts waiting.</div>', unsafe_allow_html=True)

else:
    st.markdown("## Audit trail")
    st.markdown('<div class="sub">Every stage of every run, newest first. The same events are written to logs/ as JSON lines.</div>',
                unsafe_allow_html=True)
    rows = pipeline.db.audit_trail(limit=500)
    if rows:
        st.dataframe(pd.DataFrame(rows)[["ts", "run_id", "invoice_key", "stage", "event", "payload"]],
                     hide_index=True, width="stretch")
    payments = pipeline.db.payments()
    st.markdown("#### Payments sent")
    if payments:
        st.dataframe(pd.DataFrame(payments), hide_index=True, width="stretch")
    else:
        st.markdown('<div class="sub">No payments yet.</div>', unsafe_allow_html=True)
