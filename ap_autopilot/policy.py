"""Approval policy: the guardrails around the approval agent.

The LLM proposes, policy disposes. These rules run AFTER the VP agent and
critic have agreed, and they only ever move a decision toward caution:
an agent cannot approve what policy forbids, and it cannot reject a clean
invoice without evidence (wrongful rejections cost vendor relationships and
late fees too). Anything in between goes to a human.
"""

from __future__ import annotations

from .config import Settings
from .models import Decision, Invoice, ValidationReport


# Warnings an agent may judge benign on its own. Every other warning (unknown
# vendor or item, identity/bank changes, duplicates, math errors, stock, FX,
# fraud signals, anything the LLM itself raised) needs a human. Autonomy is
# earned: widen this list only when eval data shows the agent judges it well.
AGENT_WAIVABLE = {"PRICE_ABOVE_CATALOG", "MISSING_DUE_DATE", "ZERO_QUANTITY", "DUE_BEFORE_INVOICE_DATE",
                  "TERMS_DATE_MISMATCH", "MISSING_INVOICE_DATE"}


def policy_tier(invoice: Invoice, settings: Settings) -> str:
    return "vp_scrutiny" if (invoice.total or 0) > settings.approval_threshold else "standard"


def policy_brief(settings: Settings) -> str:
    limit = f"${settings.approval_threshold:,.0f}"
    return (
        "ACME CORP INVOICE APPROVAL POLICY\n"
        "P1. Any CRITICAL finding means REJECT. This is fixed by policy; your job is to explain why in terms a vendor and a CFO would accept.\n"
        f"P2. Invoices over {limit} need additional scrutiny: APPROVE only when there are no WARNING findings at all. "
        "Otherwise ESCALATE to a human, or REJECT if the evidence shows it should not be paid.\n"
        f"P3. Invoices up to {limit}: you may APPROVE with warnings only if every warning is in this waivable list "
        f"{sorted(AGENT_WAIVABLE)} AND clearly benign; say why for each one. Any other warning requires ESCALATE or REJECT.\n"
        "P4. REJECT when evidence shows we should not pay as submitted: fraud signals, items we cannot have received (none exist in our catalog), "
        "arithmetic that does not reconcile (the vendor must send a corrected invoice), or an identical copy of an invoice already in process. "
        "ESCALATE when a human must verify a fact (bank details, purchase order, goods receipt, FX rate).\n"
        f"P5. Payments settle in {settings.base_currency} only. Foreign-currency invoices must be ESCALATED for treasury.\n"
        "P6. A wrong payment is far more expensive than a one-day delay. Never approve to be helpful.\n"
        "P7. Cite the finding codes your decision relies on. Do not invent facts that are not in the findings."
    )


def enforce(decision: Decision, invoice: Invoice, report: ValidationReport, settings: Settings) -> tuple[Decision, list[str]]:
    """Return the decision policy allows, plus a note for every override applied."""
    overrides: list[str] = []
    final = decision
    tier = policy_tier(invoice, settings)
    currency = invoice.currency or settings.base_currency

    def move(to: str, why: str) -> None:
        nonlocal final
        overrides.append(f"{final.decision} -> {to}: {why}")
        final = Decision(decision=to, rationale=f"{final.rationale} [Policy override: {why}]",
                         cited_findings=final.cited_findings, conditions=final.conditions)

    if report.critical and final.decision != "REJECT":
        move("REJECT", "critical findings present: " + ", ".join(sorted({f.code for f in report.critical})))
    elif final.decision == "APPROVE":
        if report.critical:
            move("REJECT", "critical findings present: " + ", ".join(sorted({f.code for f in report.critical})))
        elif invoice.total is None or invoice.total <= 0:
            move("REJECT", "no positive amount to pay")
        elif blocking := sorted({f.code for f in report.warnings if f.code not in AGENT_WAIVABLE or f.source == "llm"}):
            move("ESCALATE", "warnings only a human may clear: " + ", ".join(blocking))
        elif tier == "vp_scrutiny" and report.warnings:
            move("ESCALATE", f"over ${settings.approval_threshold:,.0f} with unresolved warnings: "
                             + ", ".join(sorted({f.code for f in report.warnings})))
        elif currency != settings.base_currency:
            move("ESCALATE", f"{currency} invoice cannot settle on a {settings.base_currency} rail without treasury")
    elif final.decision == "REJECT" and not report.critical and not report.warnings:
        move("ESCALATE", "agent rejected an invoice with no findings; a human must confirm")
    return final, overrides


def fast_path_reason(extraction, report, settings: Settings) -> str | None:
    """Why this invoice may skip the auditor round, or None if it needs the full VP + auditor loop.

    Only boring invoices qualify: structured file (no LLM reading), no warnings or critical
    findings (the investigator has already run), zero rule risk, low investigator risk,
    under the approval limit, in the base currency. Anything else gets the full debate.
    """
    if not settings.fast_path or extraction is None or report is None:
        return None
    inv = extraction.invoice
    if extraction.method != "structured" or report.critical or report.warnings or report.risk_score > 0:
        return None
    if (report.llm_risk or 0) > settings.fast_path_max_llm_risk:
        return None
    if inv.total is None or inv.total >= settings.approval_threshold:
        return None
    if (inv.currency or settings.base_currency).upper() != settings.base_currency:
        return None
    return "structured file, no warnings, zero fraud risk, under the approval limit"
