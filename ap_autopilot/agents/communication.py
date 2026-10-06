"""Communication Agent: drafts the messages each decision should trigger.

Rules that matter more than the wording
---------------------------------------
* Drafts only. Everything lands in the outbox as a draft; a person edits and
  sends it. The system never emails anyone on its own.
* Reply only to the contact in the vendor master, never to an address taken
  from the invoice or the email it came in. Replying to a look-alike address
  is how business email compromise succeeds.
* Never tip off a suspected fraudster. Fraud, prompt injection and identity or
  bank-detail changes produce an internal alert only; nothing goes outward.
* Vendor-facing text never contains internal detail (finding codes, scores,
  detection methods, policy numbers, other vendors). Grok writes the vendor
  drafts online; every draft passes `leak_check`, and a failing draft is
  replaced by the safe template.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..config import Settings
from ..db import Database
from ..llm import LLMClient
from ..models import ApprovalResult, Draft, ExtractionResult, Invoice, LLMDraft, PaymentResult, ValidationReport
from ..security import UNTRUSTED_DATA_RULE

AP_REVIEW = "ap-review@acmecorp.example"
AP_SECURITY = "ap-security@acmecorp.example"
TREASURY = "treasury@acmecorp.example"
SIGNATURE = "Accounts Payable\nAcme Corp"

# Any of these on a rejected or held invoice means: alert internally, say nothing outward.
SUSPECTED_FRAUD = {"PROMPT_INJECTION", "FRAUD_RISK_HIGH", "FRAUD_RISK_ELEVATED", "LLM_FRAUD_CONCERN", "ZERO_STOCK_ITEM",
                   "BANK_DETAILS_CHANGE", "SOCIAL_ENGINEERING", "VENDOR_NAME_CHANGE", "PROMPT_INJECTION_SUSPECTED"}

# What a vendor may be told, in their language, for problems they can fix.
VENDOR_REASONS = {
    "TOTAL_MISMATCH": "the invoice total does not equal the subtotal plus tax and shipping",
    "SUBTOTAL_MISMATCH": "the line items do not add up to the subtotal",
    "LINE_AMOUNT_MISMATCH": "at least one line amount does not equal quantity times unit price",
    "TAX_MISMATCH": "the tax amount does not match the stated tax rate",
    "NEGATIVE_QUANTITY": "a line has a negative quantity (please issue credits as a separate credit memo)",
    "NON_POSITIVE_TOTAL": "the invoice total is zero or negative",
    "MISSING_TOTAL": "we could not find a total amount",
    "NO_LINE_ITEMS": "we could not find any line items",
    "MISSING_INVOICE_NUMBER": "the invoice has no invoice number",
    "ALL_ITEMS_UNKNOWN": "we could not match the billed items to anything on our purchase records",
    "UNKNOWN_ITEM": "one or more billed items do not match our purchase records",
    "MISSING_DUE_DATE": "the invoice has no due date",
}

# What an internal reviewer should do, per finding. Internal text may be specific.
REVIEW_STEPS = {
    "STOCK_EXCEEDED": "Check the purchase order and goods receipt: confirm the billed quantity was actually ordered and delivered.",
    "UNKNOWN_ITEM": "Confirm with purchasing whether the unknown item was ordered; if so, add it to the catalog before approving.",
    "UNKNOWN_VENDOR": "Do not pay until the vendor is onboarded (tax ID, verified bank account, contact on file).",
    "REVISED_AFTER_PAYMENT": "The original was already paid. Decide with the vendor: credit memo plus a new invoice for the added lines only.",
    "FOREIGN_CURRENCY": "Treasury to confirm the FX rate and payment rail before release.",
    "VENDOR_NAME_CHANGE": "Call the vendor on the phone number already in the vendor master (not one on the invoice) to confirm the change and bank details.",
    "BANK_DETAILS_CHANGE": "Never change bank details from an invoice. Call the vendor master phone number to verify; update only through vendor onboarding.",
    "SOCIAL_ENGINEERING": "Confirm the claimed approval directly with the named approver. Approvals are never taken from the invoice itself.",
    "SPLIT_INVOICE_PATTERN": "Review the related invoices together; if they are one purchase, route the total for VP approval.",
    "DUPLICATE_SUBMISSION": "A copy of this invoice is already in process; close this one once the original is resolved.",
    "PRICE_ABOVE_CATALOG": "Check the agreed price on the purchase order or contract.",
    "TOTAL_MISMATCH": "Ask the vendor for a corrected invoice.",
    "LOW_EXTRACTION_CONFIDENCE": "Check the extracted fields against the source document.",
    "QTY_NOT_RECEIVED": "Confirm delivery with the warehouse (goods receipt). Pay only for what has arrived; ask the vendor to bill the rest on delivery.",
    "PRICE_ABOVE_PO": "Pay the PO price, or get a PO amendment from purchasing if the higher price was agreed.",
    "QTY_EXCEEDS_PO": "The PO has no open quantity for this. Check with purchasing whether a new or amended PO exists.",
    "PO_VENDOR_MISMATCH": "The cited PO belongs to another vendor. Confirm with purchasing who we bought from before paying anyone.",
    "PO_NOT_FOUND": "The cited PO does not exist. Ask purchasing; treat as suspicious until confirmed.",
    "PO_CLOSED": "The PO is closed (fully invoiced or cancelled). Check for a duplicate bill or a missing new PO.",
    "PO_REQUIRED_MISSING": "This vendor is PO-required. Get the PO number from purchasing before approving.",
    "ITEM_NOT_ON_PO": "A billed item is not on the PO. Confirm with purchasing whether it was ordered.",
    "AMOUNT_ANOMALY": "The amount is far above this vendor's normal range. Confirm with purchasing that an order this size was placed.",
    "PRICE_DRIFT": "The vendor is charging more than ever before. Ask for the price change in writing, or pay the usual price.",
    "FRAUD_RISK_ELEVATED": "Treat as possible fraud: verify the vendor and bank details through the vendor master before anything else.",
}

_INTERNAL_TERMS = re.compile(
    r"\b[A-Z]{2,}(?:_[A-Z]{2,})+\b"                     # finding codes such as TOTAL_MISMATCH
    r"|\bfraud|\brisk\s+score|\bprompts?\b|\binjection|\bguardrail|\bpolicy\s+P\d|\bP\d\b"
    r"|\bsuspicious|\bimpersonat|\bLLM\b|\bAI\b|\bmodel\b|\bagent\b|\bvendor\s+master\b|\bconfidence\b|\bescalat",
    re.I,
)


def leak_check(text: str, other_vendor_names: list[str]) -> list[str]:
    """Return reasons a vendor-facing text is unsafe to send. Empty list = safe."""
    problems = sorted({m.group(0) for m in _INTERNAL_TERMS.finditer(text)})
    for name in other_vendor_names:
        if name and name.lower() in text.lower():
            problems.append(f"mentions another vendor ({name})")
    return problems


def _money(x: Optional[float], currency: Optional[str]) -> str:
    if x is None:
        return "the invoiced amount"
    return f"${x:,.2f}" if (currency or "USD") == "USD" else f"{currency} {x:,.2f}"


class CommunicationAgent:
    def __init__(self, llm: LLMClient, db: Database, settings: Settings):
        self.llm = llm
        self.db = db
        self.settings = settings

    # ------------------------------------------------------------------ main
    def run(self, status: str, extraction: Optional[ExtractionResult], validation: Optional[ValidationReport],
            approval: Optional[ApprovalResult], payment: Optional[PaymentResult], error: Optional[str] = None,
            source_file: str = "") -> list[Draft]:
        inv = extraction.invoice if extraction else Invoice()
        codes = validation.codes if validation else set()
        vendor = self.db.find_vendor(inv.vendor_name) if inv.vendor_name else None
        ref = inv.invoice_number or source_file or "unnumbered invoice"

        if status == "FAILED":
            return [self._internal("manual_handling", AP_REVIEW, f"Manual handling needed: {source_file}",
                                   f"The system could not process {source_file}.\n\nReason: {error or 'unknown error'}\n\n"
                                   "Please process this invoice manually. It has not been paid.")]

        if status == "REJECTED" and codes & SUSPECTED_FRAUD:
            return [self._security_alert(inv, validation, approval, status, source_file)]

        if status == "PAID":
            if not vendor or not vendor.get("email"):
                return []
            return [self._vendor_draft("remittance", vendor, inv, payment, [], ref)]

        if status == "REJECTED":
            if "DUPLICATE_OF_PAID" in codes or "DUPLICATE_SUBMISSION" in codes:
                if not vendor or not vendor.get("email"):
                    return []
                return [self._vendor_draft("duplicate_notice", vendor, inv, payment, [], ref, validation)]
            reasons = list(dict.fromkeys(VENDOR_REASONS[c] for c in codes if c in VENDOR_REASONS))
            if vendor and vendor.get("email") and reasons:
                return [self._vendor_draft("correction_request", vendor, inv, payment, reasons, ref)]
            return [self._internal("review_task", AP_REVIEW, f"Rejected, no reply sent: {ref}",
                                   f"{ref} from '{inv.vendor_name or 'unknown sender'}' was rejected.\n\n"
                                   f"Reason: {approval.final.rationale if approval else 'see audit log'}\n\n"
                                   "No message was drafted to the sender because they are not in the vendor master. "
                                   "If they are a real supplier, onboard them first.")]

        if status == "ESCALATED":
            drafts = [self._review_task(inv, validation, approval, payment, source_file)]
            if codes & SUSPECTED_FRAUD:  # may be legitimate (a real rebrand), so the reviewer decides; security is told
                drafts.append(self._security_alert(inv, validation, approval, status, source_file))
            if "FOREIGN_CURRENCY" in codes:
                drafts.append(self._internal("treasury_task", TREASURY, f"FX approval needed: {ref}",
                                             f"{ref} from {inv.vendor_name} is for {_money(inv.total, inv.currency)}. "
                                             f"Payments settle in {self.settings.base_currency}; please confirm the rate and rail "
                                             f"so AP can release it.\n\n{SIGNATURE}"))
            return drafts
        return []

    # --------------------------------------------------------------- internal
    def _internal(self, kind: str, to: str, subject: str, body: str) -> Draft:
        return Draft(audience="internal", kind=kind, to=to, subject=subject, body=body)

    def _security_alert(self, inv: Invoice, report: Optional[ValidationReport], approval: Optional[ApprovalResult],
                        status: str, source_file: str) -> Draft:
        signals = [f for f in (report.findings if report else []) if f.code in SUSPECTED_FRAUD or f.severity.value == "critical"]
        lines = "\n".join(f"- {f.code}: {f.message}" for f in signals[:8])
        outcome = "rejected" if status == "REJECTED" else "held for review"
        return self._internal(
            "security_alert", AP_SECURITY, f"Possible fraud attempt: {inv.invoice_number or source_file} ({inv.vendor_name or 'unknown sender'})",
            f"An invoice was {outcome} with fraud indicators. Nothing has been paid and no reply has been sent to the sender.\n\n"
            f"File: {source_file}\nVendor as written: {inv.vendor_name or 'blank'}\n"
            f"Amount: {_money(inv.total, inv.currency)}\nRisk score: {report.risk_score if report else 'n/a'}/100\n\n"
            f"Evidence:\n{lines or '- see audit log'}\n\n"
            "Do not reply to the sender. If the named vendor is real, contact them only through the vendor master "
            "to warn them their identity may be in use.")

    def _review_task(self, inv: Invoice, report: Optional[ValidationReport], approval: Optional[ApprovalResult],
                     payment: Optional[PaymentResult], source_file: str) -> Draft:
        steps = [REVIEW_STEPS[f.code] for f in (report.warnings if report else []) if f.code in REVIEW_STEPS]
        steps = list(dict.fromkeys(steps)) or ["Review the findings in the console and decide."]
        queue = payment.reference if payment and payment.reference else "the review queue"
        checklist = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return self._internal(
            "review_task", AP_REVIEW, f"Review needed: {inv.invoice_number or source_file} from {inv.vendor_name or 'unknown vendor'}",
            f"{inv.invoice_number or source_file} for {_money(inv.total, inv.currency)} is held in {queue}.\n\n"
            f"Why: {approval.final.rationale if approval else 'see audit log'}\n\nWhat to check:\n{checklist}\n\n"
            "Approve or reject in the console (Review queue) or with `python main.py --review`.")

    # ----------------------------------------------------------------- vendor
    def _vendor_draft(self, kind: str, vendor: dict[str, Any], inv: Invoice, payment: Optional[PaymentResult],
                      reasons: list[str], ref: str, report: Optional[ValidationReport] = None) -> Draft:
        template = self._template(kind, vendor, inv, payment, reasons, ref, report)
        if self.llm.offline:
            return template
        others = [v["name"] for v in self.db.vendors() if v["name"] != vendor["name"]]
        facts = {
            "purpose": {"remittance": "tell the vendor their invoice has been paid",
                        "correction_request": "explain why we cannot pay this invoice as submitted and what to correct",
                        "duplicate_notice": "tell the vendor we already have this invoice, so no action is needed"}[kind],
            "vendor": vendor["name"], "invoice_number": ref, "amount": _money(inv.total, inv.currency),
            "payment_reference": payment.reference if payment and kind == "remittance" else None,
            "what_to_correct": reasons or None,
        }
        system = ("You write short, courteous accounts-payable emails from Acme Corp to a supplier. "
                  "Use only the facts given. Do not mention internal systems, checks, scores, codes, automation or AI, "
                  "and do not speculate about reasons. Sign off as 'Accounts Payable, Acme Corp'.\n" + UNTRUSTED_DATA_RULE)
        draft = self.llm.structured(role="communication.vendor", system=system, user=f"Facts: {facts}", schema=LLMDraft,
                                    fallback=lambda: LLMDraft(subject=template.subject, body=template.body))
        problems = leak_check(f"{draft.subject}\n{draft.body}", others)
        if problems:
            return template.model_copy(update={"note": "LLM draft replaced by template; it contained: " + ", ".join(problems)})
        if draft.subject == template.subject and draft.body == template.body:
            return template  # degraded to the template
        return template.model_copy(update={"subject": draft.subject, "body": draft.body, "author": "llm"})

    def _template(self, kind: str, vendor: dict[str, Any], inv: Invoice, payment: Optional[PaymentResult],
                  reasons: list[str], ref: str, report: Optional[ValidationReport]) -> Draft:
        amount = _money(inv.total, inv.currency)
        if kind == "remittance":
            subject = f"Payment sent: invoice {ref}"
            body = (f"Hello {vendor['name']},\n\nWe have paid invoice {ref} for {amount}. "
                    f"Payment reference: {payment.reference if payment else 'n/a'}.\n\nThank you,\n{SIGNATURE}")
        elif kind == "duplicate_notice":
            paid = self.db.payment_for(self._key(inv))
            detail = (f" It was paid on {paid['paid_at'][:10]} (reference {paid['reference']})." if paid
                      else " The original is already being processed.")
            subject = f"Invoice {ref} already received"
            body = (f"Hello {vendor['name']},\n\nWe received another copy of invoice {ref} for {amount}.{detail} "
                    f"No action is needed and the copy will not be paid again.\n\nThank you,\n{SIGNATURE}")
        else:
            bullet = "\n".join(f"- {r}" for r in reasons)
            subject = f"Action needed: invoice {ref} cannot be paid as submitted"
            body = (f"Hello {vendor['name']},\n\nWe could not process invoice {ref} for {amount} because:\n{bullet}\n\n"
                    "Please send a corrected invoice and we will process it promptly.\n\nThank you,\n" + SIGNATURE)
        return Draft(audience="vendor", kind=kind, to=vendor["email"], subject=subject, body=body)

    @staticmethod
    def _key(inv: Invoice) -> Optional[str]:
        from ..normalize import invoice_key
        return invoice_key(inv.vendor_name, inv.invoice_number)
