"""Payment Agent: deterministic on purpose.

Moving money is the one step with no LLM. It executes the approved decision,
enforces idempotency (one payment per vendor + invoice number, ever, backed by
a UNIQUE constraint), logs rejections with reasons, and queues escalations.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

from ..config import Settings
from ..db import Database
from ..models import ApprovalResult, Invoice, PaymentResult


def mock_payment(vendor, amount):
    """Provided by the case brief, kept verbatim. Stands in for the banking API."""
    print(f"Paid {amount} to {vendor}")
    return {"status": "success"}


class PaymentAgent:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def pay(self, invoice: Invoice, key: Optional[str], run_id: str) -> PaymentResult:
        vendor, amount = invoice.vendor_name, invoice.total
        currency = invoice.currency or self.settings.base_currency
        if not key or not vendor or amount is None or amount <= 0:
            return PaymentResult(status="failed", vendor=vendor, amount=amount, currency=currency,
                                 message="Refused: missing vendor, invoice number or positive amount.")
        existing = self.db.payment_for(key)
        if existing:  # idempotency guard, independent of what validation decided
            return PaymentResult(status="skipped_duplicate", vendor=vendor, amount=amount, currency=currency,
                                 reference=existing["reference"],
                                 message=f"Already paid on {existing['paid_at'][:10]}; no second payment made.")
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                response = mock_payment(vendor, amount)
        except Exception as exc:
            return PaymentResult(status="failed", vendor=vendor, amount=amount, currency=currency,
                                 message=f"Banking API error: {exc}")
        if (response or {}).get("status") != "success":
            return PaymentResult(status="failed", vendor=vendor, amount=amount, currency=currency,
                                 message=f"Banking API declined: {response}")
        reference = f"PAY-{run_id[-6:].upper()}"
        self.db.record_payment(key, vendor, amount, currency, reference, "success")
        return PaymentResult(status="paid", vendor=vendor, amount=amount, currency=currency, reference=reference,
                             message=captured.getvalue().strip() or "Payment sent")

    def reject(self, invoice: Invoice, approval: ApprovalResult) -> PaymentResult:
        return PaymentResult(status="rejected", vendor=invoice.vendor_name, amount=invoice.total,
                             currency=invoice.currency or self.settings.base_currency,
                             message="Not paid. Decision and reasons written to the audit log.")

    def escalate(self, invoice: Invoice, approval: ApprovalResult, key: Optional[str], run_id: str,
                 source_file: str) -> PaymentResult:
        review_id = self.db.enqueue_review(
            invoice_key=key, run_id=run_id, source_file=source_file, vendor=invoice.vendor_name,
            amount=invoice.total, currency=invoice.currency or self.settings.base_currency,
            reason=approval.final.rationale,
        )
        return PaymentResult(status="queued", vendor=invoice.vendor_name, amount=invoice.total,
                             currency=invoice.currency or self.settings.base_currency, reference=f"REVIEW-{review_id}",
                             message=f"Held for human review (queue item {review_id}).")
