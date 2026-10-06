"""Typed contracts between agents.

Agents never pass free text to each other. Each hand-off is one of these
Pydantic models, which gives us three things: the LLM is forced into a schema
(structured outputs), bad output fails loudly at the boundary (and triggers a
self-correction retry), and every intermediate result is inspectable and
testable on its own.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    description: str = Field(description="Item text exactly as written on the invoice")
    item: Optional[str] = Field(None, description="Catalog SKU if it is clearly the same product (e.g. 'Widget A' -> 'WidgetA'), else the item name as written")
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = Field(None, description="Line total as printed, if present")
    note: Optional[str] = Field(None, description="Qualifier such as 'rush order' or 'volume discount'")


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    invoice_date: Optional[date] = None
    due_date: Optional[date] = Field(None, description="ISO date, null if missing or not a real date")
    due_date_raw: Optional[str] = Field(None, description="Due date text exactly as written, e.g. 'yesterday'")
    currency: Optional[str] = Field(None, description="ISO currency code if stated, else null")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = Field(None, description="Decimal fraction, 0.05 for 5%")
    tax_amount: Optional[float] = None
    shipping: Optional[float] = None
    total: Optional[float] = None
    payment_terms: Optional[str] = None
    notes: Optional[str] = Field(None, description="Free-text notes or instructions on the invoice")
    revision: Optional[str] = None
    po_number: Optional[str] = Field(None, description="Purchase order number referenced on the invoice, e.g. 'PO-4500012', else null")

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if isinstance(v, str) and v.strip() else None


class VisionExtraction(BaseModel):
    """What the vision model returns for a scanned invoice or photo."""
    transcription: str = Field(description="Every piece of text visible on the page, word for word, top to bottom, "
                                           "including small, faint or unusual text. Do not summarise or omit anything.")
    invoice: Invoice


class ExtractionResult(BaseModel):
    invoice: Invoice
    source_file: str
    source_path: Optional[str] = None
    source_format: str
    method: Literal["structured", "llm", "heuristic", "vision", "ocr"]
    attempts: int = 1
    corrections: list[str] = Field(default_factory=list, description="Issues the self-correction loop fed back")
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    raw_text: str = ""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    INFO = "info"          # recorded, does not block
    WARNING = "warning"    # needs judgement: the approval agent or a human decides
    CRITICAL = "critical"  # hard stop: the invoice cannot be paid


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    item: Optional[str] = None
    source: Literal["rule", "llm"] = "rule"
    evidence: dict[str, Any] = Field(default_factory=dict)


class LLMFinding(BaseModel):
    """What the validation LLM may add. It cannot raise CRITICAL (see agents/validation.py)."""
    code: str = Field(description="UPPER_SNAKE_CASE identifier, e.g. SUSPICIOUS_VENDOR_ADDRESS")
    severity: Literal["info", "warning"]
    message: str
    item: Optional[str] = None


class LLMValidationReview(BaseModel):
    additional_findings: list[LLMFinding] = Field(default_factory=list)
    fraud_risk: int = Field(0, ge=0, le=100, description="Your independent fraud risk estimate, 0-100")
    summary: str = Field(description="Two or three sentences a VP of Finance can read")


class ValidationReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    risk_score: int = 0
    risk_signals: list[str] = Field(default_factory=list)
    item_matches: dict[str, Optional[str]] = Field(default_factory=dict)
    llm_summary: Optional[str] = None
    llm_risk: Optional[int] = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    po_match: Optional[dict[str, Any]] = None   # three-way match: PO, receipts and a per-line table
    history_context: Optional[dict[str, Any]] = None   # this vendor's amount baseline (synthetic history)

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def critical(self) -> list[Finding]:
        return self.by_severity(Severity.CRITICAL)

    @property
    def warnings(self) -> list[Finding]:
        return self.by_severity(Severity.WARNING)

    @property
    def codes(self) -> set[str]:
        return {f.code for f in self.findings}


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #
DecisionType = Literal["APPROVE", "REJECT", "ESCALATE"]


class Decision(BaseModel):
    decision: DecisionType
    rationale: str = Field(description="Plain-English reasoning a VP would sign")
    cited_findings: list[str] = Field(default_factory=list, description="Finding codes this decision relies on")
    conditions: list[str] = Field(default_factory=list, description="Follow-ups required, e.g. 'confirm bank details by phone'")


class Critique(BaseModel):
    agrees: bool
    issues: list[str] = Field(default_factory=list, description="Specific problems with the decision, empty if none")
    recommended_decision: DecisionType


class ApprovalRound(BaseModel):
    round: int
    decision: Decision
    critique: Critique


class ApprovalResult(BaseModel):
    final: Decision
    policy_tier: Literal["standard", "vp_scrutiny"]
    decided_by: Literal["policy", "agent", "human"]
    rounds: list[ApprovalRound] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list, description="Guardrails that changed the agent's decision")
    fast_path: bool = False   # clean invoice: VP approved, auditor round skipped (see policy.fast_path_reason)


# --------------------------------------------------------------------------- #
# Payment and outcome
# --------------------------------------------------------------------------- #
class PaymentResult(BaseModel):
    status: Literal["paid", "rejected", "queued", "skipped_duplicate", "failed"]
    vendor: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    reference: Optional[str] = None
    message: str = ""


class Draft(BaseModel):
    """A message the system prepared. Never sent automatically: a person edits and sends it."""
    id: Optional[int] = None
    audience: Literal["vendor", "internal"]
    kind: Literal["remittance", "correction_request", "duplicate_notice", "review_task", "security_alert",
                  "treasury_task", "manual_handling"]
    to: str
    subject: str
    body: str
    author: Literal["llm", "template"] = "template"
    note: Optional[str] = None   # e.g. why an LLM draft was replaced by the template


class LLMDraft(BaseModel):
    subject: str = Field(description="Short, specific email subject")
    body: str = Field(description="Plain-text email body, polite and concise, signed 'Accounts Payable, Acme Corp'")


OutcomeStatus = Literal["PAID", "REJECTED", "ESCALATED", "FAILED"]


class ProcessingOutcome(BaseModel):
    run_id: str
    source_file: str
    invoice_key: Optional[str] = None
    status: OutcomeStatus
    extraction: Optional[ExtractionResult] = None
    validation: Optional[ValidationReport] = None
    approval: Optional[ApprovalResult] = None
    payment: Optional[PaymentResult] = None
    error: Optional[str] = None
    llm_mode: str = "offline"
    timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    drafts: list[Draft] = Field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 1)

    @property
    def amount(self) -> Optional[float]:
        return self.extraction.invoice.total if self.extraction else None
