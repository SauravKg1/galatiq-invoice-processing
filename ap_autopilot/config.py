"""Runtime settings.

Every business threshold lives here, not inside agent code, so finance can
change policy (approval limit, fraud cut-off) without touching the agents.
Values can be overridden with environment variables or a local `.env` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader (avoids an extra dependency). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), _clean_value(value))


def _clean_value(value: str) -> str:
    """`KEY=value   # comment` -> `value`; quoted values keep any '#' inside the quotes."""
    value = value.strip()
    if value[:1] in {'"', "'"} and value[0] in value[1:]:
        return value[1:value.index(value[0], 1)]
    return value.split(" #", 1)[0].split("\t#", 1)[0].strip()


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Storage and observability
    db_path: str = field(default_factory=lambda: _env("AP_DB_PATH", str(PROJECT_ROOT / "inventory.db")))
    log_dir: str = field(default_factory=lambda: _env("AP_LOG_DIR", str(PROJECT_ROOT / "logs")))
    log_to_console: bool = field(default_factory=lambda: _env("AP_LOG_EVENTS", "0") == "1")

    # LLM provider: auto | grok | openai | offline
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    xai_api_key: str = field(default_factory=lambda: _env("XAI_API_KEY", ""))
    xai_base_url: str = field(default_factory=lambda: _env("XAI_BASE_URL", "https://api.x.ai/v1"))
    xai_model: str = field(default_factory=lambda: _env("XAI_MODEL", "grok-4"))
    xai_vision_model: str = field(default_factory=lambda: _env("XAI_VISION_MODEL", ""))   # blank = XAI_MODEL
    # Model routing: reading and drafting are pattern work, judgement is not. Blank = XAI_MODEL.
    xai_model_fast: str = field(default_factory=lambda: _env("XAI_MODEL_FAST", ""))
    xai_model_reasoning: str = field(default_factory=lambda: _env("XAI_MODEL_REASONING", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_vision_model: str = field(default_factory=lambda: _env("OPENAI_VISION_MODEL", ""))
    openai_model_fast: str = field(default_factory=lambda: _env("OPENAI_MODEL_FAST", ""))
    openai_model_reasoning: str = field(default_factory=lambda: _env("OPENAI_MODEL_REASONING", ""))
    llm_timeout_s: float = 60.0
    llm_api_retries: int = 2          # transient API errors (network, 5xx, rate limits)
    llm_repair_attempts: int = 2      # schema-invalid output fed back to the model

    # Business policy
    base_currency: str = "USD"
    approval_threshold: float = field(default_factory=lambda: float(_env("AP_APPROVAL_THRESHOLD", "10000")))
    threshold_proximity_pct: float = 0.05   # within 5% under the limit = possible threshold dodging
    amount_tolerance: float = 0.05          # cents-level rounding tolerance for arithmetic checks
    price_variance_pct: float = 0.10        # unit price above catalog by >10% needs a reason
    fraud_reject_score: int = 70            # rule-based risk score that forces rejection
    fraud_review_score: int = 40            # risk score that forces human review

    # Agent loop limits (bounded loops are a production requirement, not an option)
    max_extraction_retries: int = 2
    use_local_ocr: bool = field(default_factory=lambda: _env("AP_USE_OCR", "1") == "1")  # Tesseract if installed
    max_critique_rounds: int = 2
    max_tool_steps: int = 8

    # Throughput (see ap_autopilot/bench.py for the before/after numbers)
    read_workers: int = field(default_factory=lambda: int(_env("AP_READ_WORKERS", "4")))  # parallel reads in a batch
    fast_path: bool = field(default_factory=lambda: _env("AP_FAST_PATH", "1") == "1")    # skip the auditor on clean invoices
    fast_path_max_llm_risk: int = 10        # investigator risk above this always gets the full VP + auditor loop

    def resolved_provider(self) -> str:
        """`auto` picks Grok when a key exists, otherwise runs fully offline."""
        provider = self.llm_provider.lower()
        if provider != "auto":
            return provider
        if self.xai_api_key:
            return "grok"
        if self.openai_api_key:
            return "openai"
        return "offline"
