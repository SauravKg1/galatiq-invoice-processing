"""Evaluation harness: run the labelled set on a fresh database and score it.

Metrics, in order of business importance:
  wrongful_payments  paid something that must not be paid (must be 0)
  missed_payments    blocked something clean (costs late fees, vendor goodwill)
  status_accuracy    final status within the acceptable set
  flag_recall        known issues actually raised
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from . import ocr
from .config import PROJECT_ROOT, Settings
from .graph import InvoicePipeline
from .llm import LLMClient
from .models import ProcessingOutcome

EXPECTED_PATH = PROJECT_ROOT / "eval" / "expected_outcomes.yaml"


def load_expectations(path: Path = EXPECTED_PATH) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(path.read_text())


def run_eval(settings: Optional[Settings] = None, llm: Optional[LLMClient] = None,
             expected_path: Path = EXPECTED_PATH) -> dict[str, Any]:
    base = settings or Settings()
    expected = load_expectations(expected_path)
    with tempfile.TemporaryDirectory() as tmp:
        eval_settings = Settings(**{**base.__dict__, "db_path": str(Path(tmp) / "eval.db"),
                                    "log_dir": str(Path(tmp) / "logs")})
        pipeline = InvoicePipeline(eval_settings, llm=llm)
        can_read_scans = (not pipeline.llm.offline and pipeline.llm.supports_vision) or \
            (eval_settings.use_local_ocr and ocr.available())
        rows = []
        outcomes: list[ProcessingOutcome] = []
        started = time.perf_counter()
        batch = pipeline.iter_batch([PROJECT_ROOT / rel for rel in expected], workers=eval_settings.read_workers)
        for (rel_path, exp), outcome in zip(expected.items(), batch):
            outcomes.append(outcome)
            codes = outcome.validation.codes if outcome.validation else set()
            missing = [c for c in exp.get("must_flag", []) if c not in codes]
            paid = outcome.status == "PAID"
            skipped = bool(exp.get("needs_vision")) and not can_read_scans
            rows.append({
                "skipped": skipped,
                "file": Path(rel_path).name,
                "expected": "/".join(exp["status"]),
                "actual": outcome.status,
                "status_ok": outcome.status in exp["status"],
                "wrongful_payment": paid and not exp["pay"],
                "missed_payment": exp["pay"] and not paid,
                "missing_flags": missing,
                "flags_expected": len(exp.get("must_flag", [])),
                "why": exp.get("why", ""),
                "ms": outcome.total_ms,
            })
        wall_s = time.perf_counter() - started
        batch_wait_s = pipeline.batch_wait_s
    scored = [r for r in rows if not r["skipped"]]   # scans with no vision/OCR available are reported apart
    n = len(scored)
    flags_expected = sum(r["flags_expected"] for r in scored)
    flags_found = flags_expected - sum(len(r["missing_flags"]) for r in scored)
    return {
        "rows": rows,
        "outcomes": outcomes,
        "metrics": {
            "invoices": n,
            "status_accuracy": round(sum(r["status_ok"] for r in scored) / n, 3),
            "wrongful_payments": sum(r["wrongful_payment"] for r in rows),   # always counted, skipped or not
            "missed_payments": sum(r["missed_payment"] for r in scored),
            "needs_vision_skipped": len(rows) - n,
            "flag_recall": round(flags_found / flags_expected, 3) if flags_expected else 1.0,
            "avg_ms": round(sum(r["ms"] for r in scored) / n, 1),
        },
        "wall_s": wall_s,
        "batch_wait_s": batch_wait_s,
    }
