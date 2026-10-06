"""Observability: every stage emits structured events.

Events go to three places: a JSON-lines log file (machine-readable, ships to
any log stack), the SQLite audit_log (queryable history per invoice), and the
in-memory trace returned with the outcome (what the CLI and UI render).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .db import Database

logger = logging.getLogger("ap_autopilot")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


class Tracer:
    def __init__(self, run_id: str, db: Optional[Database], log_dir: Optional[str], echo: bool = False):
        self.run_id = run_id
        self.echo = echo
        self.db = db
        self.invoice_key: Optional[str] = None
        self.events: list[dict[str, Any]] = []
        self.timings_ms: dict[str, float] = {}
        self._log_path: Optional[Path] = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._log_path = Path(log_dir) / f"ap_events_{day}.jsonl"

    def event(self, stage: str, event: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "invoice_key": self.invoice_key,
            "stage": stage,
            "event": event,
            **payload,
        }
        self.events.append(record)
        logger.debug("%s.%s %s", stage, event, payload)
        line = json.dumps(record, default=str)
        if self.echo:
            print(line, file=sys.stderr, flush=True)  # stderr keeps --json stdout clean
        if self._log_path:
            with self._log_path.open("a") as fh:
                fh.write(line + "\n")
        if self.db:
            try:
                self.db.audit(self.run_id, self.invoice_key, stage, event, payload or None)
            except Exception:  # audit must never take the pipeline down
                logger.exception("audit write failed")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        self.event(name, "started")
        try:
            yield
        except Exception as exc:
            self.event(name, "error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed
            self.event(name, "finished", duration_ms=elapsed)
