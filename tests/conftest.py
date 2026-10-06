"""Shared fixtures. A scripted fake of the OpenAI-compatible API lets us test the
LLM code paths (schema repair, tool calling, reflection, guardrails) without a
key or network, and lets us make the 'model' misbehave on purpose."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ap_autopilot.config import Settings  # noqa: E402
from ap_autopilot.db import Database  # noqa: E402
from ap_autopilot.graph import InvoicePipeline  # noqa: E402
from ap_autopilot.llm import OfflineLLM  # noqa: E402

INVOICES = ROOT / "data" / "invoices"
EXTRA = ROOT / "data" / "invoices_extra"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # pinned so a developer's .env (e.g. AP_USE_OCR=0) cannot change test behaviour
    return Settings(db_path=str(tmp_path / "test.db"), log_dir=str(tmp_path / "logs"), llm_provider="offline",
                    use_local_ocr=True, fast_path=True, read_workers=4)


@pytest.fixture
def db(settings: Settings) -> Database:
    return Database(settings.db_path).ensure()


@pytest.fixture
def pipeline(settings: Settings, db: Database) -> InvoicePipeline:
    return InvoicePipeline(settings, llm=OfflineLLM(), db=db)


# --------------------------------------------------------------------------- #
# Fake OpenAI-compatible client
# --------------------------------------------------------------------------- #
def text_reply(content: str) -> Any:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def tool_reply(*calls: tuple[str, dict]) -> Any:
    tool_calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                  for i, (name, args) in enumerate(calls)]
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


class FakeChatClient:
    """`responder(kwargs, call_index) -> response`. Records every request."""

    def __init__(self, responder: Callable[[dict, int], Any]):
        self.requests: list[dict] = []
        self._responder = responder
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responder(kwargs, len(self.requests) - 1)
