"""LLM access layer.

Design goals
------------
1. One interface, swappable providers. Grok is the primary reasoning engine
   (xAI exposes an OpenAI-compatible API), any OpenAI-compatible model is a
   drop-in alternative, and an offline mode runs the same pipeline with
   deterministic rules so the system works with no key and no internet.
2. Structured outputs only. Every call targets a Pydantic schema. If the model
   returns invalid JSON or breaks the schema, the validation error is fed back
   and the model repairs its own output (self-correction loop #1).
3. Real tool use. `tool_loop` runs a bounded function-calling loop: the model
   calls our tools (inventory lookups, vendor master, invoice history) and
   finishes by calling `submit_result` with schema-valid arguments.
4. Graceful degradation. If the provider is down or keeps producing garbage,
   each call falls back to a deterministic implementation and the degradation
   is recorded in the trace. An invoice is never dropped because an API hiccuped.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Optional[Callable[..., Any]] = None

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` into a self-contained schema. Several providers reject
    `$ref` inside function parameters or response_format schemas."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**resolve(target), **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def parse_json_block(text: Optional[str]) -> Any:
    """Accept raw JSON, ```json fenced``` JSON, or JSON embedded in prose."""
    if not text:
        raise ValueError("empty model response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("no JSON object found in model response")
        return json.loads(match.group(0))


# Roles that are pattern work (copying fields, writing a polite email) go to the fast model;
# everything that judges risk or money goes to the reasoning model.
FAST_ROLES = ("ingestion.extract", "ingestion.self_correct", "communication.")


def is_fast_role(role: str) -> bool:
    return role.startswith(FAST_ROLES)


class LLMClient:
    """Base class. Subclasses implement `structured` and `tool_loop`."""

    provider = "offline"
    model = "deterministic-rules"
    offline = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._local = threading.local()

    @contextmanager
    def tagged(self, tag: str) -> Iterator[None]:
        """Label calls made on this thread (parallel reads in a batch), so each invoice keeps its own calls."""
        previous = getattr(self._local, "tag", None)
        self._local.tag = tag
        try:
            yield
        finally:
            self._local.tag = previous

    def model_for(self, role: str) -> str:
        return self.model

    supports_vision = False

    def _record(self, role: str, ok: bool, latency_ms: float = 0.0, note: str = "", usage: Any = None,
                model: Optional[str] = None) -> None:
        self.calls.append({
            "role": role,
            "provider": self.provider,
            "model": model or self.model,
            "ok": ok,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "note": note,
            "tag": getattr(getattr(self, "_local", None), "tag", None),
        })

    def structured(self, *, role: str, system: str, user: str, schema: type[T],
                   fallback: Optional[Callable[[], T]] = None, images: Optional[list[bytes]] = None) -> T:
        """`images` (PNG bytes) are attached to the user message and routed to the vision model."""
        raise NotImplementedError

    def tool_loop(self, *, role: str, system: str, user: str, tools: list[ToolSpec], final_schema: type[T],
                  fallback: Optional[Callable[[], T]] = None, max_steps: int = 8) -> tuple[T, list[dict[str, Any]]]:
        raise NotImplementedError


class OfflineLLM(LLMClient):
    """No model: every step runs its deterministic implementation.
    This is the baseline the LLM must beat, and what the test suite runs against."""

    def structured(self, *, role, system, user, schema, fallback=None, images=None):
        if fallback is None:
            raise LLMError(f"{role}: offline mode needs a deterministic fallback")
        self._record(role, ok=True, note="offline rules")
        return fallback()

    def tool_loop(self, *, role, system, user, tools, final_schema, fallback=None, max_steps=8):
        if fallback is None:
            raise LLMError(f"{role}: offline mode needs a deterministic fallback")
        self._record(role, ok=True, note="offline rules")
        return fallback(), []


class ChatLLM(LLMClient):
    """OpenAI-compatible chat client. Used for xAI Grok and any compatible provider."""

    offline = False

    def __init__(self, *, provider: str, api_key: str, base_url: str, model: str, timeout_s: float = 60.0,
                 api_retries: int = 2, repair_attempts: int = 2, client: Any = None,
                 vision_model: Optional[str] = None, fast_model: Optional[str] = None,
                 reasoning_model: Optional[str] = None):
        super().__init__()
        self.provider = provider
        self.model = model
        self.vision_model = vision_model or model
        self.fast_model = fast_model or model
        self.reasoning_model = reasoning_model or model
        self.supports_vision = True
        self.api_retries = api_retries
        self.repair_attempts = repair_attempts
        self._json_schema_supported = True
        self._sampling: dict[str, Any] = {"temperature": 0}  # dropped if the model rejects it
        if client is None:
            from openai import OpenAI  # imported lazily so offline mode needs no SDK config
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self.client = client

    def model_for(self, role: str) -> str:
        return self.fast_model if is_fast_role(role) else self.reasoning_model

    # ------------------------------------------------------------------ core
    def _create(self, role: str, model: Optional[str] = None, **kwargs: Any) -> Any:
        """One chat completion with bounded exponential backoff on transient errors."""
        model = model or self.model_for(role)
        last_exc: Optional[Exception] = None
        for attempt in range(self.api_retries + 1):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(model=model, **kwargs)
                self._record(role, ok=True, latency_ms=(time.perf_counter() - start) * 1000,
                             usage=getattr(response, "usage", None), model=model)
                return response
            except Exception as exc:  # SDK raises many subclasses; classify below
                last_exc = exc
                self._record(role, ok=False, latency_ms=(time.perf_counter() - start) * 1000,
                             note=f"{type(exc).__name__}: {str(exc)[:160]}", model=model)
                if _is_bad_request(exc):
                    raise  # retrying an invalid request is pointless; caller may adapt
                if attempt < self.api_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"{role}: provider call failed after retries: {last_exc}")

    # ------------------------------------------------------- structured JSON
    def structured(self, *, role, system, user, schema, fallback=None, images=None):
        schema_json = inline_refs(schema.model_json_schema())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + _schema_instruction(schema_json)},
            {"role": "user", "content": _user_content(user, images)},
        ]
        model = self.vision_model if images else self.model_for(role)
        attempts = 1 + self.repair_attempts
        for attempt in range(attempts):
            try:
                response = self._create(role, model=model, messages=messages, **self._sampling,
                                        response_format=self._response_format(schema.__name__, schema_json))
            except Exception as exc:
                if _is_bad_request(exc) and self._adapt_to(exc):
                    continue
                return self._degrade(role, fallback, exc)
            content = response.choices[0].message.content
            try:
                return schema.model_validate(parse_json_block(content))
            except (ValueError, ValidationError) as exc:
                # Self-correction: show the model its own output and the exact error.
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({"role": "user", "content": (
                    f"Your previous output failed schema validation:\n{_short_error(exc)}\n"
                    "Return ONLY the corrected JSON object.")})
                self.calls[-1]["note"] = f"schema repair needed (attempt {attempt + 1})"
        return self._degrade(role, fallback, LLMError("output never satisfied the schema"))

    # ------------------------------------------------------------- tool loop
    def tool_loop(self, *, role, system, user, tools, final_schema, fallback=None, max_steps=8):
        submit = ToolSpec(
            name="submit_result",
            description="Submit your final answer. Call exactly once, after you have finished investigating.",
            parameters=inline_refs(final_schema.model_json_schema()),
        )
        registry = {t.name: t for t in tools}
        specs = [t.as_openai() for t in tools] + [submit.as_openai()]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        invocations: list[dict[str, Any]] = []
        try:
            for _ in range(max_steps):
                try:
                    response = self._create(role, messages=messages, tools=specs, tool_choice="auto", **self._sampling)
                except Exception as exc:
                    if _is_bad_request(exc) and "temperature" in self._sampling:
                        self._sampling.pop("temperature")
                        response = self._create(role, messages=messages, tools=specs, tool_choice="auto")
                    else:
                        raise
                message = response.choices[0].message
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                messages.append(_assistant_message(message))

                if not tool_calls:
                    try:  # some models answer in plain JSON instead of calling submit_result
                        return final_schema.model_validate(parse_json_block(message.content)), invocations
                    except (ValueError, ValidationError):
                        messages.append({"role": "user", "content": "Call submit_result with your final answer."})
                        continue

                final: Optional[BaseModel] = None
                for call in tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        result: Any = {"error": f"arguments were not valid JSON: {exc}"}
                    else:
                        if name == submit.name:
                            try:
                                final = final_schema.model_validate(args)
                                result = {"status": "accepted"}
                            except ValidationError as exc:
                                result = {"error": f"schema validation failed, fix and resubmit: {_short_error(exc)}"}
                        elif name in registry and registry[name].fn:
                            try:
                                result = registry[name].fn(**args)
                            except Exception as exc:
                                result = {"error": f"{type(exc).__name__}: {exc}"}
                            invocations.append({"tool": name, "args": args, "result": result})
                        else:
                            result = {"error": f"unknown tool '{name}'"}
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
                if final is not None:
                    return final, invocations
        except Exception as exc:
            return self._degrade(role, fallback, exc), invocations
        return self._degrade(role, fallback, LLMError(f"no final answer within {max_steps} steps")), invocations

    # --------------------------------------------------------------- helpers
    def _adapt_to(self, exc: Exception) -> bool:
        """Learn from a 400 once: drop an unsupported parameter, then retry."""
        text = str(exc).lower()
        if "temperature" in text and "temperature" in self._sampling:
            self._sampling.pop("temperature")
            return True
        if self._json_schema_supported:
            self._json_schema_supported = False  # provider lacks json_schema mode: use json_object
            return True
        if "temperature" in self._sampling:
            self._sampling.pop("temperature")
            return True
        return False

    def _response_format(self, name: str, schema_json: dict[str, Any]) -> dict[str, Any]:
        if self._json_schema_supported:
            return {"type": "json_schema", "json_schema": {"name": name, "schema": schema_json}}
        return {"type": "json_object"}

    def _degrade(self, role: str, fallback: Optional[Callable[[], T]], exc: Exception) -> T:
        if fallback is None:
            raise LLMError(f"{role}: {exc}") from exc
        self._record(role, ok=False, note=f"DEGRADED to deterministic fallback: {str(exc)[:160]}")
        return fallback()


# ----------------------------------------------------------------- utilities
def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (400, 422) or type(exc).__name__ in {"BadRequestError", "UnprocessableEntityError"}


def _user_content(text: str, images: Optional[list[bytes]]) -> Any:
    """OpenAI-compatible multimodal content: text part plus base64 PNG image parts."""
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for png in images:
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": "high"}})
    return parts


def _schema_instruction(schema_json: dict[str, Any]) -> str:
    return ("\n\nRespond with a single JSON object that conforms to this JSON Schema. No prose, no markdown.\n"
            + json.dumps(schema_json))


def _short_error(exc: Exception) -> str:
    return str(exc)[:1200]


def _assistant_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
        return {k: v for k, v in data.items() if k in {"role", "content", "tool_calls"}} | {"role": "assistant"}
    return {"role": "assistant", "content": getattr(message, "content", "")}


def build_llm(settings: Settings, provider: Optional[str] = None) -> LLMClient:
    choice = (provider or settings.resolved_provider()).lower()
    if choice == "auto":
        choice = settings.resolved_provider()
    if choice == "offline":
        return OfflineLLM()
    if choice == "grok":
        if not settings.xai_api_key:
            raise LLMError("LLM_PROVIDER=grok but XAI_API_KEY is not set (see .env.example)")
        return ChatLLM(provider="grok", api_key=settings.xai_api_key, base_url=settings.xai_base_url,
                       model=settings.xai_model, vision_model=settings.xai_vision_model or settings.xai_model,
                       fast_model=settings.xai_model_fast or None, reasoning_model=settings.xai_model_reasoning or None,
                       timeout_s=settings.llm_timeout_s,
                       api_retries=settings.llm_api_retries, repair_attempts=settings.llm_repair_attempts)
    if choice == "openai":
        if not settings.openai_api_key:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return ChatLLM(provider="openai", api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                       model=settings.openai_model, vision_model=settings.openai_vision_model or settings.openai_model,
                       fast_model=settings.openai_model_fast or None, reasoning_model=settings.openai_model_reasoning or None,
                       timeout_s=settings.llm_timeout_s,
                       api_retries=settings.llm_api_retries, repair_attempts=settings.llm_repair_attempts)
    raise LLMError(f"unknown LLM provider '{choice}' (use auto, grok, openai or offline)")
