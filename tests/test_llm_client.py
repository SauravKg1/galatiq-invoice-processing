"""LLM client behaviour against a scripted fake provider."""

import json

from pydantic import BaseModel

from ap_autopilot.llm import ChatLLM, ToolSpec, parse_json_block
from conftest import FakeChatClient, text_reply, tool_reply


class Answer(BaseModel):
    value: int
    label: str


def make(responder):
    fake = FakeChatClient(responder)
    return ChatLLM(provider="fake", api_key="x", base_url="http://fake", model="fake-1",
                   api_retries=1, repair_attempts=2, client=fake), fake


def test_parse_json_block_handles_fences_and_prose():
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_schema_repair_loop_feeds_errors_back():
    replies = iter([text_reply('{"value": "not a number", "label": "x"}'), text_reply('{"value": 3, "label": "x"}')])
    llm, fake = make(lambda kw, i: next(replies))
    result = llm.structured(role="t", system="s", user="u", schema=Answer)
    assert result == Answer(value=3, label="x")
    second_request = fake.requests[1]["messages"]
    assert "failed schema validation" in second_request[-1]["content"]  # the model saw its own error


def test_falls_back_to_json_object_when_json_schema_unsupported():
    class BadRequest(Exception):
        status_code = 400

    def responder(kw, i):
        if kw["response_format"]["type"] == "json_schema":
            raise BadRequest("response_format json_schema not supported")
        return text_reply('{"value": 1, "label": "ok"}')

    llm, fake = make(responder)
    assert llm.structured(role="t", system="s", user="u", schema=Answer).value == 1
    assert fake.requests[-1]["response_format"] == {"type": "json_object"}


def test_provider_outage_degrades_to_deterministic_fallback():
    def responder(kw, i):
        raise ConnectionError("network down")

    llm, _ = make(responder)
    llm.api_retries = 0
    result = llm.structured(role="t", system="s", user="u", schema=Answer, fallback=lambda: Answer(value=0, label="rules"))
    assert result.label == "rules"
    assert any("DEGRADED" in c["note"] for c in llm.calls)


def test_tool_loop_executes_tools_then_submits():
    seen = []

    def lookup(item_name):
        seen.append(item_name)
        return {"found": True, "stock": 5}

    replies = iter([
        tool_reply(("lookup", {"item_name": "GadgetX"})),
        tool_reply(("submit_result", {"value": "bad"})),          # invalid submission -> error returned to model
        tool_reply(("submit_result", {"value": 5, "label": "in stock"})),
    ])
    llm, fake = make(lambda kw, i: next(replies))
    tool = ToolSpec("lookup", "look up", {"type": "object", "properties": {"item_name": {"type": "string"}}}, lookup)
    result, calls = llm.tool_loop(role="t", system="s", user="u", tools=[tool], final_schema=Answer)
    assert result.value == 5 and seen == ["GadgetX"]
    assert calls[0]["tool"] == "lookup"
    tool_messages = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
    assert any("schema validation failed" in m["content"] for m in tool_messages)
    assert json.loads(tool_messages[0]["content"]) == {"found": True, "stock": 5}


def test_adapts_when_model_rejects_temperature():
    class BadRequest(Exception):
        status_code = 400

    def responder(kw, i):
        if "temperature" in kw:
            raise BadRequest("Model grok-x does not support parameter: temperature")
        return text_reply('{"value": 9, "label": "ok"}')

    llm, fake = make(responder)
    assert llm.structured(role="t", system="s", user="u", schema=Answer).value == 9
    assert "temperature" not in fake.requests[-1]
    assert fake.requests[-1]["response_format"]["type"] == "json_schema"  # kept the better mode


def test_nested_schemas_are_inlined_for_tools():
    from ap_autopilot.llm import inline_refs
    from ap_autopilot.models import LLMValidationReview

    schema = inline_refs(LLMValidationReview.model_json_schema())
    assert "$ref" not in str(schema) and "$defs" not in str(schema)
    assert schema["properties"]["additional_findings"]["items"]["properties"]["severity"]["enum"] == ["info", "warning"]
