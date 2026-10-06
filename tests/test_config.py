"""The .env loader must accept the documented .env.example as-is."""

from ap_autopilot.config import PROJECT_ROOT, _clean_value


def test_inline_comments_are_stripped():
    assert _clean_value("auto            # auto | grok | openai | offline") == "auto"
    assert _clean_value("grok-4\t# any model") == "grok-4"
    assert _clean_value("") == ""


def test_quoted_values_keep_hashes():
    assert _clean_value('"abc#123"  # key') == "abc#123"
    assert _clean_value("'x y'") == "x y"


def test_env_example_parses_to_clean_values():
    for line in (PROJECT_ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            value = _clean_value(line.split("=", 1)[1])
            assert "#" not in value and value == value.strip(), line


def test_check_llm_explains_a_bad_model_instead_of_crashing(monkeypatch, capsys):
    import main
    from ap_autopilot.llm import ChatLLM
    from conftest import FakeChatClient

    def refuse(kw, i):
        raise type("NotFoundError", (Exception,), {"status_code": 404})("Error code: 404 - model grok-old not found")

    llm = ChatLLM(provider="grok", api_key="x", base_url="http://fake", model="grok-old",
                  client=FakeChatClient(refuse), api_retries=0)
    monkeypatch.setattr(main, "build_llm", lambda settings, provider=None: llm)
    assert main.main(["--check-llm"]) == 2
    out = capsys.readouterr().out
    assert "FAILED" in out and "model ID" in out
