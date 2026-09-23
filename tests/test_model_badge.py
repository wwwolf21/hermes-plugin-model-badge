"""Pure-function tests; no Hermes runtime required (run: ``python -m pytest tests``)."""

import importlib.util
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("model_badge", _ROOT / "__init__.py")
mb = importlib.util.module_from_spec(_spec)
sys.modules["model_badge"] = mb
_spec.loader.exec_module(mb)


class _Ctx:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.hooks = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def _fresh(settings=None):
    ctx = _Ctx(settings)
    mb.register(ctx)
    mb._last_usage.clear()
    mb._last_thinking.clear()
    mb._ctx_cache.clear()
    return ctx


def test_pretty_model():
    assert mb.pretty_model("claude-fable-5-1") == "Claude Fable 5.1"
    assert mb.pretty_model("gpt-6-sol") == "GPT 6 Sol"
    assert mb.pretty_model("qwen/qwen3.8:27b") == "Qwen3.8 27b"
    assert mb.pretty_model("claude-sonnet-4-20250805") == "Claude Sonnet 4"


def test_fmt_tokens_and_bar():
    assert mb.fmt_tokens(950) == "950"
    assert mb.fmt_tokens(5300) == "5.3k"
    assert mb.fmt_tokens(48350) == "48k"
    assert mb.fmt_tokens(1_000_000) == "1M"
    assert mb.fmt_tokens(1_500_000) == "1.5M"
    assert mb.bar(0, 100) == "▱▱▱▱▱"
    assert mb.bar(1, 100) == "▰▱▱▱▱"  # non-zero usage always shows one segment
    assert mb.bar(100, 100) == "▰▰▰▰▰"
    assert mb.bar(50, 100, width=4) == "▰▰▱▱"


def test_thinking_from_request_covers_all_transports():
    T = mb.thinking_from_request
    assert T({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}) == "high"
    assert T({"thinking": {"type": "adaptive"}}) == "auto"
    assert T({"thinking": {"type": "disabled"}}) == "off"
    assert T({"thinking": {"type": "enabled", "budget_tokens": 8000}}) == "budget:8.0k"
    assert T({"model": "x", "messages": []}) is None
    assert T({"reasoning": {"effort": "medium"}}) == "medium"
    assert T({"reasoning": {"effort": "none"}}) == "off"
    assert T({"extra_body": {"reasoning": {"enabled": True, "effort": "low"}}}) == "low"
    assert T({"extra_body": {"reasoning": {"enabled": False}}}) == "off"
    assert T({"reasoning_effort": "high"}) == "high"
    assert T(None) is None


def test_thinking_label():
    assert mb.thinking_label(None) == "🧠A"
    assert mb.thinking_label("off") == "🚫"
    assert mb.thinking_label("high") == "🧠H"
    assert mb.thinking_label("medium") == "🧠M"
    assert mb.thinking_label("low") == "🧠L"
    assert mb.thinking_label("max") == "🧠+"
    assert mb.thinking_label("budget:8.0k") == "🧠8.0k"


def test_footer_full_flow(monkeypatch):
    ctx = _fresh()
    monkeypatch.setattr(mb, "_context_length", lambda *a: 1_000_000)
    ctx.hooks["pre_api_request"](session_id="s", request={"body": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}})
    ctx.hooks["post_api_request"](session_id="s", model="claude-fable-5-1", provider="anthropic", base_url="", usage={"prompt_tokens": 48350})
    out = ctx.hooks["transform_llm_output"](response_text="Ответ.", model="claude-fable-5-1", platform="telegram", session_id="s")
    assert out == "Ответ.\n\n> ![🎭](tg://emoji?id=5302992173296300813) `Claude Fable 5.1 · 🧠H ▰▱▱▱▱ 4% · 48k/1M`"


def test_footer_uses_model_of_last_call_not_agent_model(monkeypatch):
    ctx = _fresh()
    monkeypatch.setattr(mb, "_context_length", lambda *a: 272_000)
    ctx.hooks["post_api_request"](session_id="s", model="gpt-6-sol", provider="openai-codex", usage={"prompt_tokens": 120_000})
    out = ctx.hooks["transform_llm_output"](response_text="x", model="claude-fable-5-1", platform="telegram", session_id="s")
    assert "GPT 6 Sol" in out and "🤖" in out


def test_non_telegram_and_subagents_untouched():
    ctx = _fresh()
    hook = ctx.hooks["transform_llm_output"]
    assert hook(response_text="x", model="claude-fable-5-1", platform="subagent", session_id="s") is None
    assert hook(response_text="x", model="claude-fable-5-1", platform="discord", session_id="s") is None
    assert hook(response_text="", model="claude-fable-5-1", platform="telegram", session_id="s") is None


def test_unknown_model_gets_no_footer():
    ctx = _fresh()
    assert ctx.hooks["transform_llm_output"](response_text="x", model="mystery-9000", platform="telegram", session_id="s") is None


def test_no_double_footer():
    ctx = _fresh()
    once = ctx.hooks["transform_llm_output"](response_text="x", model="claude-fable-5-1", platform="telegram", session_id="s")
    assert once is not None
    assert ctx.hooks["transform_llm_output"](response_text=once, model="claude-fable-5-1", platform="telegram", session_id="s") is None


def test_settings_override_rows_platforms_and_toggles(monkeypatch):
    ctx = _fresh({
        "badges": [{"match": "mystery", "emoji_id": "1", "fallback": "🧪"}, {"match": "claude", "emoji_id": "2"}],
        "platforms": ["discord"],
        "show_thinking": False,
        "show_context": False,
    })
    hook = ctx.hooks["transform_llm_output"]
    ctx.hooks["pre_api_request"](session_id="s", request={"body": {"reasoning": {"effort": "high"}}})
    ctx.hooks["post_api_request"](session_id="s", model="mystery-9000", usage={"prompt_tokens": 500})
    assert hook(response_text="x", model="mystery-9000", platform="telegram", session_id="s") is None
    out = hook(response_text="x", model="mystery-9000", platform="discord", session_id="s")
    assert out == "x\n\n> ![🧪](tg://emoji?id=1) `Mystery 9000`"
    # user row shadows the built-in claude row
    assert "tg://emoji?id=2" in hook(response_text="x", model="claude-x", platform="discord", session_id="none")


def test_session_map_is_bounded():
    ctx = _fresh()
    for i in range(mb._MAX_SESSIONS + 10):
        ctx.hooks["post_api_request"](session_id=f"s{i}", model="m", usage={"prompt_tokens": 1})
    assert len(mb._last_usage) == mb._MAX_SESSIONS
    assert "s0" not in mb._last_usage
