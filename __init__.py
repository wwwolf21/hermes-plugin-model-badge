"""Model badge - a footer under every Telegram reply.

    > 🎭 `Claude Fable 5.1 · 🧠H ▰▱▱▱▱ 4% · 48k/1M`

Provider logo (Telegram custom emoji, MarkdownV2 ``![emoji](tg://emoji?id=N)``), a human-readable
model name, the reasoning effort that actually went out on the wire and the context usage of the
last API call (``prompt_tokens / context_length``).

Data flow: ``pre_api_request`` records the thinking setting of every outgoing request,
``post_api_request`` records its usage; ``transform_llm_output`` (once per turn, after the tool
loop) renders the footer from the LAST call - i.e. the live context size and the model that really
answered (fallback chains can swap models mid-turn).

Custom emoji render only when the bot owner has Telegram Premium; otherwise Telegram shows the
fallback unicode emoji. The adapter must pass ``![..](tg://emoji?id=N)`` through unescaped
(hermes-agent PR: "pass MarkdownV2 custom emoji links through format_message").
"""

import logging
import re
import threading

logger = logging.getLogger(__name__)

# Built-in rows: (regex on lowercase model id, custom_emoji_id, fallback emoji). First match wins.
# Emoji ids come from the public pack https://t.me/addemoji/llm_badges_by_wolfpw_bot; any pack works
# as long as the bot's owner can use it - override via plugin settings ``badges``.
DEFAULT_BADGES = (
    (r"claude", "5302992173296300813", "🎭"),
    (r"gpt|codex|(?<![a-z0-9])o[134](?![a-z0-9])", "5305337002101611535", "🤖"),
    (r"qwen", "5305430374690627622", "🟣"),
)

_FOOTER_RE = re.compile(r"\n\n> !\[[^\]]*\]\(tg://emoji\?id=\d+\)[^\n]*$")
_ACRONYMS = {"gpt": "GPT", "o1": "o1", "o3": "o3", "o4": "o4"}
_EFFORT_LETTER = {"minimal": "L", "low": "L", "medium": "M", "high": "H", "xhigh": "X", "max": "+", "ultra": "+", "auto": "A", "on": "A"}

# session_id -> {"model", "provider", "base_url", "prompt_tokens", "thinking"}
_last_usage: dict = {}
# session_id -> thinking state of the last outgoing request (see thinking_from_request)
_last_thinking: dict = {}
# (model, base_url, provider) -> context length (None = lookup failed)
_ctx_cache: dict = {}
_lock = threading.Lock()
_MAX_SESSIONS = 256

_ctx = None  # PluginContext, set in register()


def _cfg(key, default):
    if _ctx is None:
        return default
    try:
        value = _ctx.get_config(key, default)
    except Exception:
        return default
    return default if value is None else value


# --------------------------------------------------------------------------- rendering helpers

def badge_rows():
    """User rows (settings ``badges``) first, then the built-ins."""
    rows = []
    for row in _cfg("badges", []) or []:
        if isinstance(row, dict) and row.get("match") and row.get("emoji_id"):
            rows.append((str(row["match"]), str(row["emoji_id"]), str(row.get("fallback") or "🤖")))
    rows.extend(DEFAULT_BADGES)
    return rows


def badge_for(model: str):
    m = (model or "").lower()
    for pattern, emoji_id, fallback in badge_rows():
        try:
            if re.search(pattern, m):
                return emoji_id, fallback
        except re.error as exc:
            logger.warning("model_badge: bad regex %r in settings.badges: %s", pattern, exc)
    return None


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v >= 10 or v == int(v) else f"{v:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}k"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def pretty_model(model: str) -> str:
    """``claude-fable-5-1`` -> ``Claude Fable 5.1``; ``qwen/qwen3.8:27b`` -> ``Qwen3.8 27b``."""
    name = (model or "").split("/")[-1].replace(":", "-")
    words = [w for w in name.split("-") if w]
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if w.isdigit() and len(w) == 8:  # date suffix 20250805
            i += 1
            continue
        if w.isdigit() and i + 1 < len(words) and words[i + 1].isdigit() and len(words[i + 1]) < 8:  # version pair 5-1 -> 5.1
            out.append(f"{w}.{words[i + 1]}")
            i += 2
            continue
        out.append(_ACRONYMS.get(w.lower(), w if (w[0].isdigit() or w.isupper()) else w[0].upper() + w[1:]))
        i += 1
    return " ".join(out)


def bar(used: int, ctx: int, width: int = 5) -> str:
    width = max(1, int(width))
    filled = min(width, max(0, round(width * used / ctx)))
    if used > 0 and filled == 0:
        filled = 1
    return "▰" * filled + "▱" * (width - filled)


def _context_length(model: str, base_url: str, provider: str):
    key = (model, base_url or "", provider or "")
    with _lock:
        if key in _ctx_cache:
            return _ctx_cache[key]
    ctx = None
    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(model, base_url=base_url or "", provider=provider or "")
        ctx = int(ctx) if ctx and int(ctx) > 0 else None
    except Exception as exc:
        logger.debug("model_badge: context length lookup failed for %s: %s", model, exc)
    with _lock:
        _ctx_cache[key] = ctx
    return ctx


def thinking_from_request(body: dict | None) -> str | None:
    """What the wire actually carries, normalised to ``off`` / ``<effort>`` / ``budget:Nk`` / None
    (parameter absent = route default). Covers Anthropic (``thinking`` + ``output_config.effort``),
    Responses/Codex (``reasoning.effort``), chat_completions (``reasoning_effort`` or
    ``extra_body.reasoning``)."""
    if not isinstance(body, dict):
        return None
    think = body.get("thinking")
    if isinstance(think, dict):
        t = str(think.get("type", "")).lower()
        if t == "disabled":
            return "off"
        if t == "enabled" and think.get("budget_tokens"):
            return f"budget:{fmt_tokens(int(think['budget_tokens']))}"
        oc = body.get("output_config")
        if isinstance(oc, dict) and oc.get("effort"):
            return str(oc["effort"]).lower()
        return "auto" if t == "adaptive" else None
    r = body.get("reasoning")
    if not isinstance(r, dict):
        eb = body.get("extra_body")
        if isinstance(eb, dict):
            r = eb.get("reasoning")
            if isinstance(eb.get("thinking"), dict) and not isinstance(r, dict):
                return "off" if eb["thinking"].get("type") == "disabled" else "on"
    if isinstance(r, dict):
        if r.get("enabled") is False or str(r.get("effort", "")).lower() == "none":
            return "off"
        if r.get("effort"):
            return str(r["effort"]).lower()
    if body.get("reasoning_effort"):
        e = str(body["reasoning_effort"]).lower()
        return "off" if e == "none" else e
    return None


def thinking_label(state: str | None) -> str:
    if state is None:
        return "🧠A"  # parameter absent: adaptive models decide on their own
    if state == "off":
        return "🚫"
    if state.startswith("budget:"):
        return f"🧠{state[7:]}"
    return "🧠" + _EFFORT_LETTER.get(state, state[:1].upper())


def render_footer(model: str, usage: dict | None) -> str | None:
    """``> ![🎭](tg://emoji?id=N) `Claude Fable 5.1 · 🧠H ▰▱▱▱▱ 4% · 48k/1M``` or None (no badge)."""
    badge = badge_for(model)
    if badge is None:
        return None
    emoji_id, fallback = badge
    info = pretty_model(model)
    if _cfg("show_thinking", True) and usage and "thinking" in usage:
        info += f" · {thinking_label(usage['thinking'])}"
    if _cfg("show_context", True) and usage and usage.get("prompt_tokens"):
        used = int(usage["prompt_tokens"])
        ctx = _context_length(usage.get("model") or model, usage.get("base_url", ""), usage.get("provider", ""))
        if ctx:
            info += f" {bar(used, ctx, _cfg('bar_width', 5))} {100 * used // ctx}% · {fmt_tokens(used)}/{fmt_tokens(ctx)}"
        else:
            info += f" · {fmt_tokens(used)}"
    return f"> ![{fallback}](tg://emoji?id={emoji_id}) `{info}`"


def render_badge(response_text: str, model: str, platform: str, usage: dict | None = None):
    """Return the badged text, or None when nothing should change."""
    if platform not in set(_cfg("platforms", ["telegram"])) or not response_text or not response_text.strip():
        return None
    if _FOOTER_RE.search(response_text):
        return None
    footer = render_footer(model, usage)
    if footer is None:
        return None
    return f"{response_text.rstrip()}\n\n{footer}"


# --------------------------------------------------------------------------- hooks

def _pre_api_request(session_id="", request=None, **_kw):
    if not session_id:
        return None
    body = (request or {}).get("body") if isinstance(request, dict) else None
    with _lock:
        _last_thinking[session_id] = thinking_from_request(body)
    return None


def _post_api_request(session_id="", model="", provider="", base_url="", usage=None, **_kw):
    if not session_id or not usage:
        return None
    with _lock:
        if len(_last_usage) >= _MAX_SESSIONS:
            oldest = next(iter(_last_usage))
            _last_usage.pop(oldest)
            _last_thinking.pop(oldest, None)
        _last_usage[session_id] = {
            "model": model, "provider": provider or "", "base_url": base_url or "",
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "thinking": _last_thinking.get(session_id),
        }
    return None


def _transform_llm_output(response_text="", model="", platform="", session_id="", **_kw):
    try:
        with _lock:
            usage = _last_usage.get(session_id)
        # The usage row belongs to the model that made the last call; prefer it over ``model`` so a
        # mid-turn fallback shows the model that actually answered.
        shown_model = (usage or {}).get("model") or model
        return render_badge(response_text, shown_model, platform, usage)
    except Exception as exc:  # never break the reply
        logger.debug("model_badge failed: %s", exc)
        return None


def register(ctx):
    global _ctx
    _ctx = ctx
    ctx.register_hook("pre_api_request", _pre_api_request)
    ctx.register_hook("post_api_request", _post_api_request)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
