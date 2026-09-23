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
fallback unicode emoji.

Two ways to get ``![..](tg://emoji?id=N)`` through the adapter's MarkdownV2 escaping:
- a core that passes it through (hermes-agent PR "pass MarkdownV2 custom emoji links through
  format_message") - detected at load time, nothing else to do;
- a stock core - ``TelegramAdapter.format_message`` is wrapped once (see ``install_shim``) to
  un-escape exactly that construct after the original formatter ran. Signature-guarded and idempotent;
  if the adapter changes shape the shim steps aside and the footer degrades to a plain link.
"""

import functools
import inspect
import logging
import re
import sys
import threading

logger = logging.getLogger(__name__)

_ADAPTER_MODULES = ("hermes_plugins.platforms__telegram.adapter", "plugins.platforms.telegram.adapter")
_SHIM_MARK = "_model_badge_shim"
_PROBE = "![🎭](tg://emoji?id=1)"
_ESCAPED_EMOJI_RE = re.compile(r"\\!(\[[^\]]*\]\(tg://emoji\?id=\d+\))")

# Built-in rows: (regex on lowercase model id, custom_emoji_id, fallback emoji). First match wins.
# Emoji ids come from the public pack https://t.me/addemoji/llm_badges_by_wolfpw_bot; any pack works
# as long as the bot's owner can use it - override via plugin settings ``badges``.
DEFAULT_BADGES = (
    (r"claude", "5302992173296300813", "🎭"),
    (r"gpt|codex|(?<![a-z0-9])o[134](?![a-z0-9])", "5305337002101611535", "🤖"),
    (r"qwen", "5305430374690627622", "🟣"),
)

_FOOTER_RE = re.compile(r"\n\n(?:> |\|\|)?!\[[^\]]*\]\(tg://emoji\?id=\d+\)[^\n]*$")
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
    """Effort as text. ``thinking_style``: ``letter`` (🧠H), ``word`` (🧠 high), ``icon`` (🧠 only when
    on). Icons come from ``thinking_icon`` / ``thinking_off``."""
    icon = str(_cfg("thinking_icon", "🧠"))
    off = str(_cfg("thinking_off", "🚫"))
    style = str(_cfg("thinking_style", "letter"))
    if state == "off":
        return off
    if style == "icon":
        return icon
    if state is None:
        return f"{icon}A" if style == "letter" else f"{icon} auto"
    if state.startswith("budget:"):
        return f"{icon}{state[7:]}" if style == "letter" else f"{icon} {state[7:]}"
    if style == "word":
        return f"{icon} {state}"
    return icon + _EFFORT_LETTER.get(state, state[:1].upper())


# Named looks. ``template`` fields: {logo} {model} {thinking} {bar} {pct} {used} {ctx} {sep}.
# Empty fields collapse together with the separator next to them (see _fill_template).
PRESETS = {
    "quote":   {"layout": "quote",   "mono": True,  "template": "{logo} {model}{sep}{thinking} {bar} {pct}{sep}{used}/{ctx}"},
    "line":    {"layout": "line",    "mono": False, "template": "{logo} {model}{sep}{thinking}{sep}{bar} {pct}{sep}{used}/{ctx}"},
    "mono":    {"layout": "line",    "mono": True,  "template": "{logo} {model}{sep}{thinking} {bar} {pct}{sep}{used}/{ctx}"},
    "spoiler": {"layout": "spoiler", "mono": True,  "template": "{logo} {model}{sep}{thinking} {bar} {pct}{sep}{used}/{ctx}"},
    "minimal": {"layout": "line",    "mono": False, "template": "{logo} {model}{sep}{thinking}{sep}{pct}"},
    "logo":    {"layout": "line",    "mono": False, "template": "{logo}"},
    "bar":     {"layout": "line",    "mono": True,  "template": "{logo} {bar} {pct}"},
    "full":    {"layout": "quote",   "mono": True,  "template": "{logo} {model} ({raw_model}){sep}{thinking}{sep}{bar} {pct}{sep}{used}/{ctx}"},
}

_LAYOUTS = {
    "line":    lambda body: body,
    "quote":   lambda body: f"> {body}",
    "spoiler": lambda body: f"||{body}||",
}


def _look():
    """Effective look = preset, then per-key overrides from settings (``layout``, ``mono``, ``template``)."""
    preset = PRESETS.get(str(_cfg("preset", "quote")), PRESETS["quote"])
    look = dict(preset)
    for key in ("layout", "mono", "template"):
        value = _cfg(key, None)
        if value not in (None, ""):
            look[key] = value
    return look


def _fill_template(template: str, fields: dict, sep: str) -> str:
    """Substitute fields; drop empty ones and collapse the separators/spaces around them."""
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value or "")
    out = out.replace("{sep}", sep)
    # collapse "a · · b", "a ·  " and stray "()" / "·" at the ends produced by empty fields
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s*/\s*(?=\s|$)", "", out) if not fields.get("ctx") else out
    sep_re = re.escape(sep.strip()) if sep.strip() else None
    if sep_re:
        out = re.sub(rf"(?:\s*{sep_re}\s*){{2,}}", sep, out)
        out = re.sub(rf"^\s*{sep_re}\s*|\s*{sep_re}\s*$", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


def render_footer(model: str, usage: dict | None) -> str | None:
    """Footer line per the configured look, or None when the model has no badge."""
    badge = badge_for(model)
    if badge is None:
        return None
    emoji_id, fallback = badge
    look = _look()
    sep = str(_cfg("separator", " · "))
    fields = {
        "logo": "\x00LOGO\x00",  # placeholder: the logo must stay outside the mono span
        "model": pretty_model(model) if str(_cfg("model_name", "pretty")) == "pretty" else (model or ""),
        "raw_model": model or "",
        "thinking": "", "bar": "", "pct": "", "used": "", "ctx": "",
    }
    if _cfg("show_thinking", True) and usage and "thinking" in usage:
        fields["thinking"] = thinking_label(usage["thinking"])
    if _cfg("show_context", True) and usage and usage.get("prompt_tokens"):
        used = int(usage["prompt_tokens"])
        fields["used"] = fmt_tokens(used)
        ctx = _context_length(usage.get("model") or model, usage.get("base_url", ""), usage.get("provider", ""))
        if ctx:
            fields["bar"] = bar(used, ctx, _cfg("bar_width", 5))
            fields["pct"] = f"{100 * used // ctx}%"
            fields["ctx"] = fmt_tokens(ctx)
    body = _fill_template(str(look["template"]), fields, sep)
    logo = f"![{fallback}](tg://emoji?id={emoji_id})"
    if look.get("mono"):
        # Mono wraps everything except the logo (custom emoji do not render inside code spans).
        head, _, tail = body.partition("\x00LOGO\x00")
        parts = [head.strip(), logo if _ else "", f"`{tail.strip()}`" if tail.strip() else ""]
        body = " ".join(p for p in parts if p)
    else:
        body = body.replace("\x00LOGO\x00", logo)
    return _LAYOUTS.get(str(look.get("layout")), _LAYOUTS["line"])(body.strip())


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
    if ensure_custom_emoji_passthrough() is None:
        # The adapter module may not be imported yet at plugin-load time; retry on session start.
        ctx.register_hook("on_session_start", lambda **_: ensure_custom_emoji_passthrough())


# --------------------------------------------------------------------------- adapter compatibility

def _adapter_class():
    for mod_name in _ADAPTER_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = __import__(mod_name, fromlist=["TelegramAdapter"])
            except Exception:
                continue
        cls = getattr(mod, "TelegramAdapter", None)
        if cls is not None:
            return cls
    return None


def core_passes_custom_emoji(cls) -> bool:
    """True when the core's formatter already leaves ``![..](tg://emoji?id=N)`` unescaped."""
    try:
        return "\\" + _PROBE not in cls.format_message(None, _PROBE)
    except Exception:
        return False


def _shim_format(original):
    @functools.wraps(original)
    def format_message(self, content, *args, **kwargs):
        out = original(self, content, *args, **kwargs)
        return _ESCAPED_EMOJI_RE.sub(r"!\1", out) if isinstance(out, str) else out
    format_message.__dict__[_SHIM_MARK] = True
    return format_message


def install_shim(cls=None) -> bool:
    """Wrap ``TelegramAdapter.format_message`` once. True when installed (or already present)."""
    cls = cls or _adapter_class()
    if cls is None:
        return False
    original = getattr(cls, "format_message", None)
    if original is None:
        return False
    if getattr(original, _SHIM_MARK, False):
        return True
    params = list(inspect.signature(original).parameters)
    if params[:2] != ["self", "content"]:
        logger.warning("TelegramAdapter.format_message signature changed (%s); custom emoji badge disabled", params)
        return False
    cls.format_message = _shim_format(original)
    logger.info("model_badge: custom emoji passthrough shim installed on TelegramAdapter.format_message")
    return True


def ensure_custom_emoji_passthrough():
    """None = adapter not importable yet; True = core handles it or shim installed; False = gave up."""
    cls = _adapter_class()
    if cls is None:
        return None
    if core_passes_custom_emoji(cls):
        return True
    return install_shim(cls)
