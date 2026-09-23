# hermes-plugin-model-badge

A footer under every Telegram reply from [Hermes Agent](https://github.com/NousResearch/hermes-agent):
provider logo (custom emoji), model name, reasoning effort and a context-usage bar.

```
> 🎭 Claude Fable 5.1 · 🧠H ▰▱▱▱▱ 4% · 48k/1M
> 🤖 GPT 6 Sol · 🧠M ▰▰▱▱▱ 44% · 120k/272k
> 🟣 Qwen3.8 27b · 🚫 ▰▱▱▱▱ 4% · 5.3k/131k
```

- **Logo** - Telegram custom emoji (`![🎭](tg://emoji?id=N)`), picked by regex on the model id.
  Built-in rows: Claude, OpenAI (gpt/codex/o1/o3/o4), Qwen from the public pack
  [llm_badges_by_wolfpw_bot](https://t.me/addemoji/llm_badges_by_wolfpw_bot). Add your own in settings.
- **Model** - human-readable (`claude-fable-5-1` -> `Claude Fable 5.1`), taken from the *last API call*
  of the turn, so a fallback-chain switch shows the model that actually answered.
- **🧠 effort** - what went out on the wire, not the config: `🧠L/M/H/X`, `🧠+` (max), `🧠A` (parameter
  absent - adaptive model decides), `🧠8.0k` (legacy budget), `🚫` (thinking disabled).
  Anthropic, OpenAI Responses/Codex and chat_completions request shapes are recognised.
- **Bar** - `prompt_tokens` of the last call vs the model's context window (same resolver as `/model`).

Only `platform == "telegram"` is touched; subagents, cron, CLI and other platforms get their text unchanged.

## Requirements

- Any current Hermes Agent. Two modes, chosen automatically at load:
  - **core passthrough** - a core that leaves `![..](tg://emoji?id=N)` unescaped in
    `TelegramAdapter.format_message` (upstream PR
    [NousResearch/hermes-agent#119945](https://github.com/NousResearch/hermes-agent/pull/119945)); the
    plugin detects it and does nothing else;
  - **shim** - on a stock core the plugin wraps `TelegramAdapter.format_message` once and un-escapes exactly
    that construct after the original formatter ran. Signature-guarded (refuses if the method shape
    changes) and idempotent; on refusal the footer still appears, with the logo as a plain link.
- The bot owner needs Telegram Premium for custom emoji to render; otherwise the fallback unicode emoji
  in the brackets is shown.

## Install

```bash
hermes plugins install wwwolf21/hermes-plugin-model-badge --enable
```

Lives outside the core tree, so `hermes update` never touches it. To move to a newer plugin commit:
`hermes plugins install https://github.com/wwwolf21/hermes-plugin-model-badge.git --force --ref <sha> --enable`
(the install is pinned to an exact commit; hooks hot-reload into a running gateway).

## Settings (`~/.hermes/config.yaml`)

```yaml
plugins:
  entries:
    model_badge:
      settings:
        platforms: [telegram]      # where to append the footer
        preset: quote              # quote | line | mono | spoiler | minimal | logo | bar | full
        show_thinking: true
        show_context: true
        bar_width: 5
        badges:                    # checked before the built-ins, first match wins
          - match: "gemini"
            emoji_id: "5300000000000000000"
            fallback: "✨"
```

`emoji_id` is a `custom_emoji_id` from any pack the bot owner can use (`getStickerSet` in the Bot API
returns them). Settings are also editable in the Desktop app, Plugins tab.

### Looks

| preset | renders as |
|---|---|
| `quote` (default) | `> 🎭 `Claude Fable 5.1 · 🧠H ▰▱▱▱▱ 4% · 48k/1M`` - blockquote, mono |
| `line` | `🎭 Claude Fable 5.1 · 🧠H · ▰▱▱▱▱ 4% · 48k/1M` - plain text |
| `mono` | same as quote without the blockquote bar |
| `spoiler` | `\|\|🎭 `…`\|\|` - hidden until tapped |
| `minimal` | `🎭 Claude Fable 5.1 · 🧠H · 4%` |
| `logo` | `🎭` |
| `bar` | `🎭 `▰▱▱▱▱ 4%`` |
| `full` | quote + raw model id in parentheses |

Fine-tuning on top of a preset (each key optional):

```yaml
        layout: spoiler            # line | quote | spoiler
        mono: true
        template: "{logo} {model}{sep}{thinking}{sep}{pct}"
        separator: " | "
        model_name: raw            # pretty | raw
        thinking_style: word       # letter (🧠H) | word (🧠 high) | icon (🧠)
        thinking_icon: "💭"
        thinking_off: "💤"
```

Template fields: `{logo} {model} {raw_model} {thinking} {bar} {pct} {used} {ctx} {sep}`. Fields that are
empty (no usage yet, `show_*: false`, unknown context window) disappear together with the separator
next to them. With `mono` the logo is placed before the code span - Telegram does not render custom emoji
inside code.

## How it works

| Hook | Role |
|---|---|
| `pre_api_request` | records the thinking setting of the outgoing request per session |
| `post_api_request` | records model / provider / `prompt_tokens` of the response per session |
| `transform_llm_output` | once per turn: renders the footer from the last call and appends it |

State is an in-memory map bounded to 256 sessions.

## Tests

```bash
python -m pytest tests
hermes plugins validate .
```

MIT
