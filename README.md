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

- Hermes Agent with the custom-emoji passthrough in the Telegram adapter
  (upstream PR *"feat(telegram): pass MarkdownV2 custom emoji links through format_message"*).
  Without it the `!` is escaped and the logo degrades to a plain link.
- The bot owner needs Telegram Premium for custom emoji to render; otherwise the fallback unicode emoji
  in the brackets is shown.

## Install

```bash
hermes plugins install wwwolf21/hermes-plugin-model-badge --enable
hermes gateway restart
```

Or clone into `~/.hermes/plugins/model_badge/` and `hermes plugins enable model_badge`.
Lives outside the core tree, so `hermes update` never touches it.

## Settings (`~/.hermes/config.yaml`)

```yaml
plugins:
  entries:
    model_badge:
      settings:
        platforms: [telegram]      # where to append the footer
        show_thinking: true
        show_context: true
        bar_width: 5
        badges:                    # checked before the built-ins, first match wins
          - match: "gemini"
            emoji_id: "5300000000000000000"
            fallback: "✨"
```

`emoji_id` is a `custom_emoji_id` from any pack the bot owner can use (`getStickerSet` in the Bot API
returns them).

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
