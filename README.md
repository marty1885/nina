# Nina

An AI assistant that lives in your chat. Built in Rust.

Nina connects to Telegram, talks through configurable LLM backends, remembers things long-term with vector search, and can act on your behalf — running commands, browsing the web, setting reminders, and more.

Nina is just the system's name. Not nessarrily the agent.

## Features

- Persistent semantic memory with vector embeddings
- Agentic tool use (shell, web, files, reminders, Lua, TTS, and more)
- Concurrent message handling with branch merging
- Autonomous self-awareness loop — she can think about you and reach out
- Weighted multi-provider LLM routing with automatic fallback
- User pairing and access control
- Extensible through markdown-based skill plugins

## Prerequisites

- Rust (2021 edition)
- [SQLite vec0 extension](https://github.com/asg017/sqlite-vec) installed at `/usr/lib/vec0.so`
- A Telegram bot token
- At least one LLM provider API key

## Quick start

```bash
# Build
cargo build --release

# Interactive setup — generates personality, identity, and user context
nina setup

# Run
nina
```

`nina setup` walks you through naming your agent, describing its personality, and telling it about yourself. It uses your configured LLM to generate a `soul.md` (persona prompt), `identity.toml`, and `context/USER.md` automatically.

To configure manually instead, copy the example config and edit it:

```bash
mkdir -p ~/.nina
cp nina.toml.example ~/.nina/nina.toml
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NINA_HOME` | No | Config & data directory (default `~/.nina`) |
| `NINA_WORKSPACE` | Yes | Working directory for shell/file operations |
| `TELEGRAM_BOT_TOKEN` | Yes | From [@BotFather](https://t.me/BotFather) |

Provider API keys (e.g. `DEEPINFRA_API_KEY`, `ANTHROPIC_API_KEY`) are referenced by name in `nina.toml`.

### Configuration

See [`nina.toml.example`](nina.toml.example) for a full example. The key sections:

- **`[agent]`** — Primary and fallback LLM providers with weights
- **`[gateway]`** — Lightweight LLM for message classification
- **`[[providers]]`** — Provider definitions (any OpenAI-compatible API)
- **`[tts]`** — Optional text-to-speech endpoint

## User pairing

New users are gated behind an approval system:

```bash
nina pair list                        # Show pending requests
nina pair approve <code>              # Approve as member
nina pair approve <code> --owner      # Approve as owner
nina pair reject <code>               # Reject
```

## Built-in skills

Shell, web search & fetch, file operations, semantic memory, reminders (one-time & cron), text-to-speech, Lua scripting, cross-session messaging, and timezone-aware time.

Custom skills can be added as markdown files in `~/.nina/skills/` — see the `skills/` directory for examples.

## Data

Everything is stored in a single SQLite database at `~/.nina/data/nina.db` — conversations, memories, reminders, identities.

## License

See [LICENSE](LICENSE) for details.
