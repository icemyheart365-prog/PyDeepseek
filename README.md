# PyDeepseek

A local chat client for the DeepSeek API. The model never executes anything
directly: it can only emit JSON, and a local Python dispatcher validates each
request against an allowlist before running it.

- **Two languages**, switchable at runtime
- **Named memory branches** with AI-assisted compaction
- **Read-only tool layer**: file commands plus web search and fetch
- **Exact token counting** via a bundled tokenizer (byte estimate when absent)
- Standard library only, apart from the optional `tokenizers` package

## Getting an API key

1. Open <https://platform.deepseek.com/api_keys>
2. Sign in, or register an account
3. Click **Create API key**
4. Copy the string starting with `sk-`

> The key is displayed **only once**. Save it before closing the dialog. If you
> lose it, delete it and create a new one.

Check your balance at <https://platform.deepseek.com/top_up>. A new account may
not have credit, and requests fail without it.

## Quick start

```bash
git clone https://github.com/icemyheart365-prog/PyDeepseek.git
cd PyDeepseek
python chat_v10.py
```

On the first run a short wizard asks for:

1. **Interface language** — 中文 or English
2. **API key** — verified with a one-token request before being saved
3. **Default model** — `deepseek-flash` or `deepseek-v4-pro`

It writes `config.json` and starts the chat. To configure manually instead:

```bash
cp config.example.json config.json
# edit config.json and fill in api_key
python chat_v10.py
```

### Optional: exact token counting

```bash
pip install -r requirements.txt
```

Without it the app still runs, reporting byte-based estimates.

## Usage

Type a message and press Enter. Slash commands:

| Command | Description |
|---|---|
| `/help` | List commands |
| `/settings` | Switch language or model |
| `/memory` | List or switch memory branches |
| `/memory add NAME` | Create a branch |
| `/memory compact` | Let the AI summarize history |
| `/file NAME` | Attach files from `library/` |
| `/files` | List `library/` |
| `/pwd` | Show the current memory branch |
| `/stats` | Cumulative token usage |
| `/system` | View or change the system prompt |
| `/clear` | Wipe history (asks first) |
| `/exit` | Quit |

Command-line flags:

```bash
python chat_v10.py --lang en_ca      # override the saved language
python chat_v10.py --max-history 20  # raw messages sent to the API
```

## Safety model

The model cannot run arbitrary commands. Every tool call passes through:

1. **Argument validation** — unknown parameters and invalid values are rejected
2. **Path containment** — reads stay inside `library/`; symlinks and junctions
   that escape it are refused
3. **Confirmation gates** — commands marked `require_confirmation` ask first
4. **Output limits** — large outputs need explicit approval before transmission

Tool definitions live in `memories/commands.json` (the contract the model sees)
and `memories/commands.impl.json` (implementation details never sent to the
model).

## Layout

```
chat_v10.py               entry point
agent_core.py             all chat logic
models.py                 model catalogue and menu slots
ui/                       terminal rendering and input primitives
agent_dispatcher.py       tool-call validation and execution
command_library_bridge.py JSON envelope over a backend
bridges/                  python (default), powershell, posix
tokenizer/                token counter + bundled vocabulary
i18n/                     UI strings per language
memories/                 command contract and memory branches
library/                  file attachment area
tools/                    developer scripts (notes.ps1)
```

## Platform notes

- **Python 3.10+** required
- On Windows, `tools/notes.ps1` may need
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- The POSIX backend needs a shell; the Python backend works everywhere and is
  the default

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `HTTP 401` | Wrong or expired key | Re-run the wizard or edit `config.json` |
| `401` with a key you believe is valid | Stray quotes or whitespace | Check the value in `config.json` |
| `config.json 缺少有效的 api_key` | Placeholder left in place | Fill in a real key |
| PowerShell refuses to run `.ps1` | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Token counts look rough | `tokenizers` not installed | `pip install -r requirements.txt` |

## License

MIT — see [LICENSE](LICENSE).
