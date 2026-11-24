# Privacy policy

AI-llm-Engine and LLMBar run language models on your Mac. This page lists what stays on the Mac, what is kept
there, and the only times the software connects to the internet.

## In short

- Your prompts, replies, files and images are processed on your Mac. They are not sent to the authors or to
  anyone else, unless you turn on one of the features listed under "When it goes online".
- There is no telemetry, analytics, crash reporting or account.
- Nothing you type is used to train or fine-tune any model. The software has no way to send it for that.
- The engine's server listens on `127.0.0.1` (this Mac only), unless you start it with another `--host`.

## What is kept on your Mac

| What | Where | How long |
|---|---|---|
| Terminal chats (messages, titles, project folder) | `~/Library/Application Support/LLMBar/sessions` | Deleted after 30 days, or beyond the newest 100 chats; `/delete` removes one at once |
| Prompt cache: the model's internal state for recent prompts, so a repeated prompt starts faster | `~/.cache/ai-llm-engine/prefix` (or the folder you set) | Up to the size you set (8 GB by default); set `disk_cache_gb` to `0` to turn it off |
| Which experts of a mixture-of-experts model were used (counts only, no text) | `~/.cache/ai-llm-engine/moe` | Until you delete it |
| Models you download | `~/.cache/huggingface/hub` | Until you delete them |
| Settings | `~/Library/Application Support/LLMBar` (the project folder when run from source) | Until you delete them |
| Google sign-in, if you connect Google | macOS Keychain | Until you disconnect |

All of it can be deleted by removing those folders.

## When it goes online

The software connects only for these features, and only to the services named:

| Feature | Connects to | What is sent |
|---|---|---|
| Downloading a model | Hugging Face (or hf-mirror.com, ModelScope if you choose them) | The name of the model |
| Setup | PyPI, GitHub (Python and llama.cpp builds) | Package and file names |
| Update check (LLMBar, at most every 6 hours; `./llm update`) | GitHub's releases API | An ordinary request with no personal data |
| Web research (when you or the model use the web tools) | DuckDuckGo, arXiv, OpenAlex, Stack Exchange, Hacker News search, GitHub, and the pages the search returns | The search query and the page addresses |
| Cloud models (only when you add an API key) | Anthropic or OpenRouter | The conversation you send to them, under their privacy policies |
| MCP servers you add (for example Context7) | That server | What the model asks that server |
| Google Workspace (only if you connect it) | Google's APIs | What you ask the assistant to read or write |

The services above have their own privacy policies. Nothing else is sent, and none of it goes to the authors.

## Contact

Questions about this policy: open an issue at https://github.com/dsavenk0/AI-llm-Engine/issues.
