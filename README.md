# Signal Stock Bot

A self-hosted Signal bot for real-time stock quotes, market data, technical analysis, and company fundamentals — with a built-in admin web UI, LLM integration (any OpenAI-compatible provider), MCP server support, and per-chat policy controls.

## Features

### Markets
- **Real-time stock quotes** via Yahoo Finance, Finnhub, Twelve Data, Alpha Vantage, and Polygon
- **Multi-provider failover** — automatic fallback when rate limited
- **Smart symbol resolution** — type `!price apple` or `!price gold` instead of tickers
- **Professional charting** with candlesticks, indicators, and comparisons
- **Technical analysis** with RSI, MACD, SMA, support/resistance
- **Earnings, dividends, options, futures, forex, crypto, and economic indicators**
- **Price alerts** with per-chat notifications

### Bot UX
- **Command chaining** — run multiple commands at once: `!price AAPL !tldr AAPL !news AAPL`
- **Universal `-help` flag** — add to any command for detailed explanations
- **Batch symbol lookups** and **inline symbol detection** (`$AAPL`)
- **@mention support** for natural-language queries
- **Intelligent caching** with type-specific TTLs

### Admin & Intelligence
- **Admin web UI** at `/admin` — login + Signal-delivered 2FA, dashboard, predictions console, per-context editor, live event feed; live edits without restart
- **LLM integration** — `!ask` with rolling per-context conversation history, OpenAI-compatible (works with OpenAI, OpenRouter, Groq, Anthropic-via-OR, Ollama, llama.cpp, etc.); always injects current UTC time + persona; supports OpenRouter provider pinning (preferred providers, no-fallback, sort axis) so latency-sensitive deployments can keep traffic on Cerebras/Groq
- **Deep-think tool** — separate slower, smarter model exposed to the writer LLM via a `deep_think(question, context)` tool with its own toolkit, status messaging, and per-context budget caps
- **MCP servers** — register and manage Model Context Protocol servers (stdio / SSE / HTTP) via the admin UI. Pyodide Python sandbox auto-registered + pre-warmed at boot so Sigil can run real numpy/pandas/scipy/yfinance computations on demand. `npx`, `uvx`, `git` available in the container
- **LLM tool calling** — bot commands and MCP tools exposed as function calls; chart and tarot attachments bubble back into the chat
- **Per-context policies** — scope which commands and MCP servers work in each group/DM; assign per-chat system prompts; toggle reactor / natural-response / deep-think / memory writes; configure per-context daily oracles
- **Per-context memory store** — `remember`/`recall`/`forget` LLM tools with subject resolution (registered names, "yourself", "this chat", or free-text), promotion-by-corroboration, cross-kind dedup so the reactor doesn't flood the same fact under different `kind`s
- **Speaker attribution in groups** — every user turn rendered as `[Name, time ago]` and assistant turns as `[to Name, time ago]` so the model can pair questions with answers across interleaved speakers; explicit `<attribution_rules>` block + leak-stripping post-processor so the bracket scaffolding never bleeds into the visible reply
- **Emoji reactor + natural response** — fire-and-forget background LLM picks emoji reactions; optionally also decides when to chime in even without a mention; per-context toggleable, rate-limited
- **Twitter / X URL expansion** — pasted tweet links auto-resolve to text via fxtwitter so the LLM sees what users shared
- **Predictions game** — `!predict <claim> by <date>`, multi-user consensus `!resolve`, leaderboard, auto-resolver (cron + Sigil with research tools); LLM tools `predict_self` / `predict_for(subject)` / `predict_update(id)` so the bot can stake its own forecasts, log on behalf of a chat member, or fix mis-extracted claims within a 15-min grace window
- **Per-context daily oracles** — each group can have any number of scheduled posts (tarot draw / I Ching cast / pre-market check / closing recap / freeform). Schedules are sunrise/sunset (NYC anchor) ± offset minutes or fixed clock time in any IANA timezone, with optional weekdays-only filter. Heuristic prepopulation seeds sensible defaults based on context labels
- **Divination** — `!tarot` (Rider-Waite-Smith deck), `!iching` (King-Wen 64 hexagrams), `!numerology` (Pythagorean — life path / personal year / expression / soul urge / personality, master numbers preserved)

---

## Quick Start

### Prerequisites

- Ubuntu 22.04+ server (or any Docker-capable host)
- Docker and Docker Compose
- A phone number for Signal

### 1. Clone and configure

```bash
git clone https://github.com/davidtorcivia/signal-stock-bot.git
cd signal-stock-bot
cp .env.example .env
nano .env  # Set your phone number
```

### 2. Start the stack

```bash
docker compose up -d
```

### 3. Link your Signal account

```bash
curl -s "http://localhost:8080/v1/qrcodelink?device_name=stockbot" | docker run -i --rm mtgto/qrencode -t ANSIUTF8
```

On your phone: **Signal → Settings → Linked Devices → Link New Device → Scan QR**

### 4. Test it

Send `!price AAPL` to your Signal number.

---

## Commands Reference

### Price & Quote Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `!price AAPL` | `!p` | Current price (supports batch: `!price AAPL MSFT GOOGL`) |
| `!quote AAPL` | `!q` | Detailed quote with OHLC, volume, market cap |
| `!info AAPL` | `!i`, `!fund` | Company fundamentals (P/E, EPS, 52W range) |

**Smart symbol resolution** — use company names instead of tickers:
```
!price apple               → AAPL
!price microsoft           → MSFT  
!price bitcoin             → BTC-USD
!price gold                → GC=F
!price oil                 → CL=F
!price 10 year treasury    → ^TNX
```

**Batch mode** — up to 10 symbols:
```
!price AAPL MSFT GOOGL NVDA
```

---

### Chart Commands

```
!chart AAPL [period] [options]
```

**Periods**: `1d`, `5d`, `1w`, `1m`, `3m`, `6m`, `1y`, `ytd`, `5y`, `max`

**Options**:
| Flag | Description |
|------|-------------|
| `-c` | Candlestick chart (default is line) |
| `-sma20`, `-sma50`, `-sma200` | Add SMA overlays |
| `-bb` | Add Bollinger Bands |
| `-rsi` | Add RSI panel below chart |
| `-compare MSFT` | Overlay another symbol for comparison |

**Examples**:
```
!chart AAPL 1m                        # 1-month line chart
!chart NVDA 3m -c                     # 3-month candlestick
!chart TSLA 1y -sma50 -sma200         # With moving averages
!chart AAPL 6m -c -bb -rsi            # Full technical chart
!chart AAPL 1m -compare MSFT          # Compare AAPL vs MSFT
```

---

### Technical Analysis Commands

| Command | Description |
|---------|-------------|
| `!ta AAPL` | Quick technical summary (trend, RSI, MACD, S/R, signal) |
| `!ta AAPL -full` | **Comprehensive analysis** with all indicators |
| `!tldr AAPL` | **Simple verdict**: Buy, Sell, or Hold |
| `!rsi AAPL` | RSI(14) with visual bar and interpretation |
| `!sma AAPL 20 50 200` | Moving averages with % difference from price |
| `!macd AAPL` | MACD line, signal, histogram, momentum |
| `!support AAPL` | Pivot-based support/resistance levels (S1, S2, R1, R2) |

**Example `!ta AAPL -full` output**:
```
⊞ AAPL Full Technical Analysis

━━━ Price & Trend ━━━
Current: $185.92
Trend: ▲ Bullish (above 50/200 SMA)

━━━ Moving Averages ━━━
SMA20: $183.50 (▲ +1.3%)
SMA50: $178.25 (▲ +4.3%)
SMA200: $165.00 (▲ +12.7%)

━━━ Oscillators ━━━
RSI(14): 62.3 [██████████░░░░░]
  → Moderately High
MACD: Bullish ▲
  Line: 2.450 | Signal: 1.890
  Histogram: 0.560 (Increasing ↑)

━━━ Support/Resistance ━━━
R2: $195.50
R1: $190.00
Pivot: $185.25
S1: $180.50
S2: $175.00

━━━ Signal ━━━
● BUY (3/4 bullish)
```

---

### Earnings & Dividend Commands

| Command | Description |
|---------|-------------|
| `!earnings AAPL` | Next earnings date, EPS, P/E, revenue, margins |
| `!dividend AAPL` | Yield, annual rate, ex-date, payout ratio, history |

---

### News Command

```
!news AAPL [count]
```

| Example | Result |
|---------|--------|
| `!news AAPL` | 5 recent headlines |
| `!news AAPL 10` | 10 headlines |
| `!news` | Market-wide news (SPY) |

---

### Watchlist Commands

Persistent per-user watchlist for tracking your favorite symbols:

```
!watch                     # View your watchlist with live prices
!watch add AAPL MSFT       # Add symbols
!watch remove TSLA         # Remove a symbol
!watch clear               # Clear entire watchlist
```

Watchlists are stored locally and persist across sessions. Limit: 50 symbols per user.

---

### Market Overview Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `!market` | `!m` | Major indices (S&P, Dow, Nasdaq, Russell, VIX) |
| `!crypto` | `!c` | Top cryptocurrencies |
| `!forex EURUSD` | `!fx` | Currency pairs |
| `!future CL` | `!fut` | Futures quotes |
| `!economy CPI` | `!eco` | Economic indicators (free via FRED) |
| `!options AAPL` | `!opt` | Options chains (requires Polygon Pro) |

---

### Economy Commands

Get economic indicators from FRED (Federal Reserve Economic Data). Free with 120 requests/min.

```
!eco [indicator]
```

| Indicator | Description |
|-----------|-------------|
| `CPI` | Consumer Price Index |
| `UNEMPLOYMENT` | Unemployment Rate |
| `GDP` | Gross Domestic Product |
| `FEDFUNDS` | Federal Funds Rate |
| `DEBT` | Federal Debt |
| `JOBS` | Nonfarm Payrolls |
| `10Y` / `2Y` / `30Y` | Treasury Rates |
| `RETAIL` | Retail Sales |
| `HOUSING` | Housing Starts |
| `MORTGAGE` | 30-Year Mortgage Rate |
| `INFLATION` | Inflation Rate |
| `CONSUMER` | Consumer Sentiment |

**Examples**:
```
!eco CPI          # Latest CPI reading
!eco UNEMPLOYMENT # Current unemployment rate
!eco FEDFUNDS     # Federal Funds Rate
!eco 10Y          # 10-Year Treasury yield
```

**Charts**:
Add `chart` or a time period (`1y`, `5y`, `max`) to see a trend graph:
```
!eco CPI chart    # 5-year trend (default)
!eco GDP 10y      # 10-year growth chart
!eco JOBS max     # All-time history
```

Requires `FRED_API_KEY` - get free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

---

### Command Chaining

Run multiple commands in a single message:

```
!price AAPL !tldr AAPL !news AAPL 3
```

Results are separated by a visual divider. Great for quick research!

---

### Universal Help Flag

Add `-help` to any command for a detailed, educational explanation:

```
!ta -help              # Basic TA explanation
!ta -full -help        # Detailed breakdown of all indicators
!chart -c -help        # Candlestick and indicator explanations
!rsi -help             # RSI interpretation guide
!macd -help            # MACD signal reading
```

Help text explains:
- What each metric means
- How to interpret values
- Trading signals to watch for
- Pro tips for retail investors

---

### LLM Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `!ask <question>` | `!a` | Send a question to the configured LLM with bot + MCP tools available |
| `!ask reset` | — | Clear conversation history for the current chat |

The `!ask` alias is editable from `/admin/llm` (default `ask`, but `!ai`, `!sigil`, etc. all work the same). See [LLM Integration](#llm-integration).

### Prediction Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `!predict <claim> by <when>` | `!bet`, `!forecast` | Log a dated prediction. Stock-shape claims auto-resolve via live price; freeform claims get LLM-judged at the deadline |
| `!predictions [@user]` | `!preds`, `!mybets` | List open predictions (yours by default) |
| `!resolve <id> right\|wrong\|unclear [reason]` | `!verdict` | Cast a resolution vote (see [Predictions Game](#predictions-game) for the consensus rules) |
| `!leaderboard` | `!lb`, `!scores` | Per-chat accuracy ranking |

Sigil can also drive the prediction store itself via the `predict_self` / `predict_for` / `predict_update` LLM tools — see [Predictions Game](#predictions-game).

### Divination Commands

Two image-attachment commands that draw from canonical decks/hexagrams and (optionally) get an LLM-narrated reading on top. Both also work via natural language when `llm_intent` is enabled for the context — the writing LLM receives a directive that *forces* it through the tool, so it can't fabricate cards or hexagrams in plain text.

#### Tarot

| Command | Description |
|---------|-------------|
| `!tarot` | Single random card |
| `!tarot 3 [question]` | Three-card past / present / future spread |
| `!tarot celtic [question]` | Ten-card Celtic Cross |
| `!tarot daily` | Card of the day, cached per user × UTC date (same card all day) |

Aliases: `!cards`, `!card`. Deck: Rider-Waite-Smith, downloaded once from Wikimedia Commons on first start (~90s) and cached in the persistent `data/tarot/` volume. Spreads are composed by `tarot_composer` into a single PNG attachment.

#### I Ching

| Command | Description |
|---------|-------------|
| `!iching [question]` | Three-coin cast (default) — `1/8, 3/8, 3/8, 1/8` distribution over `{6,7,8,9}` |
| `!iching yarrow [question]` | Yarrow-stalk simulation — traditional `1/16, 5/16, 7/16, 3/16` distribution |
| `!iching daily` | Hexagram of the day, cached per user × UTC date (same cast all day) |

Aliases: `!ic`, `!yi`, `!yijing`. The hexagram(s) are rendered procedurally onto a parchment canvas — no image assets required. Each render shows: the Chinese name in serif CJK, pinyin · English title, the trigram pair (with procedurally-drawn mini-glyphs so we don't depend on Unicode trigram font coverage), the six-line hexagram in deep ink, a cinnabar seal in the corner with the hexagram number in Chinese numerals (e.g. `二十七` for hex 27), and keywords. Changing lines (`6` and `9`) are highlighted in cinnabar with `○` (yang→yin) or `×` (yin→yang) markers; if any are present, a transformed hexagram is rendered alongside the primary with the character `變` (*biàn* — change) between them.

The casting itself uses a `random.SystemRandom` (cryptographically strong); see [`iching_command.py`](src/commands/iching_command.py) for the per-line generators.

#### Numerology

| Command | Description |
|---------|-------------|
| `!numerology <birthdate>` | Date-derived numbers only (life path, birthday, personal year, personal day) |
| `!numerology <birthdate> <full name>` | Adds expression / soul urge / personality |
| `!numerology <full name>` | Name-only |

Aliases: `!numbers`, `!num`. Pythagorean letter values; Y treated as consonant; **master numbers (11, 22, 33) preserved at every step** rather than reduced to 2/4/6. Pure-Python lookup tables — no MCP or external API.

### Admin Commands (Signal-side)

| Command | Description |
|---------|-------------|
| `!admin backup` | Export all watchlists as JSON (DM only) |
| `!admin alerts` | List active alerts system-wide |
| `!admin users` | User activity stats |
| `!metrics` | System health, uptime, request rates |
| `!cache stats` | View cache hit rates |
| `!cache clear` | Flush all caches |

Restricted to numbers in `ADMIN_NUMBERS`. The full admin surface lives in the [web UI](#admin-web-ui).

---

## Natural Language

The bot understands conversational language, context, and complex queries.

### Context Awareness
- **"Chart Apple"** → *Shows AAPL chart*
- **"What is it trading at?"** → *Remembers AAPL context → Shows Price*
- **"Show its RSI"** → *Remembers AAPL context → Shows RSI*

### Smart Matching
- **Typos**: "Price of **Nvidea**" → *Corrects to NVIDIA (NVDA)*
- **Lowercase**: "chart apple" → *Understands lowercase tickers safely*

### Advanced Queries
- **Timeframes**: "Chart TSLA for **6 months**", "Since 2023", "Last 30 days"
- **Multi-Intent**: "Chart Apple **and** show me the RSI" (Splits into two commands)
- **Comparisons**: "Chart Apple **vs** Microsoft" or "Compare AAPL to TSLA"
- **Sentiment**: "Is Apple a **buy**?", "Should I sell Tesla?" (Analyst ratings)
- **Parameters**: "Give me the RSI for AAPL"

### Examples

| **You say** | **Bot does** |
|:---|:---|
| "Chart Apple" | `!chart AAPL` |
| "What's the price of Tesla?" | `!price TSLA` |
| "Any news on Google?" | `!news GOOGL` |
| "Is Microsoft a buy?" | `!rating MSFT` (Sentiment) |
| "Chart Apple vs Tesla" | `!chart AAPL -compare TSLA` |
| "Chart it for 6 months" | `!chart [LastSymbol] 6m` |
| "Price of Nvidea" | `!price NVDA` (Typo fix) |

---

## Pro Features

### Price Alerts
Notify you when stocks hit specific targets. Alerts trigger in the same chat (DM or Group) where they were set.

`!alert AAPL above 200`
`!alert TSLA below 150`
`!alert BTC change 5` (notify on 5% move)

- `!alerts` - List active alerts
- `!alert remove [ID]` - Delete an alert
- `!alert clear` - Delete all alerts

### Advanced Analytics
- `!rating [SYMBOL]` - Analyst consensus & price targets
- `!insider [SYMBOL]` - Recent insider buying/selling
- `!short [SYMBOL]` - Short interest data & squeeze risk
- `!corr [SYM1] [SYM2]` - 30-day price correlation

---

## Admin Web UI

The bot ships an admin interface at `/admin` for live configuration of every component. It auto-mounts when both `ADMIN_PASSWORD_HASH` and `FLASK_SECRET_KEY` are set (and refuses to start otherwise — no accidentally exposing an unauthenticated admin).

### One-time setup

```bash
# Inside the project dir on the host
python -m src.admin_setup
```

Prompts for a password (≥12 chars), bcrypts it, writes the hash to `./admin_password.hash` (mode `0600`, gitignored), and prints a fresh `FLASK_SECRET_KEY` to paste into `.env`.

The hash file is bind-mounted read-only into the container at `/app/admin_password.hash`. We use a file (not an env var) so docker-compose's variable interpolation doesn't mangle the `$` characters in bcrypt output.

### Login flow

1. Browse to **http://&lt;host&gt;:5000/admin/login**
2. Enter the password
3. The bot DMs a 6-digit code to the **first** number in `ADMIN_NUMBERS` via Signal
4. Enter the code → session is established (12-hour lifetime)

Security:
- bcrypt password (12 rounds), Signal-delivered 2FA code (single-use, 5-min TTL, max 5 attempts), 5 attempts / 15 min per-IP login limit
- CSRF tokens on every form
- Session cookie is `HttpOnly` + `SameSite=Lax` (set `SESSION_COOKIE_SECURE=true` when behind TLS)

### Network exposure

By default `docker-compose.yml` publishes the admin UI on `127.0.0.1:5000`. Override with `ADMIN_BIND_HOST` in `.env`:

```bash
ADMIN_BIND_HOST=0.0.0.0      # Reachable on all interfaces (LAN + anything routed)
ADMIN_BIND_HOST=192.168.1.5  # Lock to a specific LAN interface
ADMIN_BIND_HOST=127.0.0.1    # Loopback only (default)
```

For internet exposure, put Caddy/nginx in front for TLS and set `SESSION_COOKIE_SECURE=true`.

### Pages

| Page | Purpose |
|------|---------|
| `/admin/` | Dashboard — uptime, requests/min, cache stats, provider health, deep-think usage, reactor stats, DB size, "Clear all caches" |
| `/admin/settings` | Live-editable: bot name, rate limit, message length cap, admin numbers, webhook secret. Restart-required: command prefix, provider API keys |
| `/admin/llm` | LLM provider config (incl. OpenRouter provider routing), reactor + natural-response, deep-think model, augmentation, tool-round cap |
| `/admin/mcp` | Add / start / stop / delete MCP servers, view discovered tools (Pyodide auto-registered) |
| `/admin/contexts` | Per-chat policy: command/MCP allow/deny, system prompt, LLM intent toggle, reactor / natural-response / deep-think / memory-writes flags |
| `/admin/contexts/<id>` | Edit one context **and** manage its daily oracles (kind, schedule, offset/clock, weekdays-only, prompt) |
| `/admin/contexts/<id>/memories` | Browse / add / edit / delete the per-context memory store rows |
| `/admin/predictions` | Aggregate counters, per-context leaderboards, upcoming deadlines, recent feed; per-row "resolve now" / "override verdict" / "revert to pending" actions |
| `/admin/users` | Map sender hashes → display names so attribution + memory subjects work without users typing their own name |
| `/admin/live` | Live event stream (SSE) — every inbound message, command result, reactor decision, prediction resolution, oracle post |

---

## LLM Integration

Configure once at `/admin/llm`. All values apply live — no restart.

### Provider config

| Field | Notes |
|-------|-------|
| Enabled | Master on/off |
| Base URL | Any OpenAI-compatible endpoint (`https://api.openai.com/v1`, `https://openrouter.ai/api/v1`, `https://api.groq.com/openai/v1`, `http://host:11434/v1` for Ollama, etc.) |
| Model | e.g. `gpt-4o-mini`, `anthropic/claude-haiku-4-5` (via OpenRouter), `llama3.1:8b` |
| API key | Write-only field — once saved, the form shows "configured" and never echoes the value back |
| Temperature, Max tokens, Timeout | Standard knobs |
| Conversation turns kept per user | Rolling window depth for `!ask` history |
| Retention days | Auto-purge for both conversation history and group message logs |
| Max tool-call rounds | Hard ceiling on chained tool calls per `!ask` (default 25). On cap-hit, the bot returns an honest "task incomplete" error with the tool sequence — never fabricates a summary from partial work |
| Group chat context messages | When `!ask` runs in a group, inject this many recent messages from that group into the system prompt. `0` disables both context injection and group-message recording |
| Augment these commands | Comma-separated list (e.g. `ta, tldr, earnings, news, economy, rating`). When a listed command succeeds, the LLM appends a brief plain-language interpretation. Respects per-context policy |
| Augmentation prompt | Instruction template for augmentation calls |
| Ask command alias | Adds an extra alias for `!ask` (the canonical name always works too) |
| System prompt | Default LLM system prompt; overridden per-context if set on a context |
| Extra request body (JSON) | Merged into every chat payload — use for `thinking`, `reasoning_effort`, `top_p`, etc. Validated as JSON object on save |
| Preferred providers (OpenRouter) | Comma-separated provider slugs (e.g. `Cerebras, Groq, Together`). Sets `provider.order` on the request body |
| Only these providers | When true, sets `provider.allow_fallbacks: false` so the request fails rather than silently routing to a slow host |
| Sort by | `throughput` / `latency` / `price` — sets `provider.sort` so OpenRouter ranks within (or globally absent) the preferred list |

Deep-think gets its own block with separate base URL / API key / model / temperature / max-tokens / system prompt / extra body / per-user + per-group daily caps. It's exposed to the writer LLM as a `deep_think(question, context, status_message)` tool — the writer delegates hard sub-problems and the deep model gets the same toolkit (price/news/MCP/etc.) to research before answering.

### What gets sent to the LLM on every call

- The configured (or per-context) system prompt + persona
- **Always** the current UTC + ET time with weekday — so the model never has to guess "now"
- Per-context conversation history. In groups, user turns prefix `[Name, time ago]` (or `[...4137, time ago]` for unregistered users) and the bot's own past replies prefix `[to Name, time ago]` so question/answer pairing across speakers is unambiguous
- Optional `<group_context>` block — last N inbound messages (configurable per chat) plus the bot's own posts back into the chat, time-ordered with the same bracket attribution
- All bot commands the context allows, as `bot__<name>` tools
- All MCP tools the context allows, as `<server>__<tool>` tools
- Tweet URLs in the user's question are auto-expanded to `[@handle] tweet text` before sending
- Conditional system-suffix injections, gated on per-context state:
  - **`<attribution_rules>`** — multi-speaker rules: `[Name]` and `[to Name]` brackets are internal metadata only; never echoed into visible replies (a regex-based stripper enforces this on the way out, just in case)
  - **`<context_memories>`** — auto-injected memory preamble for the active speaker, the room, and any names mentioned in the message (no `recall` tool call needed for in-frame subjects)
  - **`<conversation_memory>`** — long-running rolling summary when conversation history gets long
  - **`<reactor_reflex>` + recent-reactions log** — when reactor is enabled, so the model can answer "why did you react with X?" honestly
  - **`<tarot_tool>` / `<iching_tool>`** — force-route divination through the actual rendering tools rather than fabricating cards in text
  - **`<deep_think_tool>`** — when the deep client is wired and the context allows it
  - **`<python_tool>`** — when the Pyodide MCP is up: tells Sigil it has a real Python interpreter with numpy/pandas/scipy/yfinance/statsmodels, when to use it, and to set a 30s timeout for non-trivial work
  - **`<spontaneous_reply>`** — when the reactor's `should_respond` flagged this message; instructs the writer to bail to empty content if a real reply isn't warranted
  - **Identity / staleness notes** — e.g. "real names are now available, disregard older turns where you claimed not to know them"

### Memory tools

When a per-context memory store row exists for the chat, the writer LLM gets:

| Tool | Purpose |
|------|---------|
| `remember(subject, kind, content)` | Save a memory. Subject can be a registered name, `"yourself"` (the bot), `"this chat"` (the room), or free text. Kind: `identity` / `preference` / `fact` / `event`. Near-duplicate corroboration bumps the existing row's count + confidence rather than inserting a parallel row |
| `recall(subject?, query?)` | Look up. Active-speaker / room / bot-self memories are auto-injected, so this is for the rest |
| `forget(memory_id)` | Delete by id |

Confidence ladder: explicit user-driven memories about themselves start at 0.9; about a third party at 0.6 (corroboration from a second speaker promotes); reactor-sourced memories start at 0.4 and only promote to 1.0 once N **distinct** speakers corroborate (single-user repetition can't self-promote). Reactor pre-write also runs a looser cross-kind dedup so the same fact written under `fact` once and `preference` next doesn't land twice.

Memories can be browsed / edited from `/admin/contexts/<id>/memories`.

### `!ask` examples

```
!ask what's the macro setup this week?
!ask what was AAPL doing yesterday?           # LLM may call bot__price or yahoo-finance MCP
!ask reset                                     # Clear current chat's history
!ai how does this look?                        # If alias is configured
```

---

## Emoji Reactor

A separate fire-and-forget LLM that decides whether to react to inbound group messages with a single emoji. Configured at `/admin/llm` under "Reactor". Off by default; per-context `reactor_enabled` toggle in `/admin/contexts`.

**How it fires.** Every inbound group message kicks off a background `maybe_react()` task in parallel with normal command dispatch. The reactor runs through cheap rules first (per-sender cooldown, per-group cooldown, min message length), then passes the message + recent group context through the configured "reactor" model (typically a cheap/fast variant — e.g. Sonnet over Opus, or DeepSeek with thinking disabled) and gives the LLM exactly one tool: `emoji_react(emoji)`. If the LLM calls the tool, we POST a Signal reaction; if it doesn't, the user never sees anything. All errors are logged and swallowed — the reactor must never affect command handling or surface diagnostics.

**Coordination with the writing LLM.** The reactor and the writing LLM (`!ask` and `llm_intent` routing) are separate processes. A small bridge gives the writer enough state to own and explain its reactions:

1. A **reflex directive** in the stable system suffix explains that emoji reactions belong to the bot and tells the writer how to use the event log when one is present.
2. A **rolling log** carries the last 5 reactions in the current group with target message and emoji (for example, `💀 on [Tyler] "housing crash tweet…"`). This volatile block sits in the current user turn beside `<group_context>`, preserving the provider's cached system/history prefix when a new reaction arrives. The backing log is in-memory per process, capped at 20 per group, and clears on restart.

Both pieces are gated on `reactor_enabled` being true globally **and** for the current context — they don't appear in the prompt for chats where the reactor is off.

**Tuning.** Reactor model, max tokens, temperature, system prompt, and cooldowns all live in `/admin/llm`. Per-context system prompt overrides live in `/admin/contexts`.

---

## Predictions Game

A polymarket-style on-chat prediction registry: anyone can log a dated claim, the bot follows up at the deadline, and a per-chat leaderboard tracks accuracy.

### Logging predictions (humans + the bot)

| Path | Author identity | When |
|------|-----------------|------|
| `!predict <claim> by <when>` | The user who typed it | Manual, in-chat |
| `bot__predict` LLM tool | The asker (whoever ran `!ask`) | LLM was asked to log on the asker's behalf |
| `predict_self` LLM tool | The bot itself (`BOT_SENDER` sentinel hash) | Sigil stakes its own forecast |
| `predict_for(subject, claim)` LLM tool | A third-party chat member | "Anthony just said SPY goes down tomorrow" → logged on Anthony's row, not Sigil's, not the asker's. Subject can be a registered name **or** the `...4810` phone-tail form for unregistered users |
| `predict_update(id, claim)` LLM tool | (preserves original predictor) | Revise claim/deadline within **15 minutes** of creation. Past that, locked — admin override on dashboard only |

### Deadline parsing

`<claim> by <date>`. Stock-shape claims (`TICKER above|below $price by date`) get a deterministic regex extraction; freeform claims fall through to an LLM extractor. Bare dates default to **21:00 UTC** (≈ post-NY-close). `EOD` / `EOM` / `EOW` and weekday names (`by Friday`) are handled explicitly. The parser also corrects `dateparser`'s "PREFER_DATES_FROM=future" year-bump bug: `April 29` said early on April 29 in UTC stays in this year rather than jumping to next April 29.

### Resolution paths (consensus, not free-for-all)

`!resolve <id> right|wrong|unclear [reason]` is a **vote**, not an instant verdict:

- **Auto-resolver** (cron, every 15 min) — for structured stock-shape claims, fetches the live quote and applies. For freeform claims, hands off to Sigil running through a small tool loop with the bot's research kit (price/news/chart/Brave search/EDGAR/etc.) restricted to read-only commands. Most predictions resolve here without anyone touching `!resolve`
- **Admin** (number in `ADMIN_NUMBERS`) — resolves solo, immediately
- **Non-admin** — vote is recorded; **2 distinct non-predictor users agreeing on the same verdict** applies the resolution. Voters can change their mind (the upsert replaces their previous verdict). The **predictor cannot vote on their own** prediction
- Disagreement keeps the prediction pending — the chat response shows the tally (`1 right, 2 wrong`) and how many more agreements are needed

### Leaderboard

`!leaderboard` ranks chat members by accuracy on resolved (`right` + `wrong`) predictions. `unclear` and `expired` rows don't count. Sigil shows up on the leaderboard as its own row when it stakes via `predict_self`. The web view at `/admin/predictions` adds aggregate counters, per-context leaderboards, upcoming deadlines, and per-row admin actions (resolve-now via Sigil, override verdict + note, revert to pending).

---

## Per-Context Daily Oracles

Each group context can have any number of scheduled posts that fire once per day. Manage from `/admin/contexts/<id>` (Daily oracles panel).

### Kinds

| Kind | What Sigil posts |
|------|------------------|
| `tarot` | Single random card draw with the rendered spread image |
| `iching` | Three-coin cast with the procedurally-rendered hexagram |
| `market_open` | Pre-market check — pulls index futures + day's events via tools, writes a 2-4 sentence opener |
| `market_close` | Closing recap — index moves + leaders/laggards + notable news |
| `freeform` | Sigil-generated post from an admin-supplied prompt |

### Schedules

| Schedule | Behavior |
|----------|----------|
| `sunrise` | NYC civil sunrise + signed `offset_minutes` |
| `sunset` | NYC civil sunset + signed `offset_minutes` |
| `clock` | Fixed `HH:MM` in any IANA timezone |

`weekdays_only` flag skips Saturday and Sunday — sensible default for the market kinds.

### Heuristic prepopulation

On first boot of the new schema, every group context gets default oracle rows seeded based on its label:

- Labels matching `money / stocks / finance / market / trader / trading / invest / wsb / wallstreet` → `market_open` at 09:25 ET + `market_close` at 16:05 ET, weekdays only
- Labels matching `woo / astro / tarot / magic / witch / spirit / occult / esoteric / divin / ritual / mystic / moon` → sunrise tarot (NYC, +1 min)
- Other labels → nothing seeded; admin adds via UI

All seeded rows land **disabled** by default — admin reviews and flips enabled per row. Idempotent: contexts that already have any oracle row are skipped on re-run, so admin edits survive deploys. Per-oracle `last_fired_at` tracking makes restarts during the firing window safe (no double-post).

---

## MCP Servers

Manage at `/admin/mcp`. Supports stdio, SSE, and streamable HTTP transports.

### Container has

- `python` / `python3`
- `uvx` (Astral's uv) — Python servers from PyPI, wheels, or `git+https://...` URLs
- `node` / `npx` — npm-packaged servers
- `git` — required by uvx for git-hosted Python servers
- Pre-installed servers: `@brave/brave-search-mcp-server`, `mcp-pyodide` (used by the Python sandbox; see below)

### Auto-registered defaults

On first boot the registry seeds defaults that show up in `/admin/mcp` ready to go (admin can disable but if a default row is deleted it's recreated on next boot — toggle `enabled` rather than deleting):

| Server | What it gives Sigil |
|--------|---------------------|
| `pyodide` | Real Python interpreter via Pyodide-in-Node. Pre-loaded: numpy, pandas, scipy, matplotlib, scikit-learn (Pyodide bundle) plus yfinance + statsmodels (pre-cached). State persists across calls in a conversation. Used for actual computation — correlations, regressions, NPV/IRR, options pricing, custom indicators — not paraphrasing fetched data. Pre-warmed at bot boot so the writer LLM's first call doesn't pay the bundle-load + wheel-download cost (~10-15s otherwise). Cache dir lives on the persistent volume |

### Adding a server

The form maps 1:1 to Claude-Desktop-style MCP configs:

```json
"server-name": {
  "command": "uvx",
  "args": ["--from", "git+https://github.com/foo/bar", "bar-mcp"],
  "env": {"API_KEY": "..."}
}
```

| JSON | Form field |
|------|-----------|
| key  | Name |
| `command` | Command |
| `args` (array) | Args (one per line) |
| `env` (object) | Env (`KEY=value` per line) |

For SSE/HTTP transports, fill **URL** and **Headers** instead of Command/Args.

### Examples

**Filesystem (Node)**
- Command: `npx`
- Args: `-y` / `@modelcontextprotocol/server-filesystem` / `/app/data`

**Yahoo Finance MCP (Python via git)**
- Command: `uvx`
- Args: `--from` / `git+https://github.com/Alex2Yang97/yahoo-finance-mcp` / `yahoo-finance-mcp`

First start of any new server takes 15-40s while uvx/npx download the package; subsequent starts are fast.

Auto-started servers (`enabled = true`) come up at bot boot. Failed startups log and don't crash the bot.

---

## Per-Context Policies

Manage at `/admin/contexts`. Every chat (group or DM) can have its own:
- **Command allow/deny list** — block `!corn`, restrict `!alert` to one chat, etc.
- **MCP server allow/deny list** — financial tools only in the trading group, geopolitics tools only in the politics group, etc.
- **System prompt override** — different LLM personality per chat
- **LLM intent routing** — when on, non-command messages route through `!ask` with all allowed tools; replaces the regex NLP fallback for that chat

### Auto-registration

Groups auto-register a stub policy on the first message the bot sees there (so they show up in the admin list, ready to edit). DMs don't auto-register — they fall back to `default:dm` unless you explicitly add them.

Two protected default rows always exist:
- `default:group` — applies to any group without an explicit policy
- `default:dm` — applies to any DM without an explicit policy

### Modes

| Mode | Behavior |
|------|----------|
| Allow all | No restriction. For MCP, every schema from every running server is sent. |
| Allow only selected | Whitelist |
| Block selected | Blacklist (use this to hide just `corn` etc. without re-enabling everything else) |

Gating is by canonical command name, so allowing `price` automatically allows `!p`, `!pr`, `$`. Bot tools and MCP tools exposed to the LLM are filtered through the same policy — disallowed tools simply don't appear in the LLM's tool list.

New and auto-registered contexts start with an empty MCP allow-list. This keeps
large server schemas out of the writer prompt until an admin selects them.
Existing contexts retain their stored access mode. The context editor shows the
effective live tool count and serialized schema size; selections are ignored
when the mode is **Allow all servers**.

---

## Supported Symbols

### Stocks & ETFs
- US stocks: `AAPL`, `MSFT`, `GOOGL`, `TSLA`, etc.
- ETFs: `SPY`, `QQQ`, `VTI`, `ARKK`, etc.

### Indices
- S&P 500: `^GSPC` or `sp500`
- Dow Jones: `^DJI` or `dow`
- Nasdaq: `^IXIC` or `nasdaq`
- VIX: `^VIX` or `vix`

### Commodities & Futures
- Gold: `GC=F` or `gold`
- Silver: `SI=F` or `silver`
- Oil/Crude: `CL=F` or `oil`
- Natural Gas: `NG=F` or `gas`
- Copper, Wheat, Corn, Coffee, etc.

### Bonds & Treasuries
- 10-Year: `^TNX` or `10y` or `treasury`
- 30-Year: `^TYX` or `30y`
- TLT ETF: `TLT`

### Crypto
- Bitcoin: `BTC-USD` or `btc` or `bitcoin`
- Ethereum: `ETH-USD` or `eth`
- Solana, Cardano, Dogecoin, XRP, etc.

### Forex
- Euro: `EURUSD=X` or `euro`
- Pound: `GBPUSD=X` or `pound`
- Dollar Index: `DX-Y.NYB` or `dxy`

---

## Inline Symbol Detection

Mention symbols with `$` anywhere in a message:

```
What do you think about $AAPL?
→ Apple Inc. (AAPL) ◈ $185.92 ▲ +1.27%

Comparing $MSFT and $GOOGL today
→ ● MSFT: $378.91 (+0.89%)
  ○ GOOGL: $141.80 (-0.32%)
```

---

## Configuration

### Environment Variables

#### Bot core

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_PHONE_NUMBER` | Yes | — | Bot's Signal phone number |
| `SIGNAL_API_URL` | No | `http://signal-api:8080` (in compose) | Signal CLI REST API URL |
| `BOT_NAME` | No | `Stock Bot` | Bot name (shown on charts, live-editable in admin UI) |
| `COMMAND_PREFIX` | No | `!` | Command prefix |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `USER_RATE_LIMIT` | No | `30` | Max requests/min per user (live-editable) |
| `MAX_MESSAGE_LENGTH` | No | `4000` | Reject inbound messages longer than this (live-editable) |
| `ADMIN_NUMBERS` | No | — | Comma-separated phone numbers with admin privileges + 2FA recipient |

#### Provider API keys

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPHAVANTAGE_API_KEY` | No | — | Alpha Vantage API key (25/day free) |
| `POLYGON_API_KEY` | No | — | Polygon.io API key |
| `FINNHUB_API_KEY` | No | — | Finnhub API key (60/min free) |
| `TWELVEDATA_API_KEY` | No | — | Twelve Data API key (800/day free) |
| `FRED_API_KEY` | No | — | FRED API key for `!eco` (120/min free) |
| `MASSIVE_PRO` | No | `false` | Enable `!options` (Polygon Pro) |

#### Admin web UI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADMIN_PASSWORD_HASH` | No (env-var fallback) | — | bcrypt hash; preferred location is `./admin_password.hash` (set via `python -m src.admin_setup`) |
| `ADMIN_PASSWORD_HASH_PATH` | No | `data/admin_password.hash` (file) / `/app/admin_password.hash` (in-container) | Override hash file path |
| `FLASK_SECRET_KEY` | Required for admin UI | — | Sign session cookies. Generate with `python -m src.admin_setup` |
| `SESSION_COOKIE_SECURE` | No | `false` | Set `true` when behind HTTPS |
| `WEBHOOK_SECRET` | No | — | If set, `/webhook` requires `X-Webhook-Secret: <value>` header |
| `ADMIN_BIND_HOST` | No | `127.0.0.1` (compose) | Interface to bind admin UI port to. Set `0.0.0.0` for LAN |

LLM, MCP server, and per-context settings are managed entirely in the web UI — no env vars.

### Data Providers

The bot supports multiple data providers with automatic failover. Add more providers for better rate limit capacity:

| Provider | Free Tier | Signup |
|----------|-----------|--------|
| **Yahoo Finance** | Unlimited (unofficial) | No key needed |
| **Finnhub** | 60 calls/min | [finnhub.io](https://finnhub.io) |
| **Twelve Data** | 800 calls/day | [twelvedata.com](https://twelvedata.com) |
| **FRED** | 120 calls/min | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **Alpha Vantage** | 25 calls/day | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| **Polygon.io** | 5 calls/min | [polygon.io](https://polygon.io) |

Providers are tried in priority order. When one is rate-limited, the next is used automatically.

### Cache TTLs

| Data Type | TTL |
|-----------|-----|
| Intraday quotes | 60 seconds |
| Daily quotes | 5 minutes |
| Charts | 5 minutes |
| News | 10 minutes |
| Fundamentals | 1 hour |
| Earnings | 1 hour |
| Historical data | 24 hours |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Signal Network                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  signal-cli-rest-api (Docker)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │ WebSocket (json-rpc)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       stock-bot (Docker)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │   Dispatcher  →  Context policy  →  Command handler        │  │
│  │                                                            │  │
│  │   Commands: price quote chart ta rsi macd earnings news    │  │
│  │             rating insider short corr alert watch ask ...  │  │
│  │                                                            │  │
│  │   Optional augmentation hook: LLM appends interpretation   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Market data    │  │ LLM client   │  │ MCP manager        │   │
│  │ Providers w/   │  │ + writer +   │  │ stdio / sse / http │   │
│  │ failover +     │  │ deep_think + │  │ Pyodide auto-      │   │
│  │ caching        │  │ reactor      │  │ registered         │   │
│  └────────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                  │
│  Background workers (asyncio): alert sweeper, prediction         │
│    resolver (cron + tool-enabled Sigil), per-context oracle      │
│    scheduler, attachment-cache reaper, summarizer                │
│                                                                  │
│  Persistence (SQLite): watchlists · alerts · contexts · LLM      │
│    history · group log · per-context memories · predictions +    │
│    resolution_votes · context_oracles · MCP server configs ·     │
│    admin settings · user nicknames                               │
│                                                                  │
│  Admin UI (Flask + Jinja) at /admin — bcrypt + Signal 2FA;       │
│  dashboard, predictions console, per-context editor (incl.       │
│  oracles + memories), live SSE event feed, users registry        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Development

### Local setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-test.txt
pytest
```

### Running locally

```bash
export SIGNAL_API_URL=http://localhost:8080
export SIGNAL_PHONE_NUMBER=+15551234567
python -m src.main
```

---

## Maintenance

### Update containers

```bash
docker compose pull
docker compose up -d
```

### Backup Signal credentials

```bash
tar -czvf signal-backup-$(date +%Y%m%d).tar.gz ./data/signal-cli
```

### View logs

```bash
docker compose logs -f stock-bot
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Bot not responding | Check `docker compose ps`, verify Signal API health |
| Symbol not found | Use smart names (`apple`, `gold`) or full symbols (`BRK.B`) |
| Rate limited | Add more providers, check `!status` |
| Messages delayed | Ensure using `MODE=json-rpc` |
| Group chat fails | Bot uses fallback DM. Run `docker exec -it --user 1000 signal-api signal-cli -u <PHONE> listGroups` to force sync |
| Admin UI 502 / can't reach | Check `ADMIN_BIND_HOST` and that port 5000 is published in compose; admin only mounts when `ADMIN_PASSWORD_HASH` + `FLASK_SECRET_KEY` are both set |
| Admin password rejected | Re-run `python -m src.admin_setup` and confirm `./admin_password.hash` exists with mode `0600`; the env-var `ADMIN_PASSWORD_HASH` is fallback only |
| 2FA code never arrives | First number in `ADMIN_NUMBERS` must be reachable from the bot's Signal account — DM the bot from that number once to confirm |
| MCP server fails to start (git error) | Container needs `git`; a fresh build (`docker compose up -d --build`) installs it |
| MCP server stalls during first start | uvx/npx is downloading the package — first start can take 15-40s; subsequent starts are fast |
| `!ask` returns "task didn't complete after N rounds" | Raise *Max tool-call rounds* in `/admin/llm`. Tool history is in `logs/bot.log` to see what the LLM was chasing |
| Easter egg / unwanted command in a group | Open `/admin/contexts/<id>` and either deny-list it, or switch the chat to allow-list mode |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [signal-cli](https://github.com/AsamK/signal-cli) — Signal protocol
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
- [mplfinance](https://github.com/matplotlib/mplfinance) — Professional charts

