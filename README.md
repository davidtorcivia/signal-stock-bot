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

### Admin & Intelligence (new)
- **Admin web UI** at `/admin` — login + Signal-delivered 2FA, settings dashboard, live edits without restart
- **LLM integration** — `!ask` command with conversation history, OpenAI-compatible (works with OpenAI, OpenRouter, Groq, Anthropic-via-OR, Ollama, llama.cpp, etc.); always injects current UTC time into the system prompt
- **MCP servers** — register and manage Model Context Protocol servers (stdio / SSE / HTTP) via the admin UI; `npx`, `uvx`, `git` available in the container
- **LLM tool calling** — bot commands and MCP tools exposed to the LLM as function calls; chart attachments bubble back into the chat
- **Per-context policies** — scope which commands and MCP servers work in each group/DM; assign per-chat system prompts; toggle whether non-command messages route through the LLM (with bot tools) or the regex NLP fallback
- **LLM command augmentation** — opt-in: append a brief plain-language interpretation to the output of selected commands (e.g. `!ta`, `!rating`)
- **Twitter / X URL expansion** — pasted tweet links auto-resolve to text via fxtwitter so the LLM can see what users shared
- **Group chat memory** — shared conversation thread per group with per-speaker attribution; configurable retention with auto-purge

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
| `/admin/` | Dashboard — uptime, requests/min, cache stats, provider health, "Clear all caches" |
| `/admin/settings` | Live-editable: bot name, rate limit, message length cap, admin numbers, webhook secret. Restart-required: command prefix, provider API keys |
| `/admin/llm` | LLM provider config + augmentation + tool-round cap (see below) |
| `/admin/mcp` | Add / start / stop / delete MCP servers, view discovered tools (see below) |
| `/admin/contexts` | Per-chat policy: command allow/deny, MCP allow/deny, system prompt, LLM intent toggle (see below) |

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

### What gets sent to the LLM on every call
- The configured (or per-context) system prompt
- **Always** the current UTC time (`Current time: YYYY-MM-DD HH:MM:SS UTC`) — so the model never has to guess "now"
- Per-user (DM) or per-group (group) conversation history
- The user's question, with `[...1234]` sender attribution prefix in group threads
- All bot commands the context allows, as `bot__<name>` tools
- All MCP tools the context allows, as `<server>__<tool>` tools
- Optional group chat context (last N messages from the group, formatted with sender tails)
- Tweet URLs in the user's question are auto-expanded to `[@handle] tweet text` before sending

### `!ask` examples

```
!ask what's the macro setup this week?
!ask what was AAPL doing yesterday?           # LLM may call bot__price or yahoo-finance MCP
!ask reset                                     # Clear current chat's history
!ai how does this look?                        # If alias is configured
```

---

## MCP Servers

Manage at `/admin/mcp`. Supports stdio, SSE, and streamable HTTP transports.

### Container has

- `python` / `python3`
- `uvx` (Astral's uv) — Python servers from PyPI, wheels, or `git+https://...` URLs
- `node` / `npx` — npm-packaged servers
- `git` — required by uvx for git-hosted Python servers

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
| Allow all | No restriction (default) |
| Allow only selected | Whitelist |
| Block selected | Blacklist (use this to hide just `corn` etc. without re-enabling everything else) |

Gating is by canonical command name, so allowing `price` automatically allows `!p`, `!pr`, `$`. Bot tools and MCP tools exposed to the LLM are filtered through the same policy — disallowed tools simply don't appear in the LLM's tool list.

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
│  │ Providers w/   │  │ OpenAI-compat│  │ stdio / sse / http │   │
│  │ failover +     │  │ tool-calling │  │ npx / uvx / git    │   │
│  │ caching        │  │ + history    │  │ available          │   │
│  └────────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                  │
│  Persistence (SQLite): watchlists · alerts · contexts · LLM      │
│    history · group log · MCP server configs · admin settings     │
│                                                                  │
│  Admin UI (Flask + Jinja + HTMX) at /admin — bcrypt + Signal 2FA │
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

