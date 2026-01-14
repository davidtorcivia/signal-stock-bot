# Signal Stock Bot

A self-hosted Signal bot for real-time stock quotes, market data, technical analysis, and company fundamentals. Built with a multi-provider architecture for reliability and extensibility.

## Features

- **Real-time stock quotes** via Yahoo Finance, Finnhub, Twelve Data, Alpha Vantage, and Polygon
- **Multi-provider failover** — automatic fallback when rate limited
- **Smart symbol resolution** — type `!price apple` or `!price gold` instead of tickers
- **Professional charting** with candlesticks, indicators, and comparisons
- **Technical analysis** with RSI, MACD, SMA, support/resistance
- **Command chaining** — run multiple commands at once: `!price AAPL !tldr AAPL !news AAPL`
- **Universal -help flag** — add `-help` to any command for detailed explanations
- **Earnings & dividends** tracking
- **News headlines** for any symbol
- **Options & Futures** support (via Polygon)
- **Forex & Crypto** support
- **Economy Indicators** (CPI, GDP, etc.)
- **Batch symbol lookups** (e.g., `!price AAPL MSFT GOOGL`)
- **Inline symbol detection** (e.g., `Check $AAPL price`)
- **@ mention support** (mention the bot to get help)
- **Intelligent caching** with type-specific TTLs
- **Clean unicode output** (no emojis)

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

### Admin Commands

| Command | Description |
|---------|-------------|

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

## Admin Management

To enable admin features, set `ADMIN_NUMBERS` in `.env`.

### Commands
- `!admin backup` - Export all user watchlists (JSON)
- `!admin alerts` - View global alert stats
- `!admin users` - User activity stats
- `!metrics` - System health, uptime, request rates
- `!cache stats` - View cache hit rates
- `!cache clear` - Flush all caches

### Configuration
New `.env` options:
```bash
# Admin phone numbers (comma-separated)
ADMIN_NUMBERS=+15551234567,+15559876543

# Rate limit per user (requests/minute)
USER_RATE_LIMIT=30
```

---


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

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_PHONE_NUMBER` | Yes | — | Bot's Signal phone number |
| `SIGNAL_API_URL` | No | `http://localhost:8080` | Signal API URL |
| `BOT_NAME` | No | `Stock Bot` | Bot name (shown on charts) |
| `COMMAND_PREFIX` | No | `!` | Command prefix |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `ALPHAVANTAGE_API_KEY` | No | — | Alpha Vantage API key (25/day free) |
| `POLYGON_API_KEY` | No | — | Polygon.io API key |
| `FINNHUB_API_KEY` | No | — | Finnhub API key (60/min free) |
| `TWELVEDATA_API_KEY` | No | — | Twelve Data API key (800/day free) |
| `FRED_API_KEY` | No | — | FRED API key for `!eco` (120/min free) |
| `MASSIVE_PRO` | No | `false` | Enable `!options` (Polygon Pro) |

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
┌──────────────────────────────────────────────────────────┐
│                    Signal Network                         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              signal-cli-rest-api (Docker)                │
└─────────────────────────┬────────────────────────────────┘
                          │ JSON-RPC
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   stock-bot (Docker)                     │
│                                                          │
│  Commands: price, quote, chart, ta, rsi, macd,          │
│            earnings, dividend, news, market, crypto...   │
│                                                          │
│  Providers: Yahoo Finance, Alpha Vantage, Polygon       │
│                                                          │
│  Features: Smart caching, rate limiting, circuit breaker│
└──────────────────────────────────────────────────────────┘
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
| Group Chat Fails | Bot uses fallback DM. Run `docker exec -it --user 1000 signal-api signal-cli -u <PHONE> listGroups` to force sync. |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [signal-cli](https://github.com/AsamK/signal-cli) — Signal protocol
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
- [mplfinance](https://github.com/matplotlib/mplfinance) — Professional charts

