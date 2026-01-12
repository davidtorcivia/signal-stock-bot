# Signal Stock Bot

A self-hosted Signal bot for real-time stock quotes, market data, technical analysis, and company fundamentals. Built with a multi-provider architecture for reliability and extensibility.

## Features

- **Real-time stock quotes** via Yahoo Finance, Alpha Vantage, and Polygon
- **Smart symbol resolution** — type `!price apple` or `!price gold` instead of tickers
- **Professional charting** with candlesticks, indicators, and comparisons
- **Technical analysis** with RSI, MACD, SMA, support/resistance
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

### Market Overview Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `!market` | `!m` | Major indices (S&P, Dow, Nasdaq, Russell, VIX) |
| `!crypto` | `!c` | Top cryptocurrencies |
| `!forex EURUSD` | `!fx` | Currency pairs |
| `!future CL` | `!fut` | Futures quotes |
| `!economy CPI` | `!eco` | Economic indicators (requires Polygon Pro) |
| `!options AAPL` | `!opt` | Options chains (requires Polygon Pro) |

---

### Admin Commands

| Command | Description |
|---------|-------------|
| `!metrics` | Bot performance dashboard (uptime, req/min, cache hits, provider health) |
| `!cache stats` | Detailed cache statistics by type |
| `!cache clear` | Clear all caches |
| `!status` | Provider health status |
| `!help` | Command list |

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
| `ALPHAVANTAGE_API_KEY` | No | — | Alpha Vantage API key |
| `POLYGON_API_KEY` | No | — | Polygon.io API key |
| `MASSIVE_PRO` | No | `false` | Enable `!options` and `!economy` |

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

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [signal-cli](https://github.com/AsamK/signal-cli) — Signal protocol
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
- [mplfinance](https://github.com/matplotlib/mplfinance) — Professional charts

