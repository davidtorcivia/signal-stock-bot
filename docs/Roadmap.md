# Signal Stock Bot — Feature Roadmap

A comprehensive proposal to elevate the bot to professional-grade quality.

---

## 1. Chart Generation System ✅ (Implemented)

| Command | Description |
|---------|-------------|
| `!chart AAPL` | 1-month (daily bars) |
| `!chart AAPL 1d` | Intraday (5-min bars) |
| `!chart AAPL 5d` | 5-day (15-min bars) |
| `!chart AAPL 1y` | 1-year (daily) |
| `!chart AAPL ytd` | Year-to-date |

**Future enhancements:**
- Candlestick mode (`-c` flag)
- Technical overlays (SMA, Bollinger, RSI)
- Symbol comparison (`-compare MSFT`)

---

## 2. Watchlist System

Persistent watchlists per user for quick monitoring.

| Command | Description |
|---------|-------------|
| `!watch add AAPL MSFT` | Add symbols to watchlist |
| `!watch remove TSLA` | Remove symbol |
| `!watch` | Show watchlist with live prices |
| `!watch clear` | Clear entire watchlist |

**Storage:** SQLite database, keyed by hashed phone number.

---

## 3. Price Alerts

Push notifications when price thresholds are hit.

| Command | Description |
|---------|-------------|
| `!alert AAPL > 190` | Alert when above $190 |
| `!alert AAPL < 180` | Alert when below $180 |
| `!alert AAPL %5` | Alert on 5% move |
| `!alerts` | List active alerts |
| `!alert delete 1` | Delete alert by ID |

**Background worker:** Polling service (60s default), smart batching, alert deduplication.

---

## 4. Portfolio Tracking

| Command | Description |
|---------|-------------|
| `!portfolio buy AAPL 10 @ 185` | Log purchase |
| `!portfolio sell AAPL 5 @ 195` | Log sale |
| `!portfolio` | Show current holdings |
| `!portfolio performance` | P&L breakdown |
| `!portfolio chart` | Equity curve visualization |

---

## 5. Technical Analysis

| Command | Description |
|---------|-------------|
| `!ta AAPL` | Technical summary |
| `!rsi AAPL` | RSI with interpretation |
| `!macd AAPL` | MACD signal |
| `!sma AAPL 20 50 200` | Moving averages |
| `!support AAPL` | Support/resistance levels |

---

## 6. Earnings & Events

| Command | Description |
|---------|-------------|
| `!earnings AAPL` | Next earnings date + estimates |
| `!earnings week` | This week's notable earnings |
| `!dividend AAPL` | Dividend info + ex-date |

---

## 7. News Integration

| Command | Description |
|---------|-------------|
| `!news AAPL` | Recent headlines (3-5) |
| `!news` | Market-wide headlines |

---

## 8. Admin & Metrics

| Command | Description |
|---------|-------------|
| `!admin stats` | Usage statistics |
| `!admin users` | Active user count |
| `!admin block +1234567890` | Block user |
| `!metrics` | Cache hit rate, provider health |

---

## Implementation Priority

| Phase | Features | Timeline |
|-------|----------|----------|
| **1** | Charts ✅ | Complete |
| **2** | Watchlists & Alerts | Week 1-2 |
| **3** | Technical Analysis | Week 3-4 |
| **4** | Portfolio & Advanced | Week 5-6 |

---

## Dependencies

```txt
# Already added
matplotlib>=3.8.0

# Future phases
aiosqlite>=0.19.0      # Watchlists/alerts
apscheduler>=3.10.0    # Background jobs
pandas>=2.0.0          # TA calculations
mplfinance>=0.12.10    # Candlestick charts
```
