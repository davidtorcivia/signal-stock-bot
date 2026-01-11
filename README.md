# Signal Stock Bot

A self-hosted Signal bot for real-time stock quotes, market data, and company fundamentals. Built with a multi-provider architecture for reliability and extensibility.

## Features

- **Real-time quotes** — Current price, change, volume for any ticker
- **Batch lookups** — Query multiple symbols in one message
- **Inline symbol detection** — Type `$AAPL` in any message for quick lookup
- **@ mention support** — Tag the bot in groups to get help or stock info
- **Crypto support** — Quick view of top cryptocurrencies
- **Market overview** — Major indices at a glance
- **Company fundamentals** — P/E, EPS, market cap, sector info
- **Multi-provider fallback** — Yahoo Finance (free), Alpha Vantage, Polygon.io
- **Quote caching** — 5-minute cache reduces API calls
- **Rate limit handling** — Automatic provider rotation when limits hit
- **Retry logic** — Automatic retries with exponential backoff
- **Group chat support** — Works in both DMs and group chats

## Quick Start

### Prerequisites

- Ubuntu 22.04+ server (or any Docker-capable host)
- Docker and Docker Compose
- A phone number for Signal (can use your existing account)

### 1. Clone and configure

```bash
git clone https://github.com/youruser/signal-stock-bot.git
cd signal-stock-bot

# Create environment file
cp .env.example .env

# Edit with your phone number
nano .env
```

Minimum `.env` configuration:

```bash
SIGNAL_PHONE_NUMBER=+15551234567  # Your Signal number
```

### 2. Start the stack

```bash
docker compose up -d
```

### 3. Link your Signal account

```bash
# Generate QR code
curl -s "http://localhost:8080/v1/qrcodelink?device_name=stockbot" | docker run -i --rm mtgto/qrencode -t ANSIUTF8

# Or view in browser: http://your-server:8080/v1/qrcodelink?device_name=stockbot
```

On your phone: **Signal → Settings → Linked Devices → Link New Device** → Scan QR

### 4. Test it

Send `!price AAPL` to your Signal number. You should get a response within seconds.

## Commands

### `!price` / `!p` — Get current price

```
!price AAPL
```
```
Apple Inc. (AAPL)
💵 $185.92
📈 +2.34 (+1.27%)
📊 Vol: 52.3M
```

**Batch mode** — up to 10 symbols:

```
!price AAPL MSFT GOOGL NVDA
```
```
🟢 AAPL: $185.92 (+1.27%)
🟢 MSFT: $378.91 (+0.89%)
🔴 GOOGL: $141.80 (-0.32%)
🟢 NVDA: $495.22 (+2.15%)
```

### `!quote` / `!q` — Detailed quote

```
!quote TSLA
```
```
📊 Tesla, Inc. (TSLA)

Price: $248.50
📈 +5.20 (+2.14%)

Open: $244.00
High: $251.30
Low: $243.50
Prev Close: $243.30
Volume: 98.2M
Market Cap: $789.5B
```

### `!info` / `!i` — Company fundamentals

```
!info NVDA
```
```
📋 NVIDIA Corporation (NVDA)

Sector: Technology
Industry: Semiconductors

P/E Ratio: 65.32
EPS: $7.59
Market Cap: $1.22T
Dividend Yield: 0.03%

52W High: $505.48
52W Low: $222.97
```

### `!market` / `!m` — Major indices

```
!market
```
```
📈 Market Overview

🟢 S&P 500: 5,123.41 (+0.75%)
🟢 Dow Jones: 38,654.42 (+0.51%)
🟢 Nasdaq: 16,156.33 (+1.12%)
🔴 Russell 2000: 2,012.75 (-0.23%)
😰 VIX: 13.25 (-2.15%)
```

### `!crypto` / `!c` — Top cryptocurrencies

```
!crypto
```
```
🪙 Crypto Overview

🟢 Bitcoin: $67,234.50 (+2.15%)
🟢 Ethereum: $3,456.78 (+1.89%)
🔴 Solana: $98.45 (-0.32%)
🟢 XRP: $0.5234 (+3.21%)
🟢 Dogecoin: $0.0821 (+1.45%)
```

### `!status` — Provider health

```
!status
```
```
🔧 Provider Status

✅ yahoo: Ready
✅ alphavantage: Ready
```

Or when rate-limited:

```
🔧 Provider Status

✅ yahoo: Ready
❌ alphavantage: ⏳ Rate limited (3420s)
```

### `!help` — Command help

```
!help
```
```
📖 Stock Bot Commands

!price - Get current stock price
!quote - Get detailed stock quote
!info - Get company fundamentals
!market - Get major market indices
!crypto - Get top cryptocurrency prices
!status - Show provider status
!help - Show available commands

💡 Tip: Type $AAPL in any message for quick lookup
Type !help <command> for detailed usage
```

Detailed help for a specific command:

```
!help price
```
```
📖 !price
Aliases: p, pr, $

Get current stock price

Usage: !price AAPL [MSFT GOOGL ...]
```

## Command Aliases

All commands have short aliases for quick access:

| Command | Aliases |
|---------|---------|
| `!price` | `!p`, `!pr`, `!$` |
| `!quote` | `!q`, `!detail` |
| `!info` | `!i`, `!fundamentals`, `!fund` |
| `!market` | `!m`, `!indices` |
| `!crypto` | `!c`, `!coins` |
| `!status` | `!providers`, `!health` |
| `!help` | `!h`, `!?`, `!commands` |

### Inline Symbol Detection

You can mention stock symbols with `$` anywhere in a message:

```
What do you think about $AAPL?
```
```
Apple Inc. (AAPL)
💵 $185.92
📈 +2.34 (+1.27%)
📊 Vol: 52.3M
```

Multiple symbols work too:

```
Comparing $MSFT and $GOOGL today
```
```
🟢 MSFT: $378.91 (+0.89%)
🔴 GOOGL: $141.80 (-0.32%)
```

### @ Mentions

Tag the bot in a group chat to get help:

```
@StockBot
```
```
👋 Hey! I'm Stock Bot.

Try these:
• !price AAPL - Get stock price
• !crypto - Top cryptocurrencies
• $AAPL - Quick lookup
• !help - All commands
```

You can also include symbols when mentioning:

```
@StockBot what's $TSLA doing?
```
```
Tesla, Inc. (TSLA)
💵 $248.50
📈 +5.20 (+2.14%)
📊 Vol: 98.2M
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGNAL_PHONE_NUMBER` | Yes | — | Bot's Signal phone number (E.164 format) |
| `SIGNAL_API_URL` | No | `http://localhost:8080` | signal-cli-rest-api URL |
| `COMMAND_PREFIX` | No | `!` | Command prefix character |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ALPHAVANTAGE_API_KEY` | No | — | Alpha Vantage API key for additional data |
| `POLYGON_API_KEY` | No | — | Polygon.io API key |
| `YAHOO_PRIORITY` | No | `0` | Yahoo Finance priority (lower = higher) |
| `ALPHAVANTAGE_PRIORITY` | No | `10` | Alpha Vantage priority |
| `POLYGON_PRIORITY` | No | `5` | Polygon priority |

### Provider Priority

Providers are tried in priority order (lowest number first). When a provider fails or hits rate limits, the next provider is tried automatically.

Default order:
1. Yahoo Finance (priority 0) — Free, no API key, unlimited
2. Polygon.io (priority 5) — If configured
3. Alpha Vantage (priority 10) — If configured

### Adding API Keys

**Alpha Vantage** (free tier: 25 requests/day):
1. Get key at https://www.alphavantage.co/support/#api-key
2. Add to `.env`: `ALPHAVANTAGE_API_KEY=your_key_here`

**Polygon.io** (free tier available):
1. Sign up at https://polygon.io/
2. Add to `.env`: `POLYGON_API_KEY=your_key_here`

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Signal Network                         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              signal-cli-rest-api (Docker)                │
│                                                          │
│  • Handles Signal protocol                               │
│  • Manages encryption keys                               │
│  • Sends webhooks to stock-bot                          │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP webhook
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   stock-bot (Docker)                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Webhook   │─▶│  Dispatcher  │─▶│   Commands    │  │
│  │   Handler   │  │              │  │               │  │
│  └─────────────┘  └──────────────┘  └───────┬───────┘  │
│                                              │          │
│                                              ▼          │
│                                    ┌─────────────────┐  │
│                                    │    Provider     │  │
│                                    │    Manager      │  │
│                                    └────────┬────────┘  │
│                                              │          │
│                    ┌─────────────────────────┼─────┐    │
│                    ▼              ▼          ▼     │    │
│              ┌─────────┐  ┌────────────┐  ┌─────┐ │    │
│              │  Yahoo  │  │   Alpha    │  │ ... │ │    │
│              │ Finance │  │  Vantage   │  │     │ │    │
│              └─────────┘  └────────────┘  └─────┘ │    │
│                                                   │    │
└───────────────────────────────────────────────────┴────┘
```

## Project Structure

```
signal-stock-bot/
├── docker-compose.yml      # Full stack deployment
├── Dockerfile              # Bot container
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── requirements-test.txt   # Test dependencies
├── pytest.ini              # Test configuration
│
├── docs/
│   ├── INFRASTRUCTURE.md   # Server setup guide
│   └── DESIGN.md           # Technical design doc
│
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point
│   ├── config.py           # Configuration
│   ├── server.py           # Flask webhook server
│   │
│   ├── providers/          # Financial data providers
│   │   ├── __init__.py
│   │   ├── base.py         # Provider interface
│   │   ├── manager.py      # Fallback logic
│   │   ├── yahoo.py        # Yahoo Finance
│   │   └── alphavantage.py # Alpha Vantage
│   │
│   ├── commands/           # Bot commands
│   │   ├── __init__.py
│   │   ├── base.py         # Command interface
│   │   ├── dispatcher.py   # Message routing
│   │   └── stock_commands.py
│   │
│   └── signal/             # Signal integration
│       ├── __init__.py
│       └── handler.py
│
├── tests/
│   ├── conftest.py
│   ├── test_providers.py
│   ├── test_commands.py
│   └── test_integration.py
│
├── data/                   # Signal-cli data (created at runtime)
│   └── signal-cli/
│
└── logs/                   # Application logs
    └── bot.log
```

## Development

### Local setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_commands.py -v

# Skip integration tests (require network)
pytest -m "not integration"
```

### Running locally without Docker

```bash
# Start signal-cli-rest-api separately (see docs/INFRASTRUCTURE.md)

# Set environment
export SIGNAL_API_URL=http://localhost:8080
export SIGNAL_PHONE_NUMBER=+15551234567
export LOG_LEVEL=DEBUG

# Run bot
python -m src.main
```

### Adding a new provider

1. Create `src/providers/newprovider.py`:

```python
from .base import BaseProvider, Quote, ProviderCapability

class NewProvider(BaseProvider):
    name = "newprovider"
    capabilities = {ProviderCapability.QUOTE}
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_quote(self, symbol: str) -> Quote:
        # Implement API call
        pass
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        # Implement batch call
        pass
    
    async def health_check(self) -> bool:
        # Implement health check
        pass
```

2. Register in `src/config.py`:

```python
# In Config.from_env()
new_key = os.getenv("NEWPROVIDER_API_KEY")
if new_key:
    providers.append(ProviderConfig(
        name="newprovider",
        enabled=True,
        api_key=new_key,
        priority=int(os.getenv("NEWPROVIDER_PRIORITY", "5")),
    ))
```

3. Add to `src/main.py`:

```python
from .providers import NewProvider

# In create_provider_manager()
elif provider_config.name == "newprovider":
    manager.add_provider(NewProvider(provider_config.api_key))
```

### Adding a new command

1. Create command class in `src/commands/stock_commands.py`:

```python
class NewCommand(BaseCommand):
    name = "newcmd"
    aliases = ["nc"]
    description = "Does something new"
    usage = "!newcmd <arg>"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        # Implement command logic
        return CommandResult.ok("Response text")
```

2. Register in `src/main.py`:

```python
# In create_dispatcher()
new_cmd = NewCommand(provider_manager)
dispatcher.register(new_cmd)

# Add to help command list
help_cmd = HelpCommand([..., new_cmd])
```

## Maintenance

### Update signal-cli (monthly recommended)

Signal protocol changes require keeping signal-cli updated:

```bash
cd signal-stock-bot
docker compose pull
docker compose up -d
docker compose logs -f signal-api  # Watch for errors
```

### Backup Signal credentials

Your Signal identity is stored in `./data/signal-cli/`. Back this up securely:

```bash
tar -czvf signal-backup-$(date +%Y%m%d).tar.gz ./data/signal-cli
```

### View logs

```bash
# All logs
docker compose logs -f

# Just bot logs
docker compose logs -f stock-bot

# Or directly
tail -f logs/bot.log
```

### Check health

```bash
# Signal API
curl http://localhost:8080/v1/health

# Stock bot
curl http://localhost:5000/health
```

## Troubleshooting

### Bot not responding

1. Check containers are running: `docker compose ps`
2. Check Signal API health: `curl http://localhost:8080/v1/health`
3. Check webhook is configured: `docker compose logs signal-api | grep webhook`
4. Check bot logs: `docker compose logs stock-bot`

### "Account not found" error

Your Signal account isn't linked. Re-run the QR code linking process.

### Rate limit errors

The bot handles rate limits automatically by switching providers. If you're seeing frequent rate limits:
- Add more providers (Alpha Vantage, Polygon)
- Reduce query frequency
- Check `!status` to see which providers are rate-limited

### Messages delayed

If using `MODE=normal` instead of `MODE=json-rpc`, switch to json-rpc for faster responses (sub-second vs 3-5 seconds).

### Symbol not found

- Check the symbol is valid (US exchanges primarily)
- Try the full symbol (e.g., `BRK.B` not `BRKB`)
- Crypto uses different symbols (`BTC-USD`, `ETH-USD`)

### Group messages not working

Signal GroupV2 requires periodic sync. The `AUTO_RECEIVE_SCHEDULE` setting handles this, but you can force sync:

```bash
curl "http://localhost:8080/v1/receive/+15551234567"
```

## Supported Symbols

The bot supports any symbol available through Yahoo Finance:

- **US Stocks**: `AAPL`, `MSFT`, `GOOGL`, `TSLA`, etc.
- **ETFs**: `SPY`, `QQQ`, `VTI`, `ARKK`, etc.
- **Indices**: `^GSPC` (S&P 500), `^DJI` (Dow), `^IXIC` (Nasdaq)
- **Crypto**: `BTC-USD`, `ETH-USD`, `SOL-USD`
- **Forex**: `EURUSD=X`, `GBPUSD=X`
- **International**: `TSM` (Taiwan Semi), `BABA` (Alibaba)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [signal-cli](https://github.com/AsamK/signal-cli) — Signal protocol implementation
- [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) — REST wrapper
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
