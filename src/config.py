"""
Configuration management.

Supports loading from:
- Environment variables (default)
- YAML config file
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    priority: int = 0  # Lower = higher priority


@dataclass
class Config:
    """Application configuration"""
    
    # Signal settings
    signal_api_url: str = "http://localhost:8080"
    signal_phone_number: str = ""
    
    # Bot settings
    command_prefix: str = "!"
    bot_name: str = "Stock Bot"
    log_level: str = "INFO"
    
    # Polygon/Massive settings
    massive_pro: bool = False  # True if user has paid Polygon plan (enables options/economy)
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Provider settings
    providers: list[ProviderConfig] = field(default_factory=list)
    
    # Database settings
    watchlist_db_path: str = "data/watchlist.db"
    
    # Admin settings
    admin_numbers: list[str] = field(default_factory=list)  # Phone numbers for admin access
    user_rate_limit: int = 30  # Max requests per minute per user
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        providers = []
        
        # Yahoo Finance (no key required, always available)
        yahoo_enabled = os.getenv("YAHOO_ENABLED", "true").lower() == "true"
        if yahoo_enabled:
            providers.append(ProviderConfig(
                name="yahoo",
                enabled=True,
                priority=int(os.getenv("YAHOO_PRIORITY", "0")),
            ))
        
        # Alpha Vantage (requires API key)
        av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
        if av_key:
            providers.append(ProviderConfig(
                name="alphavantage",
                enabled=True,
                api_key=av_key,
                priority=int(os.getenv("ALPHAVANTAGE_PRIORITY", "10")),
            ))
        
        # Polygon (requires API key)
        polygon_key = os.getenv("POLYGON_API_KEY", "").strip()
        if polygon_key:
            providers.append(ProviderConfig(
                name="polygon",
                enabled=True,
                api_key=polygon_key,
                priority=int(os.getenv("POLYGON_PRIORITY", "5")),
            ))
        
        # Finnhub (requires API key, 60 calls/min free)
        finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
        if finnhub_key:
            providers.append(ProviderConfig(
                name="finnhub",
                enabled=True,
                api_key=finnhub_key,
                priority=int(os.getenv("FINNHUB_PRIORITY", "15")),
            ))
        
        # Twelve Data (requires API key, 800 calls/day free)
        twelvedata_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
        if twelvedata_key:
            providers.append(ProviderConfig(
                name="twelvedata",
                enabled=True,
                api_key=twelvedata_key,
                priority=int(os.getenv("TWELVEDATA_PRIORITY", "20")),
            ))
        
        # FRED (requires API key, 120 calls/min free - for economic data)
        fred_key = os.getenv("FRED_API_KEY", "").strip()
        if fred_key:
            providers.append(ProviderConfig(
                name="fred",
                enabled=True,
                api_key=fred_key,
                priority=int(os.getenv("FRED_PRIORITY", "1")),  # High priority for economy data
            ))
        
        # Sort by priority
        providers.sort(key=lambda p: p.priority)
        
        config = cls(
            signal_api_url=os.getenv("SIGNAL_API_URL", "http://localhost:8080"),
            signal_phone_number=os.getenv("SIGNAL_PHONE_NUMBER", ""),
            command_prefix=os.getenv("COMMAND_PREFIX", "!"),
            bot_name=os.getenv("BOT_NAME", "Stock Bot"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            massive_pro=os.getenv("MASSIVE_PRO", "false").lower() == "true",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            providers=providers,
            watchlist_db_path=os.getenv("WATCHLIST_DB_PATH", "data/watchlist.db"),
            admin_numbers=[n.strip() for n in os.getenv("ADMIN_NUMBERS", "").split(",") if n.strip()],
            user_rate_limit=int(os.getenv("USER_RATE_LIMIT", "30")),
        )
        
        logger.info(f"Loaded config: {len(providers)} providers configured")
        return config
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        providers_data = data.pop("providers", [])
        providers = []
        
        for p in providers_data:
            # Resolve environment variables in api_key
            api_key = p.get("api_key", "")
            if api_key and api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var, "")
            
            providers.append(ProviderConfig(
                name=p["name"],
                enabled=p.get("enabled", True),
                api_key=api_key if api_key else None,
                priority=p.get("priority", 0),
            ))
        
        providers.sort(key=lambda p: p.priority)
        
        return cls(providers=providers, **data)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.signal_phone_number:
            errors.append("SIGNAL_PHONE_NUMBER is required")
        
        if not self.providers:
            errors.append("At least one provider must be configured")
        
        return errors
