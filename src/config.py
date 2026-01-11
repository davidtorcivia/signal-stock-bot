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
    log_level: str = "INFO"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Provider settings
    providers: list[ProviderConfig] = field(default_factory=list)
    
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
        
        # Sort by priority
        providers.sort(key=lambda p: p.priority)
        
        config = cls(
            signal_api_url=os.getenv("SIGNAL_API_URL", "http://localhost:8080"),
            signal_phone_number=os.getenv("SIGNAL_PHONE_NUMBER", ""),
            command_prefix=os.getenv("COMMAND_PREFIX", "!"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            providers=providers,
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
