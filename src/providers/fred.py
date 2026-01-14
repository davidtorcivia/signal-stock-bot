"""
FRED (Federal Reserve Economic Data) provider.

Free API for economic indicators like CPI, GDP, unemployment, etc.
Get API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import logging
from datetime import datetime
from typing import Optional
import asyncio

from .base import (
    BaseProvider,
    EconomyIndicator,
    ProviderCapability,
    ProviderError,
    RateLimitError,
    SharedSession,
    Quote,
)

logger = logging.getLogger(__name__)


# Map user-friendly indicator names to FRED series IDs
INDICATOR_MAPPING = {
    # Inflation/Prices
    "CPI": "CPIAUCSL",           # Consumer Price Index for All Urban Consumers
    "CORECPI": "CPILFESL",       # Core CPI (excluding food and energy)
    "INFLATION": "T10YIE",       # 10-Year Breakeven Inflation Rate
    "PCE": "PCEPI",              # Personal Consumption Expenditures Price Index
    
    # Employment
    "UNEMPLOYMENT": "UNRATE",    # Unemployment Rate
    "JOBS": "PAYEMS",            # All Employees, Total Nonfarm
    "JOBLESS": "ICSA",           # Initial Jobless Claims
    "LABORFORCE": "CIVPART",     # Labor Force Participation Rate
    
    # GDP/Growth
    "GDP": "GDP",                # Gross Domestic Product
    "REALGDP": "GDPC1",          # Real GDP
    "GDPGROWTH": "A191RL1Q225SBEA",  # Real GDP Growth Rate
    
    # Interest Rates
    "FEDFUNDS": "FEDFUNDS",      # Federal Funds Rate
    "PRIME": "DPRIME",           # Bank Prime Loan Rate
    "10Y": "DGS10",              # 10-Year Treasury Rate
    "2Y": "DGS2",                # 2-Year Treasury Rate
    "30Y": "DGS30",              # 30-Year Treasury Rate
    "MORTGAGE": "MORTGAGE30US",  # 30-Year Mortgage Rate
    
    # Debt/Fiscal
    "DEBT": "GFDEBTN",           # Federal Debt: Total Public Debt
    "DEFICIT": "FYFSD",          # Federal Surplus or Deficit
    
    # Consumer/Retail
    "RETAIL": "RSXFS",           # Retail Sales
    "CONSUMER": "UMCSENT",       # Consumer Sentiment
    "CONFIDENCE": "CSCICP03USM665S",  # Consumer Confidence
    
    # Housing
    "HOUSING": "HOUST",          # Housing Starts
    "HOMEPRICE": "CSUSHPISA",    # Case-Shiller Home Price Index
    
    # Manufacturing
    "PMI": "MANEMP",             # Manufacturing Employment (proxy)
    "INDUSTRIAL": "INDPRO",      # Industrial Production Index
}

# Period mapping for different series
PERIOD_MAP = {
    "CPIAUCSL": "monthly",
    "CPILFESL": "monthly",
    "UNRATE": "monthly",
    "PAYEMS": "monthly",
    "GDP": "quarterly",
    "GDPC1": "quarterly",
    "FEDFUNDS": "monthly",
    "DGS10": "daily",
    "DGS2": "daily",
    "DGS30": "daily",
    "MORTGAGE30US": "weekly",
    "GFDEBTN": "quarterly",
    "ICSA": "weekly",
}

# Unit mapping
UNIT_MAP = {
    "CPIAUCSL": " (Index)",
    "CPILFESL": " (Index)",
    "UNRATE": "%",
    "PAYEMS": "K jobs",
    "GDP": "B USD",
    "GDPC1": "B USD",
    "FEDFUNDS": "%",
    "DGS10": "%",
    "DGS2": "%",
    "DGS30": "%",
    "MORTGAGE30US": "%",
    "GFDEBTN": "M USD",
    "ICSA": " claims",
    "T10YIE": "%",
    "UMCSENT": " (Index)",
    "HOUST": "K units",
    "INDPRO": " (Index)",
}


class FredProvider(BaseProvider):
    """
    Provider for FRED (Federal Reserve Economic Data).
    
    Free API with generous rate limits (120 requests/minute).
    Specializes in economic indicators.
    """
    
    name = "fred"
    capabilities = {ProviderCapability.ECONOMY}
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_quote(self, symbol: str) -> Quote:
        """Not supported - FRED is for economic data only."""
        raise NotImplementedError("FRED doesn't support stock quotes")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Not supported - FRED is for economic data only."""
        raise NotImplementedError("FRED doesn't support stock quotes")
    
    async def get_economy_data(self, indicator: str) -> EconomyIndicator:
        """
        Get economic indicator data from FRED.
        
        Args:
            indicator: User-friendly name (CPI, UNEMPLOYMENT, GDP, etc.)
                      or raw FRED series ID
        
        Returns:
            EconomyIndicator with latest value
        """
        indicator = indicator.upper()
        
        # Map to FRED series ID
        series_id = INDICATOR_MAPPING.get(indicator, indicator)
        
        try:
            data = await self._fetch_series(series_id)
            
            if not data.get("observations"):
                raise ProviderError(f"No data found for {indicator}")
            
            # Get the most recent observation
            observations = data["observations"]
            
            # Find the most recent non-empty value
            latest = None
            previous = None
            for i, obs in enumerate(reversed(observations)):
                if obs.get("value") and obs["value"] != ".":
                    if latest is None:
                        latest = obs
                    elif previous is None:
                        previous = obs
                        break
            
            if not latest:
                raise ProviderError(f"No valid data for {indicator}")
            
            # Parse the value
            try:
                value = float(latest["value"])
            except ValueError:
                raise ProviderError(f"Invalid value for {indicator}: {latest['value']}")
            
            # Parse date
            obs_date = datetime.strptime(latest["date"], "%Y-%m-%d")
            
            # Get previous value if available
            prev_value = None
            if previous and previous.get("value") and previous["value"] != ".":
                try:
                    prev_value = float(previous["value"])
                except ValueError:
                    pass
            
            # Get display name from metadata
            name = data.get("seriess", [{}])[0].get("title", indicator) if "seriess" in data else indicator
            
            return EconomyIndicator(
                name=name if len(name) < 50 else indicator,
                value=value,
                unit=UNIT_MAP.get(series_id, ""),
                date=obs_date,
                period=PERIOD_MAP.get(series_id, "varies"),
                provider=self.name,
                previous=prev_value,
            )
            
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"FRED error for {indicator}: {e}")
            raise ProviderError(f"Failed to fetch {indicator}: {e}")
    
    async def _fetch_series(self, series_id: str, limit: int = 5) -> dict:
        """Fetch series data from FRED API."""
        session = SharedSession.get()
        
        # Fetch observations (latest N)
        obs_url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        async with session.get(obs_url, params=params) as resp:
            if resp.status == 429:
                raise RateLimitError(60)
            
            if resp.status == 400:
                error = await resp.text()
                if "Bad Request" in error:
                    raise ProviderError(f"Unknown indicator: {series_id}")
                raise ProviderError(f"FRED API error: {error}")
            
            if resp.status != 200:
                raise ProviderError(f"FRED API returned {resp.status}")
            
            return await resp.json()
    
    async def get_economy_historical(
        self,
        indicator: str,
        period: str = "5y"
    ) -> tuple[list[tuple[datetime, float]], str, str]:
        """
        Get historical time series for an economic indicator.
        
        Args:
            indicator: User-friendly name (CPI, UNEMPLOYMENT, etc.) or FRED series ID
            period: Time period - 1y, 2y, 5y, 10y, max
        
        Returns:
            Tuple of (data_points, series_name, unit)
            where data_points is list of (datetime, value) tuples
        """
        indicator = indicator.upper()
        series_id = INDICATOR_MAPPING.get(indicator, indicator)
        
        # Map period to number of observations to fetch
        # Most FRED series are monthly, so 12 points per year
        period_map = {
            "1y": 15,      # ~1 year of monthly data
            "2y": 30,      # ~2 years
            "5y": 65,      # ~5 years
            "10y": 125,    # ~10 years
            "max": 500,    # Max available
        }
        limit = period_map.get(period.lower(), 65)
        
        try:
            data = await self._fetch_series(series_id, limit=limit)
            
            if not data.get("observations"):
                raise ProviderError(f"No historical data for {indicator}")
            
            # Parse observations into (date, value) tuples
            points = []
            for obs in data["observations"]:
                if obs.get("value") and obs["value"] != ".":
                    try:
                        date = datetime.strptime(obs["date"], "%Y-%m-%d")
                        value = float(obs["value"])
                        points.append((date, value))
                    except (ValueError, KeyError):
                        continue
            
            if not points:
                raise ProviderError(f"No valid data points for {indicator}")
            
            # Sort by date (ascending for charting)
            points.sort(key=lambda x: x[0])
            
            # Get series name
            name = indicator
            unit = UNIT_MAP.get(series_id, "")
            
            return points, name, unit
            
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"FRED historical error for {indicator}: {e}")
            raise ProviderError(f"Failed to fetch historical data: {e}")
    
    async def health_check(self) -> bool:
        """Check if FRED API is accessible."""
        try:
            session = SharedSession.get()
            url = f"{self.BASE_URL}/series"
            params = {
                "series_id": "GDP",
                "api_key": self.api_key,
                "file_type": "json",
            }
            
            async with session.get(url, params=params) as resp:
                return resp.status == 200
                
        except Exception as e:
            logger.error(f"FRED health check failed: {e}")
            return False

