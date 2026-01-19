# price_schema.py
# FROZEN SCHEMA v1.0.0 - DO NOT MODIFY WITHOUT VERSION BUMP
# Any change to this file = breaking change = major version increment

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, ClassVar, Literal
from enum import IntEnum


class TrendRegime(IntEnum):
    """
    3-state trend classification.
    Derived from smoothed price structure, not single candles.
    """
    DOWN = 0      # Price below MA20, MA20 declining
    NEUTRAL = 1   # Price near MA20 (within threshold)
    UP = 2        # Price above MA20, MA20 rising


class VolatilityRegime(IntEnum):
    """
    Volatility environment classification.
    Tells agent when the world is calm vs dangerous.
    """
    LOW = 0       # < 0.2% rolling std
    NORMAL = 1    # 0.2% - 0.6% rolling std
    HIGH = 2      # > 0.6% rolling std


@dataclass(frozen=True)
class MarketMetadata:
    """
    Explicit market context - never implied.
    Prevents agent from assuming BTC behavior applies everywhere.
    """
    market_type: Literal["crypto_spot"]
    liquidity_tier: Literal["tier1"]  # BTC is always tier1
    session: Literal["asia", "europe", "us", "overlap", "24_7"]


@dataclass(frozen=True)
class PriceObservation:
    """
    Agent's perceptual state of the market.
    
    Design principles:
    - Describes state, not actions, not rewards
    - No indicators stored (RSI, MACD banned)
    - No silent defaults (Optional = None when unavailable)
    - Immutable (frozen=True prevents mutation bugs)
    
    Schema Version: 1.0.0
    Market: Crypto Spot, BTC/USDT, Tier-1
    """
    
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    # === Identifiers ===
    time: datetime        # UTC timestamp, no local time nonsense
    symbol: str           # Market identifier (e.g., "BTCUSDT")

    # === Price Reality ===
    close: float          # Last traded price, raw unfiltered
    spread: float         # Bid-ask spread as % of mid, mandatory

    # === Continuous Dynamics ===
    volatility: float     # Rolling realized volatility (instability)
    pressure: float       # Normalized dominance [-1, 1], who won the candle

    # === Liquidity ===
    volume: Optional[float]  # Participation signal, never trusted alone

    # === Categorical Regimes (Semantic Abstraction) ===
    trend_regime: TrendRegime           # Market phase (DOWN/NEUTRAL/UP)
    volatility_regime: VolatilityRegime # Risk environment (LOW/NORMAL/HIGH)

    # === Confidence & Quality Signals ===
    is_warmup: bool          # True if insufficient history for full analysis
    trend_strength: float    # Magnitude of trend [0, 1], enables dead zone
    pressure_confidence: float  # How much to trust pressure [0, 1]

    # === Context ===
    metadata: MarketMetadata  # Explicit market type and tier