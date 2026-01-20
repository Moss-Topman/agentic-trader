# market_observation.py
# Pure container for unified market observation
# No logic. No fusion. No cleverness.

from dataclasses import dataclass
from price_schema import PriceObservation
from news_schema import NewsObservation


@dataclass(frozen=True)
class MarketObservation:
    """
    Unified market observation combining price and news perception.
    
    Design principles:
    - Price drives decisions
    - News provides context
    - Agent learns interaction through reward, not hardcoded rules
    
    This is a container, not a brain.
    No methods. No logic. No opinions.
    
    Integration contract:
    - Price and news are observed independently
    - Timestamps must align
    - Warmup states are preserved
    - No data fusion happens here
    
    Usage in future environment:
        price_obs = price_pipeline.observe(candle_df)
        news_obs = news_pipeline.observe(timestamp)
        
        market_obs = MarketObservation(
            price=price_obs,
            news=news_obs
        )
        
        # Agent sees both channels
        # Reward function teaches relationships
        # No manual coupling
    """
    
    price: PriceObservation
    news: NewsObservation
    
    # No other fields. No computed properties. Nothing.