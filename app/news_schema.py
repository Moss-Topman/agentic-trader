# news_schema.py
# FROZEN SCHEMA v1.0.0 - DO NOT MODIFY WITHOUT VERSION BUMP
# Any change to this file = breaking change = major version increment

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal
from enum import IntEnum


class EventScope(IntEnum):
    """
    Geographic/market scope of event impact.
    
    Determines correlation structure and contagion risk.
    Agent learns different behavior based on scope.
    """
    BTC_ONLY = 0      # BTC-specific (e.g., BTC ETF decision, mining difficulty)
    CRYPTO_WIDE = 1   # All crypto affected (e.g., SEC crypto crackdown, Tether FUD)
    GLOBAL_RISK = 2   # Macro risk-off (e.g., Fed emergency meeting, banking crisis)


@dataclass(frozen=True)
class NewsObservation:
    """
    Contextual risk and regime modifier for autonomous trading agents.
    
    Design principles:
    - Price remains primary. News NEVER overrides price.
    - News can only: reduce confidence, increase risk awareness, flag abnormality
    - No predictions. No opinions. No vibes.
    - Risk-aware, not sentiment-aware
    
    Philosophy:
    "Is something happening outside price that changes how much price signals 
    should be trusted?"
    
    If price is eyes/ears, news is the weather report.
    
    Schema Version: 1.0.0
    Market: Crypto Spot, BTC/USDT
    """
    
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    # === Identifiers ===
    time: datetime       # When observation was made (UTC), not event time
    symbol: str          # Market identifier (e.g., "BTCUSDT")

    # === Core Risk Signals ===
    event_risk: float           # [0, 1] - How dangerous is it to trade now?
                                # 0.0 = business as usual
                                # 0.7 = CPI in 30 minutes
                                # 1.0 = exchange halted withdrawals
    
    shock_flag: bool            # True = abnormal event detected
                                # Binary, conservative trigger
                                # Agent reaction: disable exploration, preserve capital
    
    narrative_intensity: float  # [0, 1] - How loud is the news cycle?
                                # NOT sentiment. Attention density.
                                # High = expect reflexive behavior, volatility expansion
    
    # === Temporal Context ===
    time_decay: float           # [0, 1] - Event freshness
                                # 1.0 = fresh (just happened)
                                # 0.5 = 24hrs old (half-life)
                                # 0.0 = stale (irrelevant)
                                # Computed via: exp(-hours_elapsed / 24)
    
    # === Scope & Relevance ===
    event_scope: EventScope     # Impact boundary
                                # BTC_ONLY: isolated event
                                # CRYPTO_WIDE: sector risk
                                # GLOBAL_RISK: systemic risk
    
    # === Confidence & Quality ===
    confidence: float           # [0, 1] - Trustworthiness of this signal
                                # 1.0 = official, verified (economic calendar)
                                # 0.95 = first-party API (exchange status)
                                # 0.3 = single unverified source (ignore)
    
    is_sparse: bool             # True = low coverage, uncertain data
                                # Same philosophy as warmup mode in price
                                # Agent learns: "News data is thin → ignore it"
    
    # === Classification ===
    event_type: Literal[
        "scheduled",     # Economic calendar (CPI, FOMC, ETF decisions)
        "unscheduled",   # Breaking news (hack, ban, liquidation cascade)
        "regulatory",    # SEC announcements, legal actions
        "exchange",      # Platform status changes (outage, delisting)
        "macro",         # Fed policy, inflation data, banking crisis
        "none"           # No significant events detected
    ]
    
    
# === HARD RULES (NON-NEGOTIABLE) ===
#
# 1. News NEVER encodes direction (bullish/bearish)
#    - Any field implying price prediction violates design
#    - Agent learns interpretation through reward, not schema
#
# 2. No indicators, no sentiment scores, no LLM hallucinations
#    - Only structured, deterministic signals
#    - Garbage filtered before observation creation
#
# 3. No schema changes without version bump
#    - Add field → v1.1.0 (minor)
#    - Change field type/meaning → v2.0.0 (major)
#    - Remove field → v2.0.0 (major)
#
# 4. Observations describe context, not actions, not rewards
#    - event_risk tells agent "how dangerous"
#    - Agent decides what to do about it
#
# 5. Phase 1 data sources ONLY:
#    - Economic calendar (scheduled events)
#    - Exchange status APIs (first-party)
#    - Manual/synthetic test data
#    - NO Twitter, NO Reddit, NO unverified feeds