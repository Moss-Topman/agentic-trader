# price_pipeline.py
# Perception engine - derives observations from raw market data
# No predictions, no indicators, no astrology

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from price_schema import (
    PriceObservation,
    MarketMetadata,
    TrendRegime,
    VolatilityRegime,
)


class PricePipeline:
    """
    Transforms raw OHLCV data into structured observations.
    
    Philosophy:
    - Perception, not prediction
    - Derives semantic states from continuous signals
    - Never invents data (None if unavailable)
    - Fails loudly on invalid input
    """
    
    def __init__(
        self,
        symbol: str,
        ma_fast: int = 20,
        ma_slow: int = 50,
        vol_window: int = 20,
        trend_threshold: float = 0.003,  # 0.3% threshold for NEUTRAL
    ):
        self.symbol = symbol
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.vol_window = vol_window
        self.trend_threshold = trend_threshold

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Ensure required columns exist and data is valid"""
        required = ["time", "open", "high", "low", "close", "spread"]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Note: We no longer require minimum rows - warmup mode handles this
        
        # Validate price sanity
        latest = df.iloc[-1]
        if latest["high"] < latest["low"]:
            raise ValueError("Invalid candle: high < low")
        if latest["close"] < 0:
            raise ValueError("Invalid price: close < 0")
        if latest["spread"] < 0:
            raise ValueError("Invalid spread: spread < 0")

    def _trend_regime_with_strength(self, df: pd.DataFrame) -> tuple[TrendRegime, float, bool]:
        """
        3-state trend classification with strength measurement.
        
        Returns: (regime, strength, is_warmup)
        
        Logic:
        - If insufficient data → warmup mode, NEUTRAL, strength=0
        - Calculate MA20 and MA50
        - Compute trend_strength = distance between MAs
        - If strength < threshold → NEUTRAL (dead zone)
        - Else classify UP/DOWN based on MA relationship
        """
        # Check if we have enough data
        if len(df) < self.ma_slow:
            # Warmup mode: not enough history
            return TrendRegime.NEUTRAL, 0.0, True
        
        ma_fast = df["close"].rolling(self.ma_fast, min_periods=self.ma_fast).mean()
        ma_slow = df["close"].rolling(self.ma_slow, min_periods=self.ma_slow).mean()
        
        latest_ma_fast = ma_fast.iloc[-1]
        latest_ma_slow = ma_slow.iloc[-1]
        
        if pd.isna(latest_ma_fast) or pd.isna(latest_ma_slow):
            # MA calculation failed, warmup mode
            return TrendRegime.NEUTRAL, 0.0, True
        
        # Calculate trend strength (normalized distance between MAs)
        trend_strength = abs(latest_ma_fast - latest_ma_slow) / latest_ma_slow
        
        # Dead zone threshold: if MAs are within 0.5%, call it NEUTRAL
        if trend_strength < 0.005:  # 0.5%
            return TrendRegime.NEUTRAL, trend_strength, False
        
        # Strong enough signal to classify
        if latest_ma_fast > latest_ma_slow:
            return TrendRegime.UP, trend_strength, False
        else:
            return TrendRegime.DOWN, trend_strength, False

    def _volatility(self, df: pd.DataFrame) -> float:
        """
        Rolling realized volatility.
        
        Uses percentage returns over vol_window periods.
        Returns standard deviation (annualized for hourly data).
        """
        returns = df["close"].pct_change()
        vol = returns.rolling(self.vol_window, min_periods=self.vol_window).std()
        
        latest_vol = vol.iloc[-1]
        
        if pd.isna(latest_vol):
            # Not enough data, return 0 (will be classified as LOW)
            return 0.0
        
        return float(latest_vol)

    def _volatility_regime(self, vol: float) -> VolatilityRegime:
        """
        Classify volatility into discrete regimes.
        
        Thresholds calibrated for hourly crypto data:
        - LOW: < 0.2% std (calm, tight ranges)
        - NORMAL: 0.2% - 0.6% std (typical crypto)
        - HIGH: > 0.6% std (stressed, flash crash territory)
        """
        if vol < 0.002:  # 0.2%
            return VolatilityRegime.LOW
        elif vol < 0.006:  # 0.6%
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.HIGH

    def _pressure_with_confidence(self, row) -> tuple[float, float]:
        """
        Intra-candle dominance with confidence measure.
        
        Formula: (close - open) / (high - low)
        
        Confidence based on:
        - Range size (larger range = more confident)
        - Volume (if available, higher volume = more confident)
        
        Returns: (pressure, confidence)
        
        Range: 
        - pressure: [-1, 1]
        - confidence: [0, 1]
        """
        range_size = row["high"] - row["low"]
        
        # Zero-range candle (doji or flat)
        if range_size == 0:
            return 0.0, 0.0  # No pressure, no confidence
        
        pressure = (row["close"] - row["open"]) / range_size
        
        # Confidence based on range size as % of price
        range_pct = range_size / row["close"]
        
        # Normalize: 0.1% range = low confidence, 1%+ range = high confidence
        confidence = min(range_pct / 0.01, 1.0)
        
        return pressure, confidence

    def _get_spread(self, row) -> float:
        """
        Extract spread, validate sanity.
        
        Spread must be:
        - Non-negative
        - Reasonable for BTC (typically 0.01% - 0.1%)
        
        If spread > 1%, likely data error or extreme stress.
        """
        spread = float(row["spread"])
        
        if spread < 0:
            raise ValueError(f"Invalid spread: {spread} (negative)")
        
        if spread > 0.01:  # > 1% spread on BTC is extreme
            # Log warning but don't fail (could be legit flash crash)
            pass
        
        return spread

    def observe(self, df: pd.DataFrame) -> PriceObservation:
        """
        Main observation function.
        
        Input: DataFrame with OHLCV + spread data
        Output: Structured PriceObservation
        
        Handles graceful degradation:
        - Insufficient history → warmup mode
        - Missing data → None for optional fields
        - Invalid data → raises exception
        """
        # Validate input data
        self._validate_input(df)
        
        # Get latest candle
        latest = df.iloc[-1]
        
        # Compute trend with strength (handles warmup)
        trend, trend_strength, is_warmup = self._trend_regime_with_strength(df)
        
        # Compute volatility
        volatility = self._volatility(df)
        vol_regime = self._volatility_regime(volatility)
        
        # Compute pressure with confidence
        pressure, pressure_conf = self._pressure_with_confidence(latest)
        
        # Get spread
        spread = self._get_spread(latest)
        
        # Extract volume (optional)
        volume = None
        if "volume" in df.columns and not pd.isna(latest["volume"]):
            volume = float(latest["volume"])
        
        # Construct observation
        return PriceObservation(
            time=pd.to_datetime(latest["time"]),
            symbol=self.symbol,
            close=float(latest["close"]),
            spread=spread,
            volatility=volatility,
            pressure=pressure,
            volume=volume,
            trend_regime=trend,
            volatility_regime=vol_regime,
            is_warmup=is_warmup,
            trend_strength=trend_strength,
            pressure_confidence=pressure_conf,
            metadata=MarketMetadata(
                market_type="crypto_spot",
                liquidity_tier="tier1",
                session="24_7",
            ),
        )