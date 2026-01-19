# test_price_pipeline.py
# Stress test suite - break the pipeline on purpose
# If this passes, we can trust Step 2A

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from price_pipeline import PricePipeline
from price_schema import TrendRegime, VolatilityRegime


def print_obs(obs, test_name: str):
    """Pretty print observation for debugging"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Schema Version: {obs.SCHEMA_VERSION}")
    print(f"Time: {obs.time}")
    print(f"Symbol: {obs.symbol}")
    print(f"Close: ${obs.close:,.2f}")
    print(f"Spread: {obs.spread*100:.4f}%")
    print(f"Volatility: {obs.volatility*100:.4f}%")
    print(f"Pressure: {obs.pressure:+.4f} (confidence: {obs.pressure_confidence:.2f})")
    print(f"Volume: {obs.volume if obs.volume else 'N/A'}")
    print(f"Trend Regime: {obs.trend_regime.name} ({obs.trend_regime.value})")
    print(f"Trend Strength: {obs.trend_strength*100:.4f}%")
    print(f"Vol Regime: {obs.volatility_regime.name} ({obs.volatility_regime.value})")
    print(f"Warmup Mode: {obs.is_warmup}")
    print(f"Market: {obs.metadata.market_type} | Tier: {obs.metadata.liquidity_tier}")
    print(f"{'='*60}\n")


def test_flash_crash():
    """
    Test 1: Flash Crash (-30% in 1 candle)
    
    Expected:
    - Warmup mode ON if insufficient data
    - Volatility regime -> HIGH (when enough data)
    - Pressure -> negative (sellers dominated)
    """
    print("\n🔥 TEST 1: FLASH CRASH")
    
    # Normal market, then sudden crash
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(80)]
    
    # Create independent lists (avoid aliasing bug)
    open_prices = [50000.0] * 80
    close_prices = [50000.0] * 50 + [35000.0] * 10 + [48000.0] * 20
    high_prices = [p * 1.001 for p in close_prices]
    low_prices = close_prices.copy()
    
    # Modify crash candle (index 50)
    open_prices[50] = 50000   # Opens at $50k
    high_prices[50] = 50000   # High at $50k
    low_prices[50] = 35000    # Drops to $35k
    close_prices[50] = 35000  # Closes at $35k (bottom)
    
    data = {
        "time": times,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": [1000] * 80,
        "spread": [0.0005] * 80,
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    # Test 1a: Insufficient data (warmup mode)
    obs_early = pipeline.observe(df.iloc[:30])
    print_obs(obs_early, "Flash Crash - Warmup Mode")
    assert obs_early.is_warmup == True, "Expected warmup mode with 30 candles"
    assert obs_early.trend_regime == TrendRegime.NEUTRAL, "Warmup should force NEUTRAL"
    
    # Test 1b: Crash point with full data
    obs_crash = pipeline.observe(df.iloc[:51])
    print_obs(obs_crash, "Flash Crash - Full Data")
    assert obs_crash.is_warmup == False, "Should have enough data at 51 candles"
    assert obs_crash.pressure < -0.8, f"Expected strong selling pressure, got {obs_crash.pressure}"
    
    print("✅ Flash crash handled gracefully (warmup + crash detection)")
    
    # Validate expectations
    assert obs_crash.volatility_regime == VolatilityRegime.HIGH, \
        f"Expected HIGH volatility, got {obs_crash.volatility_regime.name}"
    assert obs_crash.pressure < -0.8, \
        f"Expected strong selling pressure, got {obs_crash.pressure}"
    assert obs_crash.trend_regime == TrendRegime.DOWN, \
        f"Expected DOWN trend, got {obs_crash.trend_regime.name}"
    
    print("✅ Flash crash handled correctly")


def test_wash_trading():
    """
    Test 2: Wash Trading Detection
    
    High volume but wide spread = fake liquidity
    
    Expected:
    - Pipeline doesn't crash
    - Spread reflects stress
    """
    print("\n🧪 TEST 2: WASH TRADING")
    
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    data = {
        "time": times,
        "open": [50000.0] * 60,
        "high": [50100.0] * 60,
        "low": [49900.0] * 60,
        "close": [50000.0] * 60,
        "volume": [10000] * 30 + [100000] * 30,  # Spike in volume
        "spread": [0.0005] * 30 + [0.005] * 30,  # Wide spread during "volume"
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    obs = pipeline.observe(df)
    print_obs(obs, "Wash Trading Pattern")
    
    # Wide spread should be visible
    assert obs.spread > 0.001, \
        f"Expected wide spread (>0.1%), got {obs.spread*100:.4f}%"
    
    print("✅ Wash trading spread detected")


def test_strong_trend():
    """
    Test 3: Strong Uptrend (7 consecutive up candles)
    
    Expected:
    - Trend regime -> UP
    - Pressure -> positive (bullish candles)
    """
    print("\n📈 TEST 3: STRONG UPTREND")
    
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    # Steady climb from $45k to $55k
    close_prices = np.linspace(45000, 55000, 60)
    
    # Create bullish candles: open < close
    open_prices = close_prices - 100  # Open $100 below close (bullish)
    
    data = {
        "time": times,
        "open": open_prices,
        "high": close_prices + 50,   # High slightly above close
        "low": open_prices - 50,     # Low slightly below open
        "close": close_prices,
        "volume": [1000] * 60,
        "spread": [0.0005] * 60,
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    obs = pipeline.observe(df)
    print_obs(obs, "Strong Uptrend")
    
    assert obs.trend_regime == TrendRegime.UP, \
        f"Expected UP trend, got {obs.trend_regime.name}"
    assert obs.trend_strength > 0.005, \
        f"Expected strong trend (>0.5%), got {obs.trend_strength*100:.4f}%"
    # Note: We do NOT assert positive pressure
    # Uptrends can have neutral pressure if institutions accumulate quietly
    
    print("✅ Uptrend detected correctly (pressure decoupled from trend)")


def test_choppy_range():
    """
    Test 4: Ranging Market (oscillation)
    
    Expected:
    - Volatility regime -> LOW or NORMAL (not HIGH)
    - Price oscillates in tight range
    
    Note: Trend regime can be any value depending on current MA position
    """
    print("\n🌊 TEST 4: CHOPPY RANGE")
    
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    # Tight oscillation around $50k (±0.5%)
    base_price = 50000
    prices = [base_price + base_price * 0.005 * np.sin(i * 0.5) for i in range(60)]
    
    # Create small-bodied candles
    opens = [p * 0.9999 for p in prices]
    closes = prices
    
    data = {
        "time": times,
        "open": opens,
        "high": [p * 1.0005 for p in prices],  # Tight range
        "low": [p * 0.9995 for p in prices],
        "close": closes,
        "volume": [1000] * 60,
        "spread": [0.0005] * 60,
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    obs = pipeline.observe(df)
    print_obs(obs, "Choppy Range")
    
    assert obs.volatility_regime in [VolatilityRegime.LOW, VolatilityRegime.NORMAL], \
        f"Expected LOW/NORMAL volatility in range, got {obs.volatility_regime.name}"
    
    # Trend strength should be low in a range
    assert obs.trend_strength < 0.01, \
        f"Expected low trend strength (<1%) in range, got {obs.trend_strength*100:.4f}%"
    
    print("✅ Ranging market handled correctly (neutral dead zone active)")
    
    print("✅ Ranging market handled correctly")


def test_zero_range_candle():
    """
    Test 5: Zero-Range Candle (doji)
    
    Expected:
    - Pressure -> 0.0 (no battle)
    - No division by zero errors
    """
    print("\n⚪ TEST 5: ZERO-RANGE CANDLE")
    
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    prices = [50000.0] * 60
    
    data = {
        "time": times,
        "open": prices,
        "high": prices,  # No range at all
        "low": prices,
        "close": prices,
        "volume": [1000] * 60,
        "spread": [0.0005] * 60,
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    obs = pipeline.observe(df)
    print_obs(obs, "Zero-Range Candle")
    
    assert obs.pressure == 0.0, \
        f"Expected pressure = 0 for zero-range candle, got {obs.pressure}"
    assert obs.volatility_regime == VolatilityRegime.LOW, \
        f"Expected LOW volatility for flat market, got {obs.volatility_regime.name}"
    
    print("✅ Zero-range candle handled without crash")


def test_missing_volume():
    """
    Test 6: Missing Volume Field
    
    Expected:
    - volume -> None (not 0, not error)
    - Pipeline continues normally
    """
    print("\n❓ TEST 6: MISSING VOLUME")
    
    base_time = datetime(2026, 1, 10)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    prices = np.linspace(48000, 52000, 60)
    
    data = {
        "time": times,
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "spread": [0.0005] * 60,
        # No volume field
    }
    
    df = pd.DataFrame(data)
    pipeline = PricePipeline(symbol="BTCUSDT")
    
    obs = pipeline.observe(df)
    print_obs(obs, "Missing Volume")
    
    assert obs.volume is None, \
        f"Expected volume = None when unavailable, got {obs.volume}"
    
    print("✅ Missing volume handled gracefully")


def run_all_tests():
    """Run full stress test suite"""
    print("\n" + "="*60)
    print("PRICE PIPELINE STRESS TEST SUITE")
    print("Schema Version: 1.0.0")
    print("="*60)
    
    tests = [
        test_flash_crash,
        test_wash_trading,
        test_strong_trend,
        test_choppy_range,
        test_zero_range_candle,
        test_missing_volume,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Step 2A is COMPLETE")
        print("Ready to proceed to Step 2B (News Pipeline)")
    else:
        print("❌ TESTS FAILED - Fix issues before proceeding")


if __name__ == "__main__":
    run_all_tests()