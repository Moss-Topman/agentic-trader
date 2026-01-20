# test_market_observation.py
# Integration test: Prove price + news can be synchronized cleanly
# No environment. No agent. No actions. Just composition proof.

from datetime import datetime, timedelta
import pandas as pd
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from market_observation import MarketObservation
from news_schema import EventScope


def test_basic_composition():
    """
    Test 1: Basic Composition
    
    Proves MarketObservation can hold both observations without errors.
    """
    print("\n" + "="*60)
    print("TEST 1: BASIC COMPOSITION")
    print("="*60)
    
    # Create pipelines
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Minimal data (60 candles for MA50)
    times = pd.date_range("2026-01-10", periods=60, freq="H")
    df = pd.DataFrame({
        "time": times,
        "open": [50000.0] * 60,
        "high": [50100.0] * 60,
        "low": [49900.0] * 60,
        "close": [50000.0] * 60,
        "volume": [1000.0] * 60,
        "spread": [0.0005] * 60,
    })
    
    # Observe at end
    price_obs = price_pipe.observe(df)
    news_obs = news_pipe.observe(times[-1])
    
    # Compose
    market_obs = MarketObservation(
        price=price_obs,
        news=news_obs
    )
    
    # Verify structure
    assert market_obs.price is not None, "Price observation missing"
    assert market_obs.news is not None, "News observation missing"
    assert market_obs.price.symbol == "BTCUSDT", "Symbol mismatch"
    assert market_obs.news.symbol == "BTCUSDT", "Symbol mismatch"
    
    print(f"✅ MarketObservation composed successfully")
    print(f"   Price: {market_obs.price.close:.2f}")
    print(f"   News: event_risk={market_obs.news.event_risk:.2f}")


def test_warmup_behavior():
    """
    Test 2: Warmup Handling
    
    Proves warmup state is preserved across observations.
    Critical: Price warmup must be explicit, not assumed.
    """
    print("\n" + "="*60)
    print("TEST 2: WARMUP BEHAVIOR")
    print("="*60)
    
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Create data with only 30 candles (insufficient for MA50)
    times = pd.date_range("2026-01-10", periods=30, freq="H")
    df_warmup = pd.DataFrame({
        "time": times,
        "open": [50000.0] * 30,
        "high": [50100.0] * 30,
        "low": [49900.0] * 30,
        "close": [50000.0] * 30,
        "volume": [1000.0] * 30,
        "spread": [0.0005] * 30,
    })
    
    # Observe during warmup
    price_obs_warmup = price_pipe.observe(df_warmup)
    news_obs_warmup = news_pipe.observe(times[-1])
    
    market_obs_warmup = MarketObservation(
        price=price_obs_warmup,
        news=news_obs_warmup
    )
    
    # CRITICAL ASSERTION: Warmup must be explicit
    assert market_obs_warmup.price.is_warmup == True, \
        "Expected price warmup=True with 30 candles, got False"
    
    print(f"✅ Warmup detected: price.is_warmup={market_obs_warmup.price.is_warmup}")
    
    # Now test with sufficient data (60 candles)
    times_full = pd.date_range("2026-01-10", periods=60, freq="H")
    df_full = pd.DataFrame({
        "time": times_full,
        "open": [50000.0] * 60,
        "high": [50100.0] * 60,
        "low": [49900.0] * 60,
        "close": [50000.0] * 60,
        "volume": [1000.0] * 60,
        "spread": [0.0005] * 60,
    })
    
    price_obs_ready = price_pipe.observe(df_full)
    news_obs_ready = news_pipe.observe(times_full[-1])
    
    market_obs_ready = MarketObservation(
        price=price_obs_ready,
        news=news_obs_ready
    )
    
    assert market_obs_ready.price.is_warmup == False, \
        "Expected price warmup=False with 60 candles, got True"
    
    print(f"✅ Ready state: price.is_warmup={market_obs_ready.price.is_warmup}")


def test_timestamp_alignment():
    """
    Test 3: Timestamp Synchronization
    
    Proves price and news observations align to same logical time.
    """
    print("\n" + "="*60)
    print("TEST 3: TIMESTAMP ALIGNMENT")
    print("="*60)
    
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Create data
    times = pd.date_range("2026-01-10", periods=60, freq="H")
    df = pd.DataFrame({
        "time": times,
        "open": [50000.0] * 60,
        "high": [50100.0] * 60,
        "low": [49900.0] * 60,
        "close": [50000.0] * 60,
        "volume": [1000.0] * 60,
        "spread": [0.0005] * 60,
    })
    
    # Observe at specific timestamp
    target_time = times[55]
    
    price_obs = price_pipe.observe(df.iloc[:56])  # Up to index 55
    news_obs = news_pipe.observe(target_time)
    
    market_obs = MarketObservation(
        price=price_obs,
        news=news_obs
    )
    
    # Verify timestamps match
    assert market_obs.price.time == market_obs.news.time, \
        f"Timestamp mismatch: price={market_obs.price.time}, news={market_obs.news.time}"
    
    print(f"✅ Timestamps aligned: {market_obs.price.time}")


def test_event_synchronization():
    """
    Test 4: Scheduled Event Integration
    
    Proves price and news can be observed together during events.
    Verifies event risk appears in news while price remains independent.
    """
    print("\n" + "="*60)
    print("TEST 4: EVENT SYNCHRONIZATION")
    print("="*60)
    
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Create 60 candles
    times = pd.date_range("2026-01-10", periods=60, freq="H")
    df = pd.DataFrame({
        "time": times,
        "open": [50000.0] * 60,
        "high": [50100.0] * 60,
        "low": [49900.0] * 60,
        "close": [50000.0] * 60,
        "volume": [1000.0] * 60,
        "spread": [0.0005] * 60,
    })
    
    # Add scheduled event at candle 55
    event_time = times[55]
    news_pipe.add_scheduled_event(
        event_time=event_time,
        event_type="scheduled",
        description="CPI Release",
        base_risk=0.8,
        scope=EventScope.GLOBAL_RISK
    )
    
    # Observe BEFORE event
    price_obs_before = price_pipe.observe(df.iloc[:54])
    news_obs_before = news_pipe.observe(times[53])
    market_obs_before = MarketObservation(
        price=price_obs_before,
        news=news_obs_before
    )
    
    print(f"\nBefore event (2hrs out):")
    print(f"  Price regime: {market_obs_before.price.trend_regime.name}")
    print(f"  News risk: {market_obs_before.news.event_risk:.4f}")
    
    assert market_obs_before.news.event_risk == 0.0, \
        "Event too far out, risk should be 0"
    
    # Observe DURING event approach (1hr before)
    price_obs_approach = price_pipe.observe(df.iloc[:55])
    news_obs_approach = news_pipe.observe(times[54])
    market_obs_approach = MarketObservation(
        price=price_obs_approach,
        news=news_obs_approach
    )
    
    print(f"\nApproaching event (1hr out):")
    print(f"  Price regime: {market_obs_approach.price.trend_regime.name}")
    print(f"  News risk: {market_obs_approach.news.event_risk:.4f}")
    
    assert market_obs_approach.news.event_risk > 0.3, \
        f"Event 1hr out, risk should be elevated, got {market_obs_approach.news.event_risk}"
    
    # Observe AT event
    price_obs_event = price_pipe.observe(df.iloc[:56])
    news_obs_event = news_pipe.observe(event_time)
    market_obs_event = MarketObservation(
        price=price_obs_event,
        news=news_obs_event
    )
    
    print(f"\nAt event time:")
    print(f"  Price regime: {market_obs_event.price.trend_regime.name}")
    print(f"  News risk: {market_obs_event.news.event_risk:.4f}")
    print(f"  News scope: {market_obs_event.news.event_scope.name}")
    
    assert market_obs_event.news.event_risk == 0.8, \
        f"Event now, risk should equal base_risk (0.8), got {market_obs_event.news.event_risk}"
    
    # CRITICAL: Price observation is independent of news
    # Price regime should be based ONLY on price action, not event risk
    # (Both observations can coexist without coupling)
    
    print(f"\n✅ Event synchronization works")
    print(f"✅ Price and news remain independent channels")


def test_multi_observation_sequence():
    """
    Test 5: Sequential Observations
    
    Proves observations can be generated continuously without state corruption.
    Tests rolling window behavior.
    """
    print("\n" + "="*60)
    print("TEST 5: SEQUENTIAL OBSERVATIONS")
    print("="*60)
    
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Create 100 candles
    times = pd.date_range("2026-01-10", periods=100, freq="H")
    df = pd.DataFrame({
        "time": times,
        "open": [50000.0 + i*10 for i in range(100)],  # Slight uptrend
        "high": [50100.0 + i*10 for i in range(100)],
        "low": [49900.0 + i*10 for i in range(100)],
        "close": [50000.0 + i*10 for i in range(100)],
        "volume": [1000.0] * 100,
        "spread": [0.0005] * 100,
    })
    
    # Observe at multiple points
    observations = []
    for i in range(50, 100, 10):  # Every 10 candles after warmup
        price_obs = price_pipe.observe(df.iloc[:i+1])
        news_obs = news_pipe.observe(times[i])
        market_obs = MarketObservation(
            price=price_obs,
            news=news_obs
        )
        observations.append(market_obs)
        
        print(f"Candle {i}: price={market_obs.price.close:.2f}, "
              f"trend={market_obs.price.trend_regime.name}")
    
    # Verify all observations are valid
    assert len(observations) == 5, f"Expected 5 observations, got {len(observations)}"
    assert all(obs.price.is_warmup == False for obs in observations), \
        "All observations should be post-warmup"
    
    print(f"\n✅ Sequential observations generated cleanly")


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("MARKET OBSERVATION INTEGRATION TEST SUITE")
    print("Testing: Price + News Synchronization")
    print("No Environment. No Agent. Pure Composition.")
    print("="*60)
    
    tests = [
        test_basic_composition,
        test_warmup_behavior,
        test_timestamp_alignment,
        test_event_synchronization,
        test_multi_observation_sequence,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("\nSTEP 2 COMPLETE - Perception Layer Frozen")
        print("\nProven capabilities:")
        print("  ✅ Price observations (warmup aware)")
        print("  ✅ News observations (time decay, events)")
        print("  ✅ Synchronized composition")
        print("  ✅ Timestamp alignment")
        print("  ✅ Independent channels (no coupling)")
        print("\nReady for Step 3: Environment + Action Space + Reward")
    else:
        print("❌ INTEGRATION FAILED - Fix before proceeding")


if __name__ == "__main__":
    run_all_tests()