# test_news_pipeline.py
# Stress test suite for news perception pipeline
# Tests: stale events, overlapping events, zero-news periods, exchange shocks

from datetime import datetime, timedelta
from news_pipeline import NewsPipeline
from news_schema import EventScope


def print_obs(obs, test_name: str):
    """Pretty print observation for debugging"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Schema Version: {obs.SCHEMA_VERSION}")
    print(f"Time: {obs.time}")
    print(f"Symbol: {obs.symbol}")
    print(f"Event Risk: {obs.event_risk:.4f}")
    print(f"Shock Flag: {obs.shock_flag}")
    print(f"Narrative Intensity: {obs.narrative_intensity:.4f}")
    print(f"Time Decay: {obs.time_decay:.4f}")
    print(f"Event Scope: {obs.event_scope.name} ({obs.event_scope.value})")
    print(f"Confidence: {obs.confidence:.2f}")
    print(f"Is Sparse: {obs.is_sparse}")
    print(f"Event Type: {obs.event_type}")
    print(f"{'='*60}\n")


def test_zero_news():
    """
    Test 1: Zero News Period
    
    Expected:
    - event_risk = 0.0
    - shock_flag = False
    - event_type = "none"
    - High confidence in "nothing happening"
    """
    print("\n📭 TEST 1: ZERO NEWS PERIOD")
    
    pipeline = NewsPipeline(symbol="BTCUSDT")
    
    # No events added to calendar
    now = datetime(2026, 1, 20, 12, 0)
    obs = pipeline.observe(now)
    
    print_obs(obs, "Zero News Period")
    
    assert obs.event_risk == 0.0, \
        f"Expected zero risk with no events, got {obs.event_risk}"
    assert obs.shock_flag == False, \
        f"Expected no shock flag, got {obs.shock_flag}"
    assert obs.event_type == "none", \
        f"Expected event_type='none', got {obs.event_type}"
    assert obs.confidence == 1.0, \
        f"Expected high confidence, got {obs.confidence}"
    
    print("✅ Zero news period handled correctly")


def test_approaching_event():
    """
    Test 2: Event Approaching
    
    Expected:
    - Risk ramps up as event approaches
    - Risk = 0 when event is >2hrs away
    - Risk peaks at event time
    """
    print("\n⏰ TEST 2: APPROACHING EVENT")
    
    pipeline = NewsPipeline(symbol="BTCUSDT", risk_window_hours=2.0)
    
    # CPI release at 14:00
    event_time = datetime(2026, 1, 20, 14, 0)
    pipeline.add_scheduled_event(
        event_time=event_time,
        event_type="scheduled",
        description="CPI Release",
        base_risk=0.8,
        scope=EventScope.GLOBAL_RISK
    )
    
    # Test at different times
    times = [
        ("4 hours before", event_time - timedelta(hours=4)),
        ("2 hours before", event_time - timedelta(hours=2)),
        ("1 hour before", event_time - timedelta(hours=1)),
        ("Event time", event_time),
    ]
    
    risks = []
    for desc, t in times:
        obs = pipeline.observe(t)
        risks.append(obs.event_risk)
        print(f"{desc}: event_risk = {obs.event_risk:.4f}")
    
    # Verify risk progression
    assert risks[0] == 0.0, "Risk should be 0 at 4hrs (outside window)"
    assert risks[1] < risks[2], "Risk should increase as event approaches"
    assert risks[2] < risks[3], "Risk should peak at event time"
    assert risks[3] == 0.8, f"Risk should equal base_risk at event time, got {risks[3]}"
    
    print("✅ Event approach risk ramping works correctly")


def test_stale_event():
    """
    Test 3: Stale Event (Time Decay)
    
    Expected:
    - Time decay exponential
    - 24hrs later → 50% decay
    - 72hrs later → ~12.5% decay
    """
    print("\n⏳ TEST 3: STALE EVENT (TIME DECAY)")
    
    pipeline = NewsPipeline(symbol="BTCUSDT", decay_halflife_hours=24.0)
    
    # Event happened at 00:00
    event_time = datetime(2026, 1, 20, 0, 0)
    pipeline.add_scheduled_event(
        event_time=event_time,
        event_type="scheduled",
        description="FOMC Decision",
        base_risk=0.9,
        scope=EventScope.GLOBAL_RISK
    )
    
    # Observe at different times after event
    times = [
        ("Event time", event_time, 1.0),
        ("12hrs later", event_time + timedelta(hours=12), 0.707),  # ~70.7%
        ("24hrs later", event_time + timedelta(hours=24), 0.5),    # 50%
        ("48hrs later", event_time + timedelta(hours=48), 0.25),   # 25%
        ("72hrs later", event_time + timedelta(hours=72), 0.125),  # 12.5%
    ]
    
    for desc, t, expected_decay in times:
        obs = pipeline.observe(t)
        print(f"{desc}: time_decay = {obs.time_decay:.4f}, event_risk = {obs.event_risk:.4f}")
        
        # Allow 5% tolerance for float math
        assert abs(obs.time_decay - expected_decay) < 0.05, \
            f"Expected decay ~{expected_decay}, got {obs.time_decay}"
    
    print("✅ Time decay exponential function works correctly")


def test_overlapping_events():
    """
    Test 4: Overlapping Events
    
    Expected:
    - Dominant event (highest risk) is selected
    - Lesser events ignored in observation
    """
    print("\n🔀 TEST 4: OVERLAPPING EVENTS")
    
    pipeline = NewsPipeline(symbol="BTCUSDT", risk_window_hours=2.0)
    
    now = datetime(2026, 1, 20, 12, 0)
    
    # Add multiple events
    # Event 1: Low risk, happening now
    pipeline.add_scheduled_event(
        event_time=now,
        event_type="scheduled",
        description="Minor economic data",
        base_risk=0.3,
        scope=EventScope.CRYPTO_WIDE
    )
    
    # Event 2: High risk, happening in 30 minutes
    pipeline.add_scheduled_event(
        event_time=now + timedelta(minutes=30),
        event_type="scheduled",
        description="CPI Release",
        base_risk=0.9,
        scope=EventScope.GLOBAL_RISK
    )
    
    # Event 3: Medium risk, happened 1 hour ago
    pipeline.add_scheduled_event(
        event_time=now - timedelta(hours=1),
        event_type="regulatory",
        description="SEC Statement",
        base_risk=0.5,
        scope=EventScope.CRYPTO_WIDE
    )
    
    obs = pipeline.observe(now)
    print_obs(obs, "Overlapping Events")
    
    # Event 2 (CPI) should dominate - highest current risk
    # At 30min before event, risk should be high
    assert obs.event_risk > 0.5, \
        f"Expected high risk from CPI event, got {obs.event_risk}"
    assert obs.event_scope == EventScope.GLOBAL_RISK, \
        f"Expected GLOBAL_RISK scope from CPI, got {obs.event_scope.name}"
    
    print("✅ Dominant event selection works correctly")


def test_exchange_shock():
    """
    Test 5: Exchange Outage (Shock Flag)
    
    Expected:
    - shock_flag = True when exchange down
    - Agent should preserve capital, disable exploration
    """
    print("\n💥 TEST 5: EXCHANGE SHOCK")
    
    pipeline = NewsPipeline(symbol="BTCUSDT")
    
    now = datetime(2026, 1, 20, 12, 0)
    
    # Normal operation
    obs_normal = pipeline.observe(now)
    print_obs(obs_normal, "Exchange Normal")
    assert obs_normal.shock_flag == False, "No shock in normal operation"
    
    # Exchange goes down
    pipeline.set_exchange_status(operational=False)
    obs_shock = pipeline.observe(now)
    print_obs(obs_shock, "Exchange Outage")
    
    assert obs_shock.shock_flag == True, \
        f"Expected shock_flag=True during outage, got {obs_shock.shock_flag}"
    
    print("✅ Exchange shock detection works correctly")


def test_event_scope_classification():
    """
    Test 6: Event Scope Classification
    
    Expected:
    - BTC-specific events → BTC_ONLY scope
    - Crypto-wide events → CRYPTO_WIDE scope
    - Macro events → GLOBAL_RISK scope
    """
    print("\n🌍 TEST 6: EVENT SCOPE CLASSIFICATION")
    
    pipeline = NewsPipeline(symbol="BTCUSDT", risk_window_hours=2.0)
    
    now = datetime(2026, 1, 20, 12, 0)
    
    # Test different scopes
    test_cases = [
        ("BTC ETF Decision", EventScope.BTC_ONLY),
        ("SEC Crypto Crackdown", EventScope.CRYPTO_WIDE),
        ("Fed Emergency Meeting", EventScope.GLOBAL_RISK),
    ]
    
    for desc, expected_scope in test_cases:
        # Clear previous events
        pipeline.scheduled_events = []
        
        # Add event with specific scope
        pipeline.add_scheduled_event(
            event_time=now,
            event_type="scheduled",
            description=desc,
            base_risk=0.7,
            scope=expected_scope
        )
        
        obs = pipeline.observe(now)
        print(f"{desc}: scope = {obs.event_scope.name}")
        
        assert obs.event_scope == expected_scope, \
            f"Expected {expected_scope.name}, got {obs.event_scope.name}"
    
    print("✅ Event scope classification works correctly")


def test_confidence_levels():
    """
    Test 7: Confidence Computation
    
    Expected:
    - Scheduled events → confidence = 1.0
    - No events → confidence = 1.0 (certain nothing is happening)
    """
    print("\n🎯 TEST 7: CONFIDENCE LEVELS")
    
    pipeline = NewsPipeline(symbol="BTCUSDT")
    
    now = datetime(2026, 1, 20, 12, 0)
    
    # Test 1: No events
    obs_none = pipeline.observe(now)
    assert obs_none.confidence == 1.0, \
        f"Expected confidence=1.0 for no events, got {obs_none.confidence}"
    print(f"No events: confidence = {obs_none.confidence:.2f} ✓")
    
    # Test 2: Scheduled event
    pipeline.add_scheduled_event(
        event_time=now,
        event_type="scheduled",
        description="CPI Release",
        base_risk=0.8,
        scope=EventScope.GLOBAL_RISK
    )
    
    obs_scheduled = pipeline.observe(now)
    assert obs_scheduled.confidence == 1.0, \
        f"Expected confidence=1.0 for scheduled event, got {obs_scheduled.confidence}"
    print(f"Scheduled event: confidence = {obs_scheduled.confidence:.2f} ✓")
    
    print("✅ Confidence computation works correctly")


def run_all_tests():
    """Run full stress test suite"""
    print("\n" + "="*60)
    print("NEWS PIPELINE STRESS TEST SUITE")
    print("Schema Version: 1.0.0")
    print("="*60)
    
    tests = [
        test_zero_news,
        test_approaching_event,
        test_stale_event,
        test_overlapping_events,
        test_exchange_shock,
        test_event_scope_classification,
        test_confidence_levels,
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
        print("✅ ALL TESTS PASSED - Step 2B is COMPLETE")
        print("Ready to integrate with Step 2A (Price Pipeline)")
    else:
        print("❌ TESTS FAILED - Fix issues before proceeding")


if __name__ == "__main__":
    run_all_tests()