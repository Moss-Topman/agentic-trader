# test_backtest_env.py
# Sanity tests for backtest environment
# Tests BEFORE any learning - validates environment isn't broken

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtest_env import BacktestEnv
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from action_space import Action


def create_test_data(n_candles: int = 600) -> pd.DataFrame:
    """Create synthetic test data for environment testing"""
    times = pd.date_range("2026-01-10", periods=n_candles, freq="h")
    
    # Create slightly trending data
    base = 50000.0
    trend = np.linspace(0, 1000, n_candles)  # Slow uptrend
    noise = np.random.normal(0, 100, n_candles)
    prices = base + trend + noise
    
    df = pd.DataFrame({
        "time": times,
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": [1000.0] * n_candles,
        "spread": [0.0005] * n_candles,  # 0.05% spread
    })
    
    return df


def test_environment_initialization():
    """
    Test 1: Environment Initialization
    
    Verifies environment can be created and reset without errors.
    """
    print("\n" + "="*60)
    print("TEST 1: ENVIRONMENT INITIALIZATION")
    print("="*60)
    
    # Create data and pipelines
    data = create_test_data(600)
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    # Create environment
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=100,
    )
    
    # Reset
    obs = env.reset(start_index=50)
    
    # Verify observation structure
    assert obs.price is not None, "Price observation missing"
    assert obs.news is not None, "News observation missing"
    
    # Verify initial state
    state = env.get_state_summary()
    assert state["position_open"] == False, "Should start flat"
    assert state["realized_pnl"] == 0.0, "Should start with zero PnL"
    
    print("✅ Environment initializes correctly")
    print(f"   Start index: {state['current_index']}")
    print(f"   Position: {state['position_open']}")


def test_random_agent():
    """
    Test 2: Random Agent Doesn't Explode
    
    Critical sanity check - environment must handle random actions
    without crashes, NaNs, or infinite loops.
    """
    print("\n" + "="*60)
    print("TEST 2: RANDOM AGENT")
    print("="*60)
    
    data = create_test_data(600)
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=100,
    )
    
    # Run 3 episodes with random actions
    episode_returns = []
    
    for ep in range(3):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 200:  # Safety limit
            # Random action
            action = np.random.choice([Action.HOLD, Action.ENTER_LONG, Action.EXIT])
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Verify no NaNs
            assert not np.isnan(reward), f"NaN reward at step {steps}"
            assert not np.isinf(reward), f"Inf reward at step {steps}"
        
        episode_returns.append(total_reward)
        print(f"Episode {ep+1}: {steps} steps, return={total_reward:.6f}")
    
    # Verify all episodes completed successfully
    assert len(episode_returns) == 3, f"Expected 3 episodes, got {len(episode_returns)}"
    
    print(f"✅ Random agent completed {len(episode_returns)} episodes")
    print(f"   Mean return: {np.mean(episode_returns):.6f}")
    print(f"   Std return: {np.std(episode_returns):.6f}")


def test_always_hold_agent():
    """
    Test 3: Always-Hold Agent
    
    Agent that never trades should get:
    - Zero realized PnL
    - Zero total reward
    
    This validates passive strategy baseline.
    """
    print("\n" + "="*60)
    print("TEST 3: ALWAYS-HOLD AGENT")
    print("="*60)
    
    data = create_test_data(600)
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=100,
    )
    
    obs = env.reset(start_index=50)
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done:
        # Always hold (do nothing)
        obs, reward, done, info = env.step(Action.HOLD)
        total_reward += reward
        steps += 1
    
    state = env.get_state_summary()
    
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Realized PnL: {state['realized_pnl']:.6f}")
    
    # Always-hold should yield zero
    assert abs(total_reward) < 1e-6, f"Expected ~0 reward, got {total_reward}"
    assert abs(state["realized_pnl"]) < 1e-6, f"Expected ~0 PnL, got {state['realized_pnl']}"
    
    print("✅ Always-hold agent behaves correctly (zero reward)")


def test_single_trade_cycle():
    """
    Test 4: Single Trade Cycle
    
    Executes one complete trade: ENTER → HOLD → EXIT
    Verifies transaction costs are applied correctly.
    """
    print("\n" + "="*60)
    print("TEST 4: SINGLE TRADE CYCLE")
    print("="*60)
    
    # Create flat data (no price change) to isolate costs
    times = pd.date_range("2026-01-10", periods=200, freq="h")
    df = pd.DataFrame({
        "time": times,
        "open": [50000.0] * 200,
        "high": [50000.0] * 200,
        "low": [50000.0] * 200,
        "close": [50000.0] * 200,
        "volume": [1000.0] * 200,
        "spread": [0.001] * 200,  # 0.1% spread
    })
    
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    env = BacktestEnv(
        data=df,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=100,
        holding_penalty=0.0001,
    )
    
    obs = env.reset(start_index=50)
    
    # Enter long
    obs, r1, done, info = env.step(Action.ENTER_LONG)
    print(f"ENTER_LONG: reward={r1:.6f}")
    assert r1 == 0.0, f"Entry should have zero immediate reward, got {r1}"
    
    # Hold for 10 candles
    hold_rewards = []
    for _ in range(10):
        obs, r, done, info = env.step(Action.HOLD)
        hold_rewards.append(r)
    
    print(f"HOLD (10 candles): avg reward={np.mean(hold_rewards):.6f}")
    assert all(abs(r + 0.0001) < 1e-8 for r in hold_rewards), \
        f"Holding should incur -0.0001 penalty, got {hold_rewards[0]}"
    
    # Exit (price unchanged, only costs)
    obs, r_exit, done, info = env.step(Action.EXIT)
    print(f"EXIT: reward={r_exit:.6f}")
    
    # Expected: log(50000/50000) - (0.001 + 0.001) = 0 - 0.002 = -0.002
    expected_exit_reward = -0.002
    assert abs(r_exit - expected_exit_reward) < 1e-6, \
        f"Expected exit reward ~{expected_exit_reward}, got {r_exit}"
    
    # Total episode reward should be negative (costs + holding penalties)
    total_expected = expected_exit_reward + sum(hold_rewards)
    print(f"Total reward: {r1 + sum(hold_rewards) + r_exit:.6f}")
    print(f"Expected: {total_expected:.6f}")
    
    print("✅ Transaction costs applied correctly")


def test_illegal_action_handling():
    """
    Test 5: Illegal Action Handling
    
    Verifies illegal actions are no-ops without crashes.
    """
    print("\n" + "="*60)
    print("TEST 5: ILLEGAL ACTION HANDLING")
    print("="*60)
    
    data = create_test_data(600)
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=100,
    )
    
    obs = env.reset(start_index=50)
    
    # Test 1: Try to exit when flat (illegal)
    obs, reward, done, info = env.step(Action.EXIT)
    assert reward == 0.0, "Illegal action should yield zero reward"
    assert info["illegal_action"] == True, "Should flag illegal action"
    print("✅ EXIT when flat → no-op")
    
    # Test 2: Enter long (legal)
    obs, reward, done, info = env.step(Action.ENTER_LONG)
    assert info["illegal_action"] == False, "Legal action flagged as illegal"
    print("✅ ENTER_LONG when flat → legal")
    
    # Test 3: Try to enter again (illegal)
    obs, reward, done, info = env.step(Action.ENTER_LONG)
    assert reward == 0.0, "Illegal double entry should yield zero"
    assert info["illegal_action"] == True, "Should flag illegal action"
    print("✅ ENTER_LONG when already long → no-op")
    
    # Test 4: Exit (legal)
    obs, reward, done, info = env.step(Action.EXIT)
    assert info["illegal_action"] == False, "Legal exit flagged as illegal"
    print("✅ EXIT when long → legal")
    
    print("✅ All illegal actions handled as no-ops")


def test_episode_termination():
    """
    Test 6: Episode Termination
    
    Verifies episodes end correctly and force-close logic works.
    """
    print("\n" + "="*60)
    print("TEST 6: EPISODE TERMINATION")
    print("="*60)
    
    data = create_test_data(600)
    price_pipe = PricePipeline(symbol="BTCUSDT")
    news_pipe = NewsPipeline(symbol="BTCUSDT")
    
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipe,
        news_pipeline=news_pipe,
        episode_length=50,  # Short episode
    )
    
    obs = env.reset(start_index=100)
    
    # Enter long and hold until episode end
    obs, r, done, info = env.step(Action.ENTER_LONG)
    
    steps = 0
    while not done and steps < 100:  # Safety limit
        obs, r, done, info = env.step(Action.HOLD)
        steps += 1
    
    # Verify episode ended
    assert done == True, "Episode should have ended"
    
    # Episode length is 50 candles
    # We took: 1 ENTER + 49 HOLD = 50 total actions
    # Test only counted HOLD actions (49)
    assert steps == 49, f"Expected 49 HOLD actions after ENTER, got {steps}"
    
    # Verify position was force-closed
    state = env.get_state_summary()
    assert state["position_open"] == False, "Position should be force-closed at episode end"
    
    print(f"✅ Episode terminated correctly")
    print(f"   Total actions: 1 ENTER + {steps} HOLD = {1+steps}")
    print(f"   Position force-closed: {not state['position_open']}")
    print(f"   Final PnL: {state['realized_pnl']:.6f}")


def run_all_tests():
    """Run all environment sanity tests"""
    print("\n" + "="*60)
    print("BACKTEST ENVIRONMENT SANITY TEST SUITE")
    print("Testing BEFORE any learning")
    print("="*60)
    
    tests = [
        test_environment_initialization,
        test_random_agent,
        test_always_hold_agent,
        test_single_trade_cycle,
        test_illegal_action_handling,
        test_episode_termination,
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
        print("✅ ALL SANITY TESTS PASSED")
        print("\nEnvironment is validated:")
        print("  ✅ Random agent doesn't explode")
        print("  ✅ Always-hold yields zero")
        print("  ✅ Transaction costs work")
        print("  ✅ Illegal actions handled")
        print("  ✅ Episodes terminate correctly")
        print("\nReady for agent training")
    else:
        print("❌ ENVIRONMENT BROKEN - Fix before training")


if __name__ == "__main__":
    run_all_tests()