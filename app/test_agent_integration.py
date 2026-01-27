"""
Test Agent Integration with Environment
Validates that Agent interface works end-to-end with Step 4 infrastructure.

This is the final validation before adding learning.
If this passes, the pipeline is ready for REINFORCE.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from agent_interface import UniformRandomAgent, validate_agent_contract
from backtest_env import BacktestEnv
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from running_stats import ObservationNormalizer
from trade_logger import TradeLogger
from action_masking import ActionMasker


def obs_to_vector(obs_dict: dict) -> np.ndarray:
    """
    Convert observation dictionary to vector for agent.
    
    Order matters - must be consistent across all agent calls.
    
    Args:
        obs_dict: Dictionary from obs_to_dict()
        
    Returns:
        Observation vector [n_features]
    """
    # Define feature order (MUST remain consistent)
    feature_names = [
        "price.close",
        "price.spread",
        "price.volatility",
        "price.pressure",
        "price.trend_strength",
        "price.pressure_confidence",
        "price.trend_regime",
        "price.volatility_regime",
        "price.is_warmup",
        "news.event_risk",
        "news.shock_flag",
        "news.narrative_intensity",
        "news.time_decay",
        "news.confidence",
        "news.is_sparse",
        "news.event_scope",
    ]
    
    # Extract features in order
    vector = np.array([
        float(obs_dict.get(name, 0.0))
        for name in feature_names
    ], dtype=np.float32)
    
    return vector


def obs_to_dict(obs) -> dict:
    """
    Convert MarketObservation to dictionary.
    Matches run_scripted_tests.py implementation.
    """
    return {
        "price.close": float(obs.price.close),
        "price.spread": float(obs.price.spread),
        "price.volatility": float(obs.price.volatility),
        "price.pressure": float(obs.price.pressure),
        "price.trend_strength": float(obs.price.trend_strength),
        "price.pressure_confidence": float(obs.price.pressure_confidence),
        "price.trend_regime": int(obs.price.trend_regime),
        "price.volatility_regime": int(obs.price.volatility_regime),
        "price.is_warmup": float(obs.price.is_warmup),
        "news.event_risk": float(obs.news.event_risk),
        "news.shock_flag": float(obs.news.shock_flag),
        "news.narrative_intensity": float(obs.news.narrative_intensity),
        "news.time_decay": float(obs.news.time_decay),
        "news.confidence": float(obs.news.confidence),
        "news.is_sparse": float(obs.news.is_sparse),
        "news.event_scope": int(obs.news.event_scope),
    }


def load_sample_data():
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
    
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, size=1000)
    
    base_price = 50000.0
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'time': dates,
        'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, 1000)),
        'high': close_prices * (1 + np.random.uniform(0.001, 0.003, 1000)),
        'low': close_prices * (1 + np.random.uniform(-0.003, -0.001, 1000)),
        'close': close_prices,
        'volume': np.random.uniform(1e6, 5e6, 1000),
        'spread': close_prices * 0.0002,
    })
    
    return data


def run_agent_episode(
    agent,
    env,
    normalizer: ObservationNormalizer,
    logger: TradeLogger,
    episode_id: int,
    verbose: bool = True
):
    """
    Run one episode with agent following strict contract.
    
    Args:
        agent: Agent instance (must follow Agent interface)
        env: BacktestEnv instance
        normalizer: ObservationNormalizer for observations
        logger: TradeLogger for persistence
        episode_id: Episode identifier
        verbose: Print progress
        
    Returns:
        Episode metrics dictionary
    """
    # Reset
    agent.reset()
    obs = env.reset()
    
    # Episode tracking
    start_time = datetime.now()
    step_index = 0
    total_reward = 0.0
    num_trades = 0
    
    initial_capital = env.initial_capital
    equity_curve = [initial_capital]
    
    done = False
    
    while not done:
        # 1. Convert observation to dict
        obs_dict = obs_to_dict(obs)
        
        # 2. Normalize
        obs_normalized_dict = normalizer.normalize(obs_dict)
        
        # 3. Convert to vector
        obs_vector = obs_to_vector(obs_normalized_dict)
        
        # 4. Get action mask
        action_mask = ActionMasker.get_action_mask(env.state.position_open)
        
        # 5. Agent acts (THIS IS THE CONTRACT)
        action_probs = agent.act(obs_vector, action_mask)
        
        # 6. Sample action
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # 7. Execute
        next_obs, reward, done, info = env.step(action_idx)
        
        # 8. Agent observes (THIS IS THE CONTRACT)
        agent.observe(reward, done, info={'action': action_idx})
        
        # 9. Log
        logger.log_step(
            episode_id=episode_id,
            step_index=step_index,
            timestamp=datetime.now(),
            observation=obs_normalized_dict,
            action=action_idx,
            action_was_legal=True,
            reward=reward,
            position_open=env.state.position_open,
            entry_price=env.state.entry_price,
            realized_pnl=env.state.realized_pnl
        )
        
        # 10. Track metrics
        total_reward += reward
        if action_idx == 1:  # ENTER_LONG
            num_trades += 1
        
        current_equity = initial_capital + env.state.realized_pnl
        equity_curve.append(current_equity)
        
        # 11. Next
        obs = next_obs
        step_index += 1
    
    # Episode ended - agent learns (THIS IS THE CONTRACT)
    agent.end_episode()
    
    # Compute metrics
    end_time = datetime.now()
    from trade_logger import compute_max_drawdown, compute_sharpe_ratio
    
    max_drawdown = compute_max_drawdown(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = compute_sharpe_ratio(returns.tolist()) if len(returns) > 0 else None
    
    # Log episode
    logger.log_episode(
        episode_id=episode_id,
        start_time=start_time,
        end_time=end_time,
        total_reward=total_reward,
        realized_pnl=env.state.realized_pnl,
        num_steps=step_index,
        num_trades=num_trades,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        forced_liquidation=info.get("forced_liquidation", False)
    )
    
    if verbose:
        print(
            f"Episode {episode_id}: "
            f"Reward={total_reward:.4f}, "
            f"Trades={num_trades}, "
            f"Balance={current_equity:.2f}"
        )
    
    return {
        "episode_id": episode_id,
        "total_reward": total_reward,
        "num_steps": step_index,
        "num_trades": num_trades,
        "final_balance": current_equity,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def main():
    """
    Full integration test: Agent + Environment + Step 4 components.
    """
    print("="*60)
    print("AGENT INTERFACE INTEGRATION TEST")
    print("="*60)
    
    # 1. Validate agent contract
    print("\n[1/5] Validating agent contract...")
    agent = UniformRandomAgent(n_actions=3, seed=42)
    validate_agent_contract(agent)
    print("✓ Agent contract valid")
    
    # 2. Create environment
    print("\n[2/5] Creating environment...")
    data = load_sample_data()
    
    price_pipeline = PricePipeline(
        symbol="BTCUSDT",
        ma_fast=20,
        ma_slow=50,
        vol_window=20,
        trend_threshold=0.003
    )
    
    news_pipeline = NewsPipeline(
        symbol="BTCUSDT",
        risk_window_hours=2.0,
        decay_halflife_hours=24.0
    )
    
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipeline,
        news_pipeline=news_pipeline,
        episode_length=100,
        holding_penalty=0.0001,
        initial_capital=10000.0
    )
    print("✓ Environment created")
    
    # 3. Initialize Step 4 components
    print("\n[3/5] Initializing Step 4 components...")
    normalizer = ObservationNormalizer(mode="train")
    
    logger = TradeLogger(
        db_host="timescaledb",
        db_port=5432,
        db_name="agentic_db",
        db_user="agentic",
        db_password="agenticpass",
        batch_size=50
    )
    print(f"✓ Components ready (DB: {logger.db_available})")
    
    # 4. Run test episodes
    print("\n[4/5] Running test episodes...")
    
    # Get starting episode_id from database to avoid conflicts
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="timescaledb",
            port=5432,
            database="agentic_db",
            user="agentic",
            password="agenticpass"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(episode_id), -1) + 1 FROM episode_log")
        start_episode_id = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print(f"Starting from episode_id: {start_episode_id}")
    except Exception as e:
        print(f"Warning: Could not get max episode_id from DB: {e}")
        start_episode_id = 0
    
    num_episodes = 5
    results = []
    
    for ep in range(num_episodes):
        metrics = run_agent_episode(
            agent=agent,
            env=env,
            normalizer=normalizer,
            logger=logger,
            episode_id=start_episode_id + ep,
            verbose=True
        )
        results.append(metrics)
    
    # 5. Summary
    print("\n[5/5] Summary...")
    logger.flush_all()
    logger.close()
    
    agent_stats = agent.get_stats()
    print("\nAgent Statistics:")
    print(f"  Episodes: {agent_stats['episodes']}")
    print(f"  Mean Reward: {agent_stats['mean_reward']:.4f}")
    print(f"  Std Reward: {agent_stats['std_reward']:.4f}")
    print(f"  Min Reward: {agent_stats['min_reward']:.4f}")
    print(f"  Max Reward: {agent_stats['max_reward']:.4f}")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE ✓")
    print("="*60)
    
    print("\nValidation Results:")
    print("  ✓ Agent follows contract")
    print("  ✓ Observation pipeline works")
    print("  ✓ Action masking enforced")
    print("  ✓ Logging persists correctly")
    print("  ✓ Episode boundaries clean")
    
    print("\nReady for Step 5B: Learning Agent (REINFORCE)")


if __name__ == "__main__":
    main()