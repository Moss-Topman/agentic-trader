"""
Run scripted agent tests with the actual BacktestEnv.
This is the final Step 4 validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from backtest_env import BacktestEnv
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from scripted_agents import (
    RandomAgent,
    AlwaysEnterAgent,
    MomentumFollowAgent,
    ConservativeAgent,
    AgentTestRunner
)
from running_stats import ObservationNormalizer
from trade_logger import TradeLogger
from market_observation import MarketObservation


def load_sample_data():
    """Create synthetic OHLCV data for testing."""
    print("  Generating synthetic market data...")
    
    # Generate 1000 hourly candles
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
    
    # Create realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, size=1000)
    
    base_price = 50000.0
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLC from close
    data = pd.DataFrame({
        'time': dates,
        'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, 1000)),
        'high': close_prices * (1 + np.random.uniform(0.001, 0.003, 1000)),
        'low': close_prices * (1 + np.random.uniform(-0.003, -0.001, 1000)),
        'close': close_prices,
        'volume': np.random.uniform(1e6, 5e6, 1000),
        
        # Add spread column (required by price pipeline)
        'spread': close_prices * 0.0002,  # 2bp spread
    })
    
    print(f"  Generated {len(data)} candles")
    print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data


def obs_to_dict(obs: MarketObservation) -> dict:
    """
    Convert MarketObservation to dictionary for normalizer.
    
    Matches actual PriceObservation and NewsObservation schema v1.0.0.
    All values converted to native Python types for database compatibility.
    """
    return {
        # Price fields (from PriceObservation)
        "price.close": float(obs.price.close),
        "price.spread": float(obs.price.spread),
        "price.volatility": float(obs.price.volatility),
        "price.pressure": float(obs.price.pressure),
        "price.trend_strength": float(obs.price.trend_strength),
        "price.pressure_confidence": float(obs.price.pressure_confidence),
        "price.trend_regime": int(obs.price.trend_regime),
        "price.volatility_regime": int(obs.price.volatility_regime),
        "price.is_warmup": float(obs.price.is_warmup),  # Convert bool to float
        
        # News fields (from NewsObservation)
        "news.event_risk": float(obs.news.event_risk),
        "news.shock_flag": float(obs.news.shock_flag),  # Convert bool to float
        "news.narrative_intensity": float(obs.news.narrative_intensity),
        "news.time_decay": float(obs.news.time_decay),
        "news.confidence": float(obs.news.confidence),
        "news.is_sparse": float(obs.news.is_sparse),  # Convert bool to float
        "news.event_scope": int(obs.news.event_scope),
    }


class CustomAgentTestRunner(AgentTestRunner):
    """
    Custom test runner that works with your BacktestEnv.
    
    Overrides _obs_to_dict to use MarketObservation structure.
    """
    
    def _obs_to_dict(self, obs) -> dict:
        """Convert MarketObservation to dictionary."""
        return obs_to_dict(obs)


def main():
    print("="*60)
    print("STEP 4 SCRIPTED AGENT TESTS")
    print("="*60)
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1/6] Loading data...")
    data = load_sample_data()
    
    # ========================================================================
    # 2. CREATE PIPELINES
    # ========================================================================
    print("\n[2/6] Creating perception pipelines...")
    
    price_pipeline = PricePipeline(
        symbol="BTCUSDT",
        ma_fast=20,
        ma_slow=50,
        vol_window=20,
        trend_threshold=0.003
    )
    print("  Price pipeline created")
    
    news_pipeline = NewsPipeline(
        symbol="BTCUSDT",
        risk_window_hours=2.0,
        decay_halflife_hours=24.0
    )
    print("  News pipeline created")
    
    # ========================================================================
    # 3. CREATE ENVIRONMENT
    # ========================================================================
    print("\n[3/6] Creating environment...")
    env = BacktestEnv(
        data=data,
        price_pipeline=price_pipeline,
        news_pipeline=news_pipeline,
        episode_length=100,  # Shorter for testing
        holding_penalty=0.0001,
        initial_capital=10000.0
    )
    print("  Environment created")
    
    # ========================================================================
    # 4. INITIALIZE STEP 4 COMPONENTS
    # ========================================================================
    print("\n[4/6] Initializing Step 4 components...")
    
    normalizer = ObservationNormalizer(mode="train")
    print("  Normalizer created")
    
    logger = TradeLogger(
        db_host="timescaledb",
        db_port=5432,
        db_name="agentic_db",
        db_user="agentic",
        db_password="agenticpass",
        batch_size=50
    )
    print(f"  Logger created (DB available: {logger.db_available})")
    
    # ========================================================================
    # 5. CREATE TEST RUNNER
    # ========================================================================
    runner = CustomAgentTestRunner(
        env=env,
        normalizer=normalizer,
        logger=logger,
        validate_leakage=False  # Enable if you want leakage checks
    )
    print("  Test runner created")
    
    # ========================================================================
    # 6. CREATE AND RUN AGENTS
    # ========================================================================
    print("\n[5/6] Creating agents...")
    agents = [
        RandomAgent(seed=42),
        AlwaysEnterAgent(),
        MomentumFollowAgent(),
        ConservativeAgent()
    ]
    print(f"  Created {len(agents)} agents")
    
    print("\n[6/6] Running agent tests...")
    print("-"*60)
    
    num_episodes = 5  # 5 episodes per agent
    all_results = {}
    
    for agent in agents:
        print(f"\n{'='*60}")
        print(f"Testing: {agent.name}")
        print(f"{'='*60}")
        
        try:
            results = runner.run_multiple_episodes(agent, num_episodes, verbose=True)
            all_results[agent.name] = results
            
            # Print summary
            stats = agent.get_stats()
            print(f"\n{agent.name} Summary:")
            print(f"  Episodes Completed: {stats['episodes']}")
            print(f"  Mean Reward: {stats['mean_reward']:.4f}")
            print(f"  Std Reward: {stats.get('std_reward', 0.0):.4f}")
            print(f"  Min Reward: {stats.get('min_reward', 0.0):.4f}")
            print(f"  Max Reward: {stats.get('max_reward', 0.0):.4f}")
            print(f"  Mean Trades: {stats['mean_trades']:.1f}")
            
        except Exception as e:
            print(f"\n❌ ERROR testing {agent.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 7. CLEANUP AND SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    logger.flush_all()
    logger.close()
    print("✓ Logger closed")
    
    # Print overall summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_episodes = sum(len(results) for results in all_results.values())
    print(f"\nTotal Episodes Run: {total_episodes}")
    print(f"Agents Tested: {len(all_results)}")
    
    print("\nAgent Performance Comparison:")
    print(f"{'Agent':<20} {'Mean Reward':<15} {'Mean Trades':<15}")
    print("-"*50)
    
    for agent_name, results in all_results.items():
        mean_reward = np.mean([r['total_reward'] for r in results])
        mean_trades = np.mean([r['num_trades'] for r in results])
        print(f"{agent_name:<20} {mean_reward:<15.4f} {mean_trades:<15.1f}")
    
    print("\n" + "="*60)
    print("STEP 4 VALIDATION COMPLETE ✓")
    print("="*60)
    
    print("\nTo view logged data in database:")
    print("  docker exec -it timescaledb psql -U agentic -d agentic_db")
    print("  \\dt  -- List tables")
    print("  SELECT * FROM episode_log ORDER BY episode_id DESC LIMIT 10;")
    print("  SELECT * FROM action_distribution LIMIT 5;")
    
    print("\nNext Step: Proceed to Step 5 (Neural Network Training)")


if __name__ == "__main__":
    main()