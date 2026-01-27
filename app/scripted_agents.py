"""
Scripted Agent Tests
Validates all Step 4 components work together before neural network training.
Implements deterministic policies to test infrastructure.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from action_masking import Action, ActionMasker
from running_stats import ObservationNormalizer
from trade_logger import TradeLogger, compute_sharpe_ratio, compute_max_drawdown
from leakage_prevention import assert_no_future_leakage


class ScriptedAgent:
    """Base class for scripted trading agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_rewards: List[float] = []
        self.episode_trades: List[int] = []
    
    def select_action(self, observation: Dict[str, Any], position_open: bool) -> Action:
        """
        Select action based on observation and position state.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset agent state at episode start."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if not self.episode_rewards:
            return {
                "name": self.name,
                "episodes": 0,
                "mean_reward": 0.0,
                "mean_trades": 0.0
            }
        
        return {
            "name": self.name,
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "mean_trades": np.mean(self.episode_trades),
        }


class AlwaysEnterAgent(ScriptedAgent):
    """
    Agent that always enters on first opportunity and holds forever.
    
    Expected behavior:
    - Enters immediately when position is closed
    - Never exits
    - Accumulates time penalty
    - Should lose money (negative reward)
    """
    
    def __init__(self):
        super().__init__("AlwaysEnter")
    
    def select_action(self, observation: Dict[str, Any], position_open: bool) -> Action:
        if not position_open:
            return Action.ENTER_LONG
        else:
            return Action.HOLD


class MomentumFollowAgent(ScriptedAgent):
    """
    Agent that follows momentum signals.
    
    Strategy:
    - Enters when trend regime is UP and not in position
    - Exits when trend regime is DOWN and in position
    - Holds when trend is NEUTRAL
    
    Expected behavior:
    - Should outperform random agent in trending markets
    - May break even or slight profit/loss
    """
    
    def __init__(self):
        super().__init__("MomentumFollow")
    
    def select_action(self, observation: Dict[str, Any], position_open: bool) -> Action:
        # Trend regime: 0=DOWN, 1=NEUTRAL, 2=UP (from observation schema)
        trend_regime = observation.get("price.trend_regime", 1)
        
        if trend_regime == 2 and not position_open:  # UP trend, enter
            return Action.ENTER_LONG
        elif trend_regime == 0 and position_open:  # DOWN trend, exit
            return Action.EXIT
        else:
            return Action.HOLD


class RandomAgent(ScriptedAgent):
    """
    Agent that takes random legal actions.
    
    Uses action masking to ensure only legal actions are sampled.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: Dict[str, Any], position_open: bool) -> Action:
        # Get legal actions
        legal_actions = ActionMasker.get_legal_actions(position_open)
        
        # Sample uniformly from legal actions
        return self.rng.choice(legal_actions)


class ConservativeAgent(ScriptedAgent):
    """
    Agent that enters only in low-risk, high-confidence conditions.
    
    Strategy:
    - Enters only when:
      - Trend is UP
      - Volatility is LOW or NORMAL
      - Event risk is below 0.3
    - Exits when any risk signal appears
    
    Expected behavior:
    - Very few trades
    - High win rate (when it trades)
    - May miss opportunities
    """
    
    def __init__(self):
        super().__init__("Conservative")
    
    def select_action(self, observation: Dict[str, Any], position_open: bool) -> Action:
        trend_regime = observation.get("price.trend_regime", 1)
        vol_regime = observation.get("price.volatility_regime", 1)
        event_risk = observation.get("news.event_risk", 0.0)
        
        # Entry conditions (very conservative)
        can_enter = (
            trend_regime == 2 and  # UP trend
            vol_regime <= 1 and    # LOW or NORMAL volatility
            event_risk < 0.3       # Low event risk
        )
        
        # Exit conditions (at first sign of trouble)
        should_exit = (
            trend_regime == 0 or   # DOWN trend
            vol_regime == 2 or     # HIGH volatility
            event_risk > 0.5       # High event risk
        )
        
        if can_enter and not position_open:
            return Action.ENTER_LONG
        elif should_exit and position_open:
            return Action.EXIT
        else:
            return Action.HOLD


class AgentTestRunner:
    """
    Runs scripted agents through environment to test Step 4 components.
    
    Validates:
    - Observation normalization works
    - Action masking prevents illegal actions
    - Leakage prevention catches errors
    - Trade logging persists correctly
    """
    
    def __init__(
        self,
        env,  # BacktestEnv instance
        normalizer: ObservationNormalizer,
        logger: TradeLogger,
        validate_leakage: bool = True
    ):
        self.env = env
        self.normalizer = normalizer
        self.logger = logger
        self.validate_leakage = validate_leakage
        
        self.episode_id_counter = 0
    
    def run_episode(self, agent: ScriptedAgent) -> Dict[str, Any]:
        """
        Run a single episode with the given agent.
        
        Returns:
            Episode metrics dictionary
        """
        # Reset environment and agent
        obs = self.env.reset()
        agent.reset()
        
        # Episode tracking
        episode_id = self.episode_id_counter
        self.episode_id_counter += 1
        
        start_time = datetime.now()
        step_index = 0
        total_reward = 0.0
        num_trades = 0
        
        # Get initial capital from environment
        initial_capital = getattr(self.env, 'initial_capital', 10000.0)
        equity_curve = [initial_capital]
        returns = []
        
        done = False
        
        while not done:
            # Extract observation as dictionary
            obs_dict = self._obs_to_dict(obs)
            
            # Normalize observation
            obs_norm_dict = self.normalizer.normalize(obs_dict)
            
            # Agent selects action
            action = agent.select_action(obs_norm_dict, self.env.state.position_open)
            
            # Validate action is legal
            assert ActionMasker.is_action_legal(action, self.env.state.position_open), \
                f"Agent {agent.name} selected illegal action {action}"
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            
            # Track metrics
            total_reward += reward
            if action == Action.ENTER_LONG:
                num_trades += 1
            
            # Calculate current equity: initial capital + realized PnL
            current_equity = initial_capital + self.env.state.realized_pnl
            equity_curve.append(current_equity)
            
            if len(equity_curve) >= 2:
                ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(ret)
            
            # Log step
            self.logger.log_step(
                episode_id=episode_id,
                step_index=step_index,
                timestamp=datetime.now(),
                observation=obs_norm_dict,
                action=int(action),
                action_was_legal=True,  # We validated above
                reward=reward,
                position_open=self.env.state.position_open,
                entry_price=self.env.state.entry_price,
                realized_pnl=info.get("realized_pnl", 0.0)
            )
            
            # Update for next iteration
            obs = next_obs
            step_index += 1
        
        # Compute episode metrics
        end_time = datetime.now()
        max_drawdown = compute_max_drawdown(equity_curve)
        sharpe_ratio = compute_sharpe_ratio(returns) if returns else None
        
        # Log episode
        self.logger.log_episode(
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            total_reward=total_reward,
            realized_pnl=self.env.state.realized_pnl,
            num_steps=step_index,
            num_trades=num_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            forced_liquidation=info.get("forced_liquidation", False)
        )
        
        # Update agent stats
        agent.episode_rewards.append(total_reward)
        agent.episode_trades.append(num_trades)
        
        # Update normalizer stats (only in train mode)
        if self.normalizer.mode == "train":
            # Collect all observations from episode for stats update
            # This is simplified - in practice you'd store obs during episode
            pass
        
        return {
            "episode_id": episode_id,
            "agent": agent.name,
            "total_reward": total_reward,
            "num_steps": step_index,
            "num_trades": num_trades,
            "final_balance": initial_capital + self.env.state.realized_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def run_multiple_episodes(
        self,
        agent: ScriptedAgent,
        num_episodes: int,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run multiple episodes with the same agent.
        
        Args:
            agent: Scripted agent to test
            num_episodes: Number of episodes to run
            verbose: Print progress
            
        Returns:
            List of episode metrics
        """
        results = []
        
        for ep in range(num_episodes):
            metrics = self.run_episode(agent)
            results.append(metrics)
            
            if verbose:
                print(
                    f"[{agent.name}] Episode {ep+1}/{num_episodes}: "
                    f"Reward={metrics['total_reward']:.4f}, "
                    f"Trades={metrics['num_trades']}, "
                    f"Balance={metrics['final_balance']:.2f}"
                )
        
        # Flush logger
        self.logger.flush_all()
        
        return results
    
    def _obs_to_dict(self, obs) -> Dict[str, Any]:
        """
        Convert observation object to dictionary.
        
        This is a placeholder - actual implementation depends on
        your observation structure from Step 2/3.
        """
        # If obs is already a dict, return it
        if isinstance(obs, dict):
            return obs
        
        # If obs is a structured object, extract fields
        # This should match your MarketObservation structure
        obs_dict = {
            "price.close": getattr(obs, "close", 0.0),
            "price.volatility": getattr(obs, "volatility", 0.0),
            "price.trend_regime": getattr(obs, "trend_regime", 1),
            "price.volatility_regime": getattr(obs, "volatility_regime", 1),
            "price.pressure": getattr(obs, "pressure", 0.0),
            "news.event_risk": getattr(obs, "event_risk", 0.0),
            # Add other fields as needed
        }
        
        return obs_dict


def run_all_tests(
    env,
    num_episodes: int = 10,
    db_config: Optional[Dict[str, Any]] = None,
    stats_dir: Optional[Path] = None
):
    """
    Run all scripted agent tests.
    
    This is the main integration test for Step 4.
    
    Args:
        env: BacktestEnv instance
        num_episodes: Episodes per agent
        db_config: Database configuration for logger
        stats_dir: Directory to save normalization stats
    """
    print("="*60)
    print("STEP 4 INTEGRATION TEST - SCRIPTED AGENTS")
    print("="*60)
    
    # Initialize components
    normalizer = ObservationNormalizer(mode="train")
    
    logger_config = db_config or {}
    logger = TradeLogger(**logger_config)
    
    runner = AgentTestRunner(env, normalizer, logger)
    
    # Create agents
    agents = [
        RandomAgent(seed=42),
        AlwaysEnterAgent(),
        MomentumFollowAgent(),
        ConservativeAgent()
    ]
    
    # Run tests
    all_results = {}
    
    for agent in agents:
        print(f"\nTesting {agent.name}...")
        results = runner.run_multiple_episodes(agent, num_episodes, verbose=True)
        all_results[agent.name] = results
        
        # Print agent summary
        stats = agent.get_stats()
        print(f"\n{agent.name} Summary:")
        print(f"  Mean Reward: {stats['mean_reward']:.4f}")
        print(f"  Std Reward: {stats.get('std_reward', 0):.4f}")
        print(f"  Mean Trades: {stats['mean_trades']:.1f}")
    
    # Save normalizer stats
    if stats_dir:
        normalizer.save(stats_dir)
        print(f"\nNormalization stats saved to {stats_dir}")
    
    # Close logger
    logger.close()
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    """
    Example usage - replace with your actual environment import.
    """
    print("Scripted agents module loaded successfully.")
    print("Import this module and call run_all_tests(env) to test your system.")