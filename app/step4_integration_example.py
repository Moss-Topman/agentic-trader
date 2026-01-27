"""
Step 4 Integration Example
Shows how all components work together in a training loop.

This is NOT the actual training code (that's Step 5).
This is a blueprint showing how to integrate all Step 4 components.
"""

from pathlib import Path
import numpy as np
from datetime import datetime

from running_stats import ObservationNormalizer
from action_masking import Action, ActionMasker, MaskingStats
from trade_logger import TradeLogger, compute_sharpe_ratio, compute_max_drawdown
from leakage_prevention import LeakageValidator
from scripted_agents import (
    RandomAgent,
    AlwaysEnterAgent,
    MomentumFollowAgent,
    ConservativeAgent,
    AgentTestRunner
)


class Step4TrainingPipeline:
    """
    Complete training pipeline with all Step 4 components integrated.
    
    This demonstrates the correct flow for training with:
    - Observation normalization
    - Action masking
    - Leakage prevention
    - Complete logging
    """
    
    def __init__(
        self,
        env,  # Your BacktestEnv from Step 3
        stats_dir: Path,
        log_dir: Path,
        db_config: dict
    ):
        """
        Initialize training pipeline.
        
        Args:
            env: BacktestEnv instance
            stats_dir: Directory to save normalization statistics
            log_dir: Directory for CSV fallback logs
            db_config: Database configuration for TradeLogger
        """
        self.env = env
        self.stats_dir = Path(stats_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.normalizer = ObservationNormalizer(mode="train")
        self.masking_stats = MaskingStats()
        self.validator = LeakageValidator()
        self.logger = TradeLogger(**db_config, csv_fallback_dir=log_dir)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
    
    def train_episode(self, policy_fn):
        """
        Run one training episode.
        
        Args:
            policy_fn: Function that takes (obs_dict, position_open) and returns action_probs
            
        Returns:
            Episode metrics dictionary
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        
        # Episode tracking
        episode_id = self.episode_count
        self.episode_count += 1
        
        start_time = datetime.now()
        step_index = 0
        total_reward = 0.0
        num_trades = 0
        equity_curve = [self.env.state.balance]
        returns = []
        
        episode_observations = []  # For stats update
        
        while not done:
            # ================================================================
            # 1. CONVERT OBSERVATION TO DICT (depends on your obs structure)
            # ================================================================
            obs_dict = self._obs_to_dict(obs)
            
            # ================================================================
            # 2. LEAKAGE VALIDATION
            # ================================================================
            # Validate no future data (if you store timestamp in obs)
            if "timestamp" in obs_dict:
                self.validator.validate_timestamp(
                    observation_time=obs_dict["timestamp"],
                    current_time=self.env.state.current_time,
                    episode_start_time=start_time
                )
            
            # Validate values are finite
            self.validator.validate_observation_values(obs_dict)
            
            # ================================================================
            # 3. NORMALIZE OBSERVATION
            # ================================================================
            obs_normalized = self.normalizer.normalize(obs_dict)
            
            # Store for stats update later
            episode_observations.append(obs_dict)
            
            # ================================================================
            # 4. POLICY FORWARD PASS (get raw action probabilities)
            # ================================================================
            # This is where your neural network would go in Step 5
            # For now, policy_fn is a placeholder
            action_probs_raw = policy_fn(obs_normalized, self.env.state.position_open)
            
            # ================================================================
            # 5. ACTION MASKING
            # ================================================================
            action_probs_masked = ActionMasker.mask_probabilities(
                action_probs_raw,
                position_open=self.env.state.position_open
            )
            
            # ================================================================
            # 6. SAMPLE ACTION
            # ================================================================
            action = ActionMasker.sample_masked_action(
                action_probs_raw,
                position_open=self.env.state.position_open
            )
            
            # Track masking stats
            self.masking_stats.record_sample(
                action_probs_raw,
                action,
                self.env.state.position_open
            )
            
            # ================================================================
            # 7. EXECUTE ACTION
            # ================================================================
            next_obs, reward, done, info = self.env.step(action)
            
            # ================================================================
            # 8. LOG STEP
            # ================================================================
            self.logger.log_step(
                episode_id=episode_id,
                step_index=step_index,
                timestamp=datetime.now(),
                observation=obs_normalized,  # Log normalized values
                action=int(action),
                action_was_legal=True,  # We masked, so always legal
                reward=reward,
                position_open=self.env.state.position_open,
                entry_price=self.env.state.entry_price,
                realized_pnl=info.get("realized_pnl", 0.0)
            )
            
            # ================================================================
            # 9. UPDATE METRICS
            # ================================================================
            total_reward += reward
            if action == Action.ENTER_LONG:
                num_trades += 1
            
            equity_curve.append(self.env.state.balance)
            
            if len(equity_curve) >= 2:
                ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(ret)
            
            # ================================================================
            # 10. PREPARE FOR NEXT STEP
            # ================================================================
            obs = next_obs
            step_index += 1
            self.total_steps += 1
        
        # ====================================================================
        # END OF EPISODE
        # ====================================================================
        
        # Update normalizer with episode observations (AFTER episode ends)
        for obs_dict in episode_observations:
            self.normalizer.update(obs_dict)
        
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
            realized_pnl=self.env.state.balance - self.env.initial_balance,
            num_steps=step_index,
            num_trades=num_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            forced_liquidation=info.get("forced_liquidation", False)
        )
        
        # Return metrics
        return {
            "episode_id": episode_id,
            "total_reward": total_reward,
            "num_steps": step_index,
            "num_trades": num_trades,
            "final_balance": self.env.state.balance,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def validate_episode(self, policy_fn):
        """
        Run validation episode (stats frozen).
        
        Args:
            policy_fn: Policy function
            
        Returns:
            Episode metrics
        """
        # Switch to eval mode (freeze stats)
        self.normalizer.set_mode("eval")
        
        try:
            metrics = self.train_episode(policy_fn)
            return metrics
        finally:
            # Always switch back to train mode
            self.normalizer.set_mode("train")
    
    def save_checkpoint(self):
        """Save normalizer stats."""
        checkpoint_dir = self.stats_dir / f"episode_{self.episode_count}"
        self.normalizer.save(checkpoint_dir)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load normalizer stats from checkpoint."""
        self.normalizer = ObservationNormalizer.load(checkpoint_dir, mode="train")
        print(f"Checkpoint loaded from {checkpoint_dir}")
    
    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        return {
            "total_episodes": self.episode_count,
            "total_steps": self.total_steps,
            "masking_stats": {
                "total_samples": self.masking_stats.total_samples,
                "illegal_attempt_rate": self.masking_stats.get_illegal_attempt_rate(),
                "action_distribution": self.masking_stats.get_action_distribution()
            },
            "logger_stats": {
                "steps_logged": self.logger.total_steps_logged,
                "episodes_logged": self.logger.total_episodes_logged,
                "db_available": self.logger.db_available
            }
        }
    
    def close(self):
        """Clean shutdown."""
        self.logger.close()
        self.save_checkpoint()
    
    def _obs_to_dict(self, obs) -> dict:
        """
        Convert observation to dictionary.
        
        Replace this with your actual observation structure.
        """
        # Placeholder - replace with actual conversion
        if isinstance(obs, dict):
            return obs
        
        # Example conversion if obs is a custom object
        return {
            "price.close": getattr(obs, "close", 0.0),
            "price.volatility": getattr(obs, "volatility", 0.0),
            "price.trend_regime": getattr(obs, "trend_regime", 1),
            "price.volatility_regime": getattr(obs, "volatility_regime", 1),
            "price.pressure": getattr(obs, "pressure", 0.0),
            "news.event_risk": getattr(obs, "event_risk", 0.0),
        }


def example_random_policy(obs_dict: dict, position_open: bool) -> np.ndarray:
    """
    Example random policy for demonstration.
    
    In Step 5, this will be replaced by a neural network.
    """
    return np.array([0.33, 0.34, 0.33])  # Uniform distribution


def main():
    """
    Example usage - replace with your actual setup.
    """
    print("="*60)
    print("STEP 4 INTEGRATION EXAMPLE")
    print("="*60)
    
    # ========================================================================
    # SETUP (replace with your actual environment)
    # ========================================================================
    
    # This is a placeholder - import your actual environment
    # from your_environment import BacktestEnv
    # env = BacktestEnv(...)
    
    print("\nTo use this integration:")
    print("1. Import your BacktestEnv from Step 3")
    print("2. Configure database connection")
    print("3. Create Step4TrainingPipeline")
    print("4. Run training episodes")
    
    # ========================================================================
    # EXAMPLE CONFIGURATION
    # ========================================================================
    
    config = {
        "stats_dir": Path("./checkpoints"),
        "log_dir": Path("./logs"),
        "db_config": {
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "agentic_db",
            "db_user": "agentic",
            "db_password": "agenticpass",
            "batch_size": 100,
        }
    }
    
    print(f"\nConfiguration:")
    print(f"  Stats directory: {config['stats_dir']}")
    print(f"  Log directory: {config['log_dir']}")
    print(f"  Database: {config['db_config']['db_name']}")
    
    # ========================================================================
    # EXAMPLE TRAINING LOOP (PSEUDO-CODE)
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE TRAINING LOOP (PSEUDO-CODE)")
    print("="*60)
    
    example_code = '''
# Initialize pipeline
pipeline = Step4TrainingPipeline(
    env=your_backtest_env,
    stats_dir=config["stats_dir"],
    log_dir=config["log_dir"],
    db_config=config["db_config"]
)

# Training loop
for episode in range(num_episodes):
    # Train episode
    metrics = pipeline.train_episode(policy_fn=your_policy)
    
    print(f"Episode {episode}: Reward={metrics['total_reward']:.4f}")
    
    # Validation every N episodes
    if episode % 10 == 0:
        val_metrics = pipeline.validate_episode(policy_fn=your_policy)
        print(f"Validation: Reward={val_metrics['total_reward']:.4f}")
    
    # Save checkpoint
    if episode % 100 == 0:
        pipeline.save_checkpoint()

# Training stats
stats = pipeline.get_training_stats()
print(stats)

# Cleanup
pipeline.close()
'''
    
    print(example_code)
    
    print("\n" + "="*60)
    print("Next: Implement your policy function in Step 5")
    print("="*60)


if __name__ == "__main__":
    main()