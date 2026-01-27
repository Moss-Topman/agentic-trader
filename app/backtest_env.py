# backtest_env.py
# Minimal backtesting environment for agentic trading
# Brutal. Honest. Learnable. Debuggable.

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

from action_space import Action, is_valid_action, ACTION_NAMES
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from market_observation import MarketObservation

logger = logging.getLogger(__name__)


@dataclass
class EnvState:
    """
    Environment state variables.
    What the world remembers, not what the agent sees.
    """
    current_index: int              # Position in data array
    position_open: bool             # Currently holding a position
    entry_price: Optional[float]    # Price at which position was entered
    entry_spread: Optional[float]   # Spread paid on entry (for cost accounting)
    realized_pnl: float             # Cumulative realized PnL for episode
    episode_start_index: int        # Starting index of current episode


class BacktestEnv:
    """
    Minimal backtesting environment for long-only directional learning.
    
    Design principles:
    - One position maximum
    - 100% capital allocation on entry
    - Spread-based transaction costs
    - Mandatory holding time penalty
    - Fixed-length episodes with forced close
    - No future information leakage
    
    Reward structure:
    - EXIT: realized log return - transaction costs
    - ENTER_LONG: 0.0 (no immediate reward)
    - HOLD (while in position): -0.0001 per candle (time penalty)
    - HOLD (while flat): 0
    - Illegal actions: 0 (no-op with warning)
    
    Episode termination:
    - Fixed length (episode_length candles)
    - Force close open positions at end
    - Apply full transaction costs on forced close
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        price_pipeline: PricePipeline,
        news_pipeline: NewsPipeline,
        episode_length: int = 500,
        holding_penalty: float = 0.0001,
        initial_capital: float = 10000.0,
    ):
        """
        Initialize backtesting environment.
        
        Args:
            data: DataFrame with OHLCV + spread columns
            price_pipeline: Configured price perception pipeline
            news_pipeline: Configured news perception pipeline
            episode_length: Fixed candles per episode
            holding_penalty: Cost per candle while in position
            initial_capital: Starting capital (for accounting)
        """
        self.data = data
        self.price_pipeline = price_pipeline
        self.news_pipeline = news_pipeline
        self.episode_length = episode_length
        self.holding_penalty = holding_penalty
        self.initial_capital = initial_capital
        
        # Validate data has required columns
        required = ["time", "open", "high", "low", "close", "spread"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
        
        # Calculate valid episode start range
        # Need: warmup (50) + episode_length
        min_data_length = 50 + episode_length
        if len(data) < min_data_length:
            raise ValueError(
                f"Data too short: need {min_data_length} candles, "
                f"got {len(data)}"
            )
        
        self.max_start_index = len(data) - episode_length - 1
        
        # State (initialized by reset)
        self.state: Optional[EnvState] = None
    
    def reset(self, start_index: Optional[int] = None) -> MarketObservation:
        """
        Reset environment to start of new episode.
        
        Args:
            start_index: Optional fixed start (for reproducibility)
                        If None, random start is chosen
        
        Returns:
            Initial observation
        """
        if start_index is None:
            # Random start (ensuring warmup + episode fits)
            start_index = np.random.randint(50, self.max_start_index)
        
        # Initialize state
        self.state = EnvState(
            current_index=start_index,
            position_open=False,
            entry_price=None,
            entry_spread=None,
            realized_pnl=0.0,
            episode_start_index=start_index,
        )
        
        # Generate initial observation
        obs = self._generate_observation()
        
        return obs
    
    def _generate_observation(self) -> MarketObservation:
        """
        Generate market observation at current state.
        
        Returns combined price + news observation.
        """
        # Get price observation (needs full history up to current index)
        price_df = self.data.iloc[:self.state.current_index + 1]
        price_obs = self.price_pipeline.observe(price_df)
        
        # Get news observation (needs current timestamp)
        current_time = self.data.iloc[self.state.current_index]["time"]
        news_obs = self.news_pipeline.observe(current_time)
        
        # Combine
        return MarketObservation(
            price=price_obs,
            news=news_obs
        )
    
    def _compute_reward(self, action: Action, executed: bool) -> float:
        """
        Compute reward for action taken.
        
        Reward structure (EXPLICIT ORDER MATTERS):
        1. Illegal action (executed=False) → 0.0
        2. Action.EXIT → realized PnL
        3. Action.ENTER_LONG → 0.0 (no immediate reward)
        4. Action.HOLD + position_open → -holding_penalty
        5. Action.HOLD + flat → 0.0
        
        Args:
            action: Action taken
            executed: Whether action was actually executed
            
        Returns:
            Scalar reward
        """
        # Rule 1: Illegal actions
        if not executed:
            return 0.0
        
        # Rule 2: EXIT - compute realized PnL
        if action == Action.EXIT:
            current_row = self.data.iloc[self.state.current_index]
            exit_price = current_row["close"]
            exit_spread = current_row["spread"]
            
            log_return = np.log(exit_price / self.state.entry_price)
            total_spread_cost = self.state.entry_spread + exit_spread
            realized_pnl = log_return - total_spread_cost
            
            return realized_pnl
        
        # Rule 3: ENTER_LONG - no immediate reward
        if action == Action.ENTER_LONG:
            return 0.0
        
        # Rule 4 & 5: HOLD
        if action == Action.HOLD:
            if self.state.position_open:
                return -self.holding_penalty
            else:
                return 0.0
        
        # Fallback (should never reach)
        logger.warning(f"Unexpected action in _compute_reward: {action}")
        return 0.0
    
    def step(self, action: int) -> Tuple[MarketObservation, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Step sequence (CRITICAL ORDER):
        1. Validate action
        2. Execute trade logic at current candle close
        3. Compute reward
        4. Advance time
        5. Generate next observation
        6. Check termination
        
        Args:
            action: Action enum value
            
        Returns:
            (observation, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        action = Action(action)  # Convert int to enum
        
        # Step 1: Validate action
        is_legal = is_valid_action(action, self.state.position_open)
        
        if not is_legal:
            # Illegal action → no-op with warning
            logger.warning(
                f"Illegal action {ACTION_NAMES[action]} at index {self.state.current_index} "
                f"(position_open={self.state.position_open})"
            )
            
            # Return current state, zero reward, not done
            obs = self._generate_observation()
            return obs, 0.0, False, {"illegal_action": True}
        
        # Step 2: Execute trade logic
        current_row = self.data.iloc[self.state.current_index]
        executed = True
        
        if action == Action.ENTER_LONG:
            # Enter long position
            self.state.position_open = True
            self.state.entry_price = current_row["close"]
            self.state.entry_spread = current_row["spread"]
            
        elif action == Action.EXIT:
            # Exit position
            self.state.position_open = False
            # (PnL computed in reward function)
            
        elif action == Action.HOLD:
            # Do nothing
            pass
        
        # Step 3: Compute reward BEFORE advancing time
        # (Reward for action at time t depends on state at time t)
        reward = self._compute_reward(action, executed)
        
        # Update realized PnL if exited
        if action == Action.EXIT and executed:
            self.state.realized_pnl += reward
        
        # Step 4: Advance time
        self.state.current_index += 1
        
        # Step 5: Check termination
        episode_end_index = self.state.episode_start_index + self.episode_length
        done = self.state.current_index >= episode_end_index
        
        if done and self.state.position_open:
            # Force close at episode end
            logger.info(
                f"Force closing position at episode end "
                f"(index {self.state.current_index})"
            )
            
            # Compute forced exit reward
            forced_exit_row = self.data.iloc[self.state.current_index]
            exit_price = forced_exit_row["close"]
            exit_spread = forced_exit_row["spread"]
            
            log_return = np.log(exit_price / self.state.entry_price)
            total_spread_cost = self.state.entry_spread + exit_spread
            forced_exit_pnl = log_return - total_spread_cost
            
            # Add to episode total and current reward
            self.state.realized_pnl += forced_exit_pnl
            reward += forced_exit_pnl
            
            # Close position
            self.state.position_open = False
        
        # Step 6: Generate next observation (if not done)
        if not done:
            obs = self._generate_observation()
        else:
            # Episode over, return last observation
            obs = self._generate_observation()
        
        # Info dict for debugging
        info = {
            "index": self.state.current_index,
            "position_open": self.state.position_open,
            "realized_pnl": self.state.realized_pnl,
            "action_taken": ACTION_NAMES[action],
            "illegal_action": False,
        }
        
        return obs, reward, done, info
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get current state for debugging.
        
        Returns:
            Dict with state variables
        """
        if self.state is None:
            return {"state": "not_initialized"}
        
        return {
            "current_index": self.state.current_index,
            "position_open": self.state.position_open,
            "entry_price": self.state.entry_price,
            "realized_pnl": self.state.realized_pnl,
            "episode_progress": f"{self.state.current_index - self.state.episode_start_index}/{self.episode_length}",
        }