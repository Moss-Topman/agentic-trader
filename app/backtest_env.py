import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, replace

from action_space import Action, is_valid_action, ACTION_NAMES
from price_pipeline import PricePipeline
from news_pipeline import NewsPipeline
from market_observation import MarketObservation

logger = logging.getLogger(__name__)


@dataclass
class EnvState:
    current_index: int
    position_open: bool
    entry_price: Optional[float]
    entry_spread: Optional[float]
    realized_pnl: float
    episode_start_index: int
    prev_price: Optional[float]


class BacktestEnv:
    def __init__(self, data: pd.DataFrame, price_pipeline: PricePipeline, news_pipeline: NewsPipeline,
                 episode_length: int = 500, holding_penalty: float = 0.0001, initial_capital: float = 10000.0,
                 regimes: np.ndarray = None):
        """
        Backtest environment with regime-aware reward structure.
        
        Args:
            data: Market data with OHLC + spread
            price_pipeline: Price feature extraction
            news_pipeline: News feature extraction
            episode_length: Steps per episode
            holding_penalty: Cost per step (deprecated, not used in new reward structure)
            initial_capital: Starting capital (for statistics)
            regimes: Ground truth regime labels (0=chop, 1=trend) - NEW!
        """
        self.data = data
        self.price_pipeline = price_pipeline
        self.news_pipeline = news_pipeline
        self.episode_length = episode_length
        self.holding_penalty = holding_penalty
        self.initial_capital = initial_capital
        self.regimes = regimes  # NEW: Store ground truth regimes
        
        required = ["time", "open", "high", "low", "close", "spread"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
        
        min_data_length = 50 + episode_length
        if len(data) < min_data_length:
            raise ValueError(f"Data too short: need {min_data_length}, got {len(data)}")
        
        self.max_start_index = len(data) - episode_length - 1
        self.state: Optional[EnvState] = None
    
    def reset(self, start_index: Optional[int] = None) -> MarketObservation:
        if start_index is None:
            start_index = np.random.randint(50, self.max_start_index)
        
        initial_price = self.data.iloc[start_index]["close"]
        
        self.state = EnvState(
            current_index=start_index,
            position_open=False,
            entry_price=None,
            entry_spread=None,
            realized_pnl=0.0,
            episode_start_index=start_index,
            prev_price=initial_price,
        )
        
        return self._generate_observation()
    
    def _generate_observation(self) -> MarketObservation:
        """Generate observation with ground truth regime if available."""
        price_df = self.data.iloc[:self.state.current_index + 1]
        price_obs = self.price_pipeline.observe(price_df)
        
        # NEW: Add ground truth regime to observation
        if self.regimes is not None and self.state.current_index < len(self.regimes):
            # PriceObservation is frozen, so we need to create a new instance with true_regime
            price_obs = replace(price_obs, true_regime=int(self.regimes[self.state.current_index]))
        
        current_time = self.data.iloc[self.state.current_index]["time"]
        news_obs = self.news_pipeline.observe(current_time)
        
        return MarketObservation(price=price_obs, news=news_obs)
    
    def _compute_reward(self, action: Action, executed: bool) -> float:
        """
        PnL-based rewards with regime-conditional penalties.
        
        Reward structure teaches agent to:
        1. Trade in trends (regime=1) → no entry penalty, profits from price moves
        2. Avoid chop (regime=0) → entry penalty discourages trading
        3. Exit with discipline → spread costs only (returns already counted during HOLD)
        
        Key principles:
        - ENTER in chop: -0.002 (teach agent to sit in cash)
        - ENTER in trend: 0.0 (neutral, let profit/loss teach quality)
        - HOLD with position: step returns (unrealized PnL)
        - EXIT: -spreads only (realized PnL already counted)
        """
        if not executed:
            return 0.0
        
        current_row = self.data.iloc[self.state.current_index]
        current_price = current_row["close"]
        
        # Get current regime (0=chop, 1=trend)
        current_regime = 0  # Default to chop if regimes not provided
        if self.regimes is not None and self.state.current_index < len(self.regimes):
            current_regime = int(self.regimes[self.state.current_index])
        
        # === EXIT: Deduct spread costs only (returns already counted) ===
        if action == Action.EXIT:
            exit_spread = current_row["spread"]
            total_spread_cost = self.state.entry_spread + exit_spread
            return -total_spread_cost
        
        # === ENTER: Regime-conditional penalty ===
        if action == Action.ENTER_LONG:
            if current_regime == 0:  # Chop regime
                # Penalize entering in chop - teaches agent to sit in cash
                chop_entry_penalty = -0.010  # -1.0% penalty (was -0.002)
                return chop_entry_penalty
            else:  # Trend regime (regime=1)
                # Neutral in trends - let profit/loss determine if good entry
                return 0.0
        
        # === HOLD: Step returns if in position, with optional chop penalty ===
        if action == Action.HOLD:
            if self.state.position_open and self.state.prev_price is not None:
                # Reward step-by-step price changes while holding
                step_return = np.log(current_price / self.state.prev_price)
                
                # Optional: Small ongoing penalty for holding in chop
                # This teaches agent to exit quickly if chop is detected
                if current_regime == 0 and self.state.position_open:
                    chop_hold_penalty = -0.0005  # Stronger ongoing cost (was -0.0001)
                    return step_return + chop_hold_penalty
                
                return step_return
            else:
                # No reward for holding cash
                return 0.0
        
        return 0.0
    
    def step(self, action: int) -> Tuple[MarketObservation, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        action = Action(action)
        is_legal = is_valid_action(action, self.state.position_open)
        
        if not is_legal:
            obs = self._generate_observation()
            return obs, 0.0, False, {"illegal_action": True}
        
        current_row = self.data.iloc[self.state.current_index]
        current_price = current_row["close"]
        
        if action == Action.ENTER_LONG:
            self.state.position_open = True
            self.state.entry_price = current_price
            self.state.entry_spread = current_row["spread"]
        elif action == Action.EXIT:
            self.state.position_open = False
        
        reward = self._compute_reward(action, True)
        
        # Track actual PnL for statistics (not used in reward)
        if action == Action.EXIT:
            actual_pnl = reward  # reward already includes all costs
            self.state.realized_pnl += actual_pnl
        
        self.state.prev_price = current_price
        self.state.current_index += 1
        
        episode_end_index = self.state.episode_start_index + self.episode_length
        done = self.state.current_index >= episode_end_index
        
        if done and self.state.position_open:
            # Forced exit at episode end - only deduct spreads (returns already counted)
            forced_exit_row = self.data.iloc[self.state.current_index]
            exit_spread = forced_exit_row["spread"]
            total_spread_cost = self.state.entry_spread + exit_spread
            forced_exit_penalty = -total_spread_cost
            
            # Calculate actual PnL for statistics (includes full return)
            exit_price = forced_exit_row["close"]
            log_return = np.log(exit_price / self.state.entry_price)
            actual_pnl = log_return - total_spread_cost
            self.state.realized_pnl += actual_pnl
            
            # But only penalize spreads in reward (returns already accumulated)
            reward += forced_exit_penalty
            self.state.position_open = False
        
        obs = self._generate_observation()
        
        info = {
            "index": self.state.current_index,
            "position_open": self.state.position_open,
            "realized_pnl": self.state.realized_pnl,
            "action_taken": ACTION_NAMES[action],
            "illegal_action": False,
        }
        
        return obs, reward, done, info
    
    def get_state_summary(self) -> Dict[str, Any]:
        if self.state is None:
            return {"state": "not_initialized"}
        
        return {
            "current_index": self.state.current_index,
            "position_open": self.state.position_open,
            "entry_price": self.state.entry_price,
            "realized_pnl": self.state.realized_pnl,
            "episode_progress": f"{self.state.current_index - self.state.episode_start_index}/{self.episode_length}",
        }