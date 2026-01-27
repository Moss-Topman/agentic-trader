"""
Trade Logger
Logs every decision to TimescaleDB for post-mortem analysis.
Implements batching, fallback to CSV, and graceful metric handling.
"""

import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import csv
import numpy as np
from contextlib import contextmanager


class TradeLogger:
    """
    Logs all trading decisions and episode metrics to database.
    
    Features:
    - Batch writes for performance
    - Automatic reconnection on DB failure
    - CSV fallback if DB unavailable
    - Graceful handling of missing metrics (e.g., Sharpe ratio)
    """
    
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "agentic_db",
        db_user: str = "agentic",
        db_password: str = "agenticpass",
        batch_size: int = 100,
        csv_fallback_dir: Optional[Path] = None
    ):
        """
        Initialize trade logger.
        
        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            batch_size: Number of steps to buffer before writing
            csv_fallback_dir: Directory for CSV fallback if DB fails
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password
        }
        
        self.batch_size = batch_size
        self.csv_fallback_dir = Path(csv_fallback_dir) if csv_fallback_dir else None
        
        # Buffers for batch writing
        self.step_buffer: List[Dict[str, Any]] = []
        self.episode_buffer: List[Dict[str, Any]] = []
        
        # Connection management
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.db_available = True
        
        # Statistics
        self.write_count = 0
        self.total_steps_logged = 0
        self.total_episodes_logged = 0
        
        # Initialize connection
        self._connect()
        
        # Initialize CSV fallback if needed
        if self.csv_fallback_dir:
            self.csv_fallback_dir.mkdir(parents=True, exist_ok=True)
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.db_available = True
        except Exception as e:
            print(f"WARNING: Failed to connect to database: {e}")
            print("Will fallback to CSV logging if configured")
            self.db_available = False
            self.conn = None
    
    def _reconnect(self):
        """Attempt to reconnect to database."""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        
        self._connect()
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor with auto-reconnect."""
        if not self.db_available:
            raise RuntimeError("Database not available")
        
        try:
            if self.conn is None or self.conn.closed:
                self._reconnect()
            
            cursor = self.conn.cursor()
            yield cursor
            self.conn.commit()
            cursor.close()
            
        except Exception as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def log_step(
        self,
        episode_id: int,
        step_index: int,
        timestamp: datetime,
        observation: Dict[str, Any],
        action: int,
        action_was_legal: bool,
        reward: float,
        position_open: bool,
        entry_price: Optional[float],
        realized_pnl: float
    ):
        """
        Log a single environment step.
        
        Args:
            episode_id: Episode identifier
            step_index: Step within episode
            timestamp: Step timestamp
            observation: Observation dictionary (should contain normalized values)
            action: Action taken (0=HOLD, 1=ENTER, 2=EXIT)
            action_was_legal: Whether action was legal
            reward: Reward received
            position_open: Whether position is open after this step
            entry_price: Entry price if position open, else None
            realized_pnl: Realized PnL for this step
        """
        # Convert all numeric values to native Python types
        step_data = {
            "episode_id": int(episode_id),
            "step_index": int(step_index),
            "timestamp": timestamp,
            
            # Observation fields (extract from dict, convert to native types)
            "obs_close": float(observation.get("price.close", 0.0)),
            "obs_volatility": float(observation.get("price.volatility", 0.0)),
            "obs_pressure": float(observation.get("price.pressure", 0.0)),
            "obs_trend_regime": int(observation.get("price.trend_regime", 0)),
            "obs_vol_regime": int(observation.get("price.volatility_regime", 0)),
            "obs_event_risk": float(observation.get("news.event_risk", 0.0)),
            "obs_is_warmup": bool(observation.get("price.is_warmup", False)),
            
            # Action
            "action": int(action),
            "action_was_legal": bool(action_was_legal),
            
            # Reward & State
            "reward": float(reward),
            "position_open": bool(position_open),
            "entry_price": float(entry_price) if entry_price is not None else None,
            "realized_pnl": float(realized_pnl)
        }
        
        self.step_buffer.append(step_data)
        
        # Flush if buffer is full
        if len(self.step_buffer) >= self.batch_size:
            self._flush_steps()
    
    def log_episode(
        self,
        episode_id: int,
        start_time: datetime,
        end_time: datetime,
        total_reward: float,
        realized_pnl: float,
        num_steps: int,
        num_trades: int,
        max_drawdown: float,
        sharpe_ratio: Optional[float],
        forced_liquidation: bool
    ):
        """
        Log episode summary.
        
        Args:
            episode_id: Episode identifier
            start_time: Episode start timestamp
            end_time: Episode end timestamp
            total_reward: Cumulative reward
            realized_pnl: Total realized PnL
            num_steps: Number of steps in episode
            num_trades: Number of trades executed
            max_drawdown: Maximum drawdown during episode
            sharpe_ratio: Sharpe ratio (None if insufficient data)
            forced_liquidation: Whether position was force-closed at end
        """
        episode_data = {
            "episode_id": int(episode_id),
            "start_time": start_time,
            "end_time": end_time,
            "total_reward": float(total_reward),
            "realized_pnl": float(realized_pnl),
            "num_steps": int(num_steps),
            "num_trades": int(num_trades),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
            "forced_liquidation": bool(forced_liquidation)
        }
        
        self.episode_buffer.append(episode_data)
        
        # Flush episode immediately (episodes are less frequent)
        self._flush_episodes()
    
    def _flush_steps(self):
        """Write buffered steps to database."""
        if not self.step_buffer:
            return
        
        try:
            if self.db_available:
                self._write_steps_to_db()
            elif self.csv_fallback_dir:
                self._write_steps_to_csv()
            
            self.step_buffer.clear()
            self.write_count += 1
            
        except Exception as e:
            print(f"ERROR: Failed to flush steps: {e}")
            
            # Try CSV fallback
            if self.csv_fallback_dir:
                try:
                    self._write_steps_to_csv()
                    self.step_buffer.clear()
                except Exception as csv_error:
                    print(f"ERROR: CSV fallback also failed: {csv_error}")
    
    def _write_steps_to_db(self):
        """Write step buffer to database."""
        sql = """
            INSERT INTO step_log (
                episode_id, step_index, timestamp,
                obs_close, obs_volatility, obs_pressure,
                obs_trend_regime, obs_vol_regime, obs_event_risk, obs_is_warmup,
                action, action_was_legal,
                reward, position_open, entry_price, realized_pnl
            ) VALUES (
                %(episode_id)s, %(step_index)s, %(timestamp)s,
                %(obs_close)s, %(obs_volatility)s, %(obs_pressure)s,
                %(obs_trend_regime)s, %(obs_vol_regime)s, %(obs_event_risk)s, %(obs_is_warmup)s,
                %(action)s, %(action_was_legal)s,
                %(reward)s, %(position_open)s, %(entry_price)s, %(realized_pnl)s
            )
        """
        
        with self._get_cursor() as cursor:
            execute_batch(cursor, sql, self.step_buffer)
            self.total_steps_logged += len(self.step_buffer)
    
    def _write_steps_to_csv(self):
        """Write step buffer to CSV fallback."""
        if not self.csv_fallback_dir:
            return
        
        csv_path = self.csv_fallback_dir / "step_log.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            if self.step_buffer:
                fieldnames = list(self.step_buffer[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(self.step_buffer)
                self.total_steps_logged += len(self.step_buffer)
    
    def _flush_episodes(self):
        """Write buffered episodes to database."""
        if not self.episode_buffer:
            return
        
        try:
            if self.db_available:
                self._write_episodes_to_db()
            elif self.csv_fallback_dir:
                self._write_episodes_to_csv()
            
            self.episode_buffer.clear()
            
        except Exception as e:
            print(f"ERROR: Failed to flush episodes: {e}")
            
            # Try CSV fallback
            if self.csv_fallback_dir:
                try:
                    self._write_episodes_to_csv()
                    self.episode_buffer.clear()
                except Exception as csv_error:
                    print(f"ERROR: CSV fallback also failed: {csv_error}")
    
    def _write_episodes_to_db(self):
        """Write episode buffer to database."""
        sql = """
            INSERT INTO episode_log (
                episode_id, start_time, end_time,
                total_reward, realized_pnl, num_steps, num_trades,
                max_drawdown, sharpe_ratio, forced_liquidation
            ) VALUES (
                %(episode_id)s, %(start_time)s, %(end_time)s,
                %(total_reward)s, %(realized_pnl)s, %(num_steps)s, %(num_trades)s,
                %(max_drawdown)s, %(sharpe_ratio)s, %(forced_liquidation)s
            )
        """
        
        with self._get_cursor() as cursor:
            execute_batch(cursor, sql, self.episode_buffer)
            self.total_episodes_logged += len(self.episode_buffer)
    
    def _write_episodes_to_csv(self):
        """Write episode buffer to CSV fallback."""
        if not self.csv_fallback_dir:
            return
        
        csv_path = self.csv_fallback_dir / "episode_log.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            if self.episode_buffer:
                fieldnames = list(self.episode_buffer[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(self.episode_buffer)
                self.total_episodes_logged += len(self.episode_buffer)
    
    def flush_all(self):
        """Force flush all buffers."""
        self._flush_steps()
        self._flush_episodes()
    
    def close(self):
        """Flush buffers and close connection."""
        self.flush_all()
        
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        return (
            f"TradeLogger(\n"
            f"  db_available={self.db_available},\n"
            f"  total_steps_logged={self.total_steps_logged},\n"
            f"  total_episodes_logged={self.total_episodes_logged},\n"
            f"  write_count={self.write_count}\n"
            f")"
        )


def compute_sharpe_ratio(returns: List[float], periods_per_year: int = 252) -> Optional[float]:
    """
    Compute Sharpe ratio from returns.
    
    Handles edge cases gracefully:
    - Returns None if insufficient data
    - Returns None if variance is zero
    - Returns None on any numerical error
    
    Args:
        returns: List of period returns
        periods_per_year: Annualization factor
        
    Returns:
        Sharpe ratio or None if cannot be computed
    """
    try:
        if len(returns) < 2:
            return None
        
        returns_array = np.array(returns)
        
        # Remove NaN/inf
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) < 2:
            return None
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0 or not np.isfinite(std_return):
            return None
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        if not np.isfinite(sharpe):
            return None
        
        return float(sharpe)
        
    except Exception:
        # Any error → return None
        return None


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """
    Compute maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values over time
        
    Returns:
        Maximum drawdown as positive fraction (0.0 to 1.0)
    """
    try:
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        
        # Compute running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Compute drawdown at each point
        drawdown = (running_max - equity_array) / running_max
        
        # Maximum drawdown
        max_dd = np.max(drawdown)
        
        if not np.isfinite(max_dd):
            return 0.0
        
        return float(max_dd)
        
    except Exception:
        return 0.0