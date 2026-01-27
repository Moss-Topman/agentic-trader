"""
Leakage Prevention Assertions
Prevents future information from contaminating observations during training.
All violations raise loud exceptions - no silent failures.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


class LeakageValidator:
    """
    Validates that no future information leaks into observations.
    
    All methods raise ValueError on detection of potential leakage.
    This is intentional - we want loud failures, not silent corruption.
    """
    
    @staticmethod
    def validate_timestamp(
        observation_time: datetime,
        current_time: datetime,
        episode_start_time: datetime
    ):
        """
        Validate observation timestamp is within valid range.
        
        Args:
            observation_time: Timestamp of the observation
            current_time: Current simulation time
            episode_start_time: When episode started
            
        Raises:
            ValueError: If observation time is invalid
        """
        if observation_time > current_time:
            raise ValueError(
                f"FUTURE LEAKAGE DETECTED: Observation time {observation_time} "
                f"is after current time {current_time}. "
                f"Delta: {(observation_time - current_time).total_seconds()}s"
            )
        
        if observation_time < episode_start_time:
            raise ValueError(
                f"TEMPORAL INCONSISTENCY: Observation time {observation_time} "
                f"is before episode start {episode_start_time}. "
                f"Delta: {(episode_start_time - observation_time).total_seconds()}s"
            )
    
    @staticmethod
    def validate_dataframe_bounds(
        df: pd.DataFrame,
        current_index: int,
        current_time: datetime,
        time_column: str = "time"
    ):
        """
        Validate DataFrame contains no future data.
        
        Args:
            df: DataFrame containing market data
            current_index: Current position in data
            current_time: Current simulation time
            time_column: Name of timestamp column
            
        Raises:
            ValueError: If DataFrame contains future data
        """
        if df.empty:
            raise ValueError("Cannot validate empty DataFrame")
        
        # Check: last row should match current time
        last_time = df.iloc[-1][time_column]
        if last_time > current_time:
            raise ValueError(
                f"FUTURE LEAKAGE: DataFrame last timestamp {last_time} "
                f"exceeds current time {current_time}"
            )
        
        # Check: DataFrame shouldn't extend beyond current index
        expected_length = current_index + 1
        actual_length = len(df)
        
        if actual_length > expected_length:
            raise ValueError(
                f"FUTURE LEAKAGE: DataFrame has {actual_length} rows "
                f"but current index is {current_index} (expected max {expected_length} rows)"
            )
        
        # Check: timestamps should be sorted
        if not df[time_column].is_monotonic_increasing:
            raise ValueError(
                f"TEMPORAL INCONSISTENCY: Timestamps in DataFrame are not sorted. "
                f"This could indicate data corruption."
            )
    
    @staticmethod
    def validate_rolling_window(
        current_index: int,
        window_size: int,
        lookback_start: int
    ):
        """
        Validate rolling window doesn't access future data.
        
        Args:
            current_index: Current position in data
            window_size: Size of rolling window
            lookback_start: Starting index for window computation
            
        Raises:
            ValueError: If window extends into future
        """
        lookback_end = lookback_start + window_size
        
        if lookback_end > current_index + 1:
            raise ValueError(
                f"FUTURE LEAKAGE: Rolling window extends to index {lookback_end} "
                f"but current index is {current_index}"
            )
        
        if lookback_start < 0:
            raise ValueError(
                f"INVALID WINDOW: Lookback start {lookback_start} is negative"
            )
    
    @staticmethod
    def validate_feature_computation(
        feature_time: datetime,
        current_time: datetime,
        feature_name: str
    ):
        """
        Validate computed feature doesn't use future information.
        
        Args:
            feature_time: Latest timestamp used in feature computation
            current_time: Current simulation time
            feature_name: Name of feature for error messages
            
        Raises:
            ValueError: If feature uses future data
        """
        if feature_time > current_time:
            raise ValueError(
                f"FUTURE LEAKAGE in {feature_name}: Feature computed using data "
                f"from {feature_time}, which is after current time {current_time}"
            )
    
    @staticmethod
    def validate_observation_values(obs_dict: dict, field_name: str = "observation"):
        """
        Validate observation contains no NaN or infinite values.
        
        Args:
            obs_dict: Dictionary of observation values
            field_name: Name for error messages
            
        Raises:
            ValueError: If any value is NaN or infinite
        """
        for key, value in obs_dict.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    raise ValueError(
                        f"INVALID VALUE in {field_name}.{key}: "
                        f"Value is {value} (NaN or Inf)"
                    )


class LeakageDebugger:
    """
    Debugging utilities for investigating potential leakage.
    
    These methods log detailed information about data access patterns
    to help diagnose leakage issues.
    """
    
    def __init__(self, enable_logging: bool = False):
        """
        Initialize debugger.
        
        Args:
            enable_logging: If True, log all validations
        """
        self.enable_logging = enable_logging
        self.validation_log = []
    
    def log_validation(
        self,
        validation_type: str,
        current_time: datetime,
        data_range: str,
        success: bool = True
    ):
        """
        Log a validation check.
        
        Args:
            validation_type: Type of validation performed
            current_time: Current simulation time
            data_range: Description of data range checked
            success: Whether validation passed
        """
        if not self.enable_logging:
            return
        
        entry = {
            "type": validation_type,
            "current_time": current_time,
            "data_range": data_range,
            "success": success,
            "timestamp": datetime.now()
        }
        
        self.validation_log.append(entry)
    
    def get_validation_summary(self) -> dict:
        """Get summary of all validations performed."""
        if not self.validation_log:
            return {"total": 0, "failures": 0}
        
        total = len(self.validation_log)
        failures = sum(1 for entry in self.validation_log if not entry["success"])
        
        return {
            "total": total,
            "failures": failures,
            "success_rate": (total - failures) / total if total > 0 else 0.0
        }
    
    def clear_log(self):
        """Clear validation log."""
        self.validation_log.clear()


# Convenience functions for common validation patterns

def assert_no_future_leakage(
    df: pd.DataFrame,
    current_index: int,
    current_time: datetime,
    time_column: str = "time"
):
    """
    Comprehensive check for future leakage in DataFrame.
    
    This is the main validation function to call before generating observations.
    
    Args:
        df: Market data DataFrame
        current_index: Current position in data
        current_time: Current simulation time
        time_column: Name of timestamp column
        
    Raises:
        ValueError: If any leakage detected
    """
    validator = LeakageValidator()
    
    # Validate DataFrame bounds
    validator.validate_dataframe_bounds(df, current_index, current_time, time_column)
    
    # Validate last timestamp
    last_time = df.iloc[-1][time_column]
    validator.validate_timestamp(last_time, current_time, df.iloc[0][time_column])


def assert_feature_integrity(
    feature_value: float,
    feature_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
):
    """
    Validate feature value is within expected bounds.
    
    Args:
        feature_value: Computed feature value
        feature_name: Name of feature
        min_value: Minimum expected value (optional)
        max_value: Maximum expected value (optional)
        
    Raises:
        ValueError: If feature is invalid
    """
    if not np.isfinite(feature_value):
        raise ValueError(
            f"INVALID FEATURE {feature_name}: Value is {feature_value}"
        )
    
    if min_value is not None and feature_value < min_value:
        raise ValueError(
            f"FEATURE OUT OF BOUNDS {feature_name}: "
            f"Value {feature_value} is below minimum {min_value}"
        )
    
    if max_value is not None and feature_value > max_value:
        raise ValueError(
            f"FEATURE OUT OF BOUNDS {feature_name}: "
            f"Value {feature_value} exceeds maximum {max_value}"
        )