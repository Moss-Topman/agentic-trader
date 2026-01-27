"""
Running Statistics Normalizer
Implements Welford's algorithm for numerically stable online mean/std computation.
Supports train/eval mode to prevent contamination during validation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union


class RunningStats:
    """
    Online computation of mean and standard deviation using Welford's algorithm.
    
    Supports two modes:
    - train: allows updates to statistics
    - eval: freezes statistics, raises exception on update attempts
    
    Attributes:
        count: Number of samples seen
        mean: Running mean
        M2: Sum of squared differences from mean (for variance computation)
        mode: "train" or "eval"
    """
    
    def __init__(self, mode: str = "train", epsilon: float = 1e-8):
        """
        Initialize running statistics tracker.
        
        Args:
            mode: "train" or "eval" - controls whether updates are allowed
            epsilon: Small constant to prevent division by zero in normalization
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"Mode must be 'train' or 'eval', got '{mode}'")
        
        self.mode = mode
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def set_mode(self, mode: str):
        """Set the mode (train/eval)."""
        if mode not in ["train", "eval"]:
            raise ValueError(f"Mode must be 'train' or 'eval', got '{mode}'")
        self.mode = mode
    
    def update(self, value: float):
        """
        Update statistics with a new value using Welford's algorithm.
        
        Args:
            value: New observation value
            
        Raises:
            RuntimeError: If called in eval mode
            ValueError: If value is NaN or infinite
        """
        if self.mode == "eval":
            raise RuntimeError(
                "Cannot update RunningStats in eval mode. "
                "Set mode='train' before updating."
            )
        
        if not np.isfinite(value):
            raise ValueError(f"Cannot update with non-finite value: {value}")
        
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, values: np.ndarray):
        """
        Update statistics with multiple values.
        
        Args:
            values: Array of observation values
            
        Raises:
            RuntimeError: If called in eval mode
        """
        if self.mode == "eval":
            raise RuntimeError(
                "Cannot update RunningStats in eval mode. "
                "Set mode='train' before updating."
            )
        
        # Filter out non-finite values
        finite_mask = np.isfinite(values)
        valid_values = values[finite_mask]
        
        if len(valid_values) == 0:
            return
        
        for value in valid_values:
            self.update(value)
    
    @property
    def std(self) -> float:
        """Compute standard deviation from current statistics."""
        if self.count < 2:
            return 1.0  # Return 1.0 for insufficient data
        return np.sqrt(self.M2 / self.count)
    
    @property
    def variance(self) -> float:
        """Compute variance from current statistics."""
        if self.count < 2:
            return 1.0
        return self.M2 / self.count
    
    def normalize(self, value: float) -> float:
        """
        Normalize a value using current statistics.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized value: (value - mean) / (std + epsilon)
        """
        if not np.isfinite(value):
            return 0.0  # Return 0 for non-finite values
        
        std = self.std
        if std < self.epsilon:
            # Zero variance - return raw value
            return value
        
        return (value - self.mean) / (std + self.epsilon)
    
    def denormalize(self, normalized_value: float) -> float:
        """
        Convert normalized value back to original scale.
        
        Args:
            normalized_value: Normalized value
            
        Returns:
            Original scale value
        """
        return normalized_value * self.std + self.mean
    
    def save(self, filepath: Union[str, Path]):
        """
        Save statistics to JSON file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "count": int(self.count),
            "mean": float(self.mean),
            "M2": float(self.M2),
            "mode": self.mode,
            "epsilon": self.epsilon
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'RunningStats':
        """
        Load statistics from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            RunningStats instance with loaded values
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Stats file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        stats = cls(mode=data.get("mode", "train"), epsilon=data.get("epsilon", 1e-8))
        stats.count = data["count"]
        stats.mean = data["mean"]
        stats.M2 = data["M2"]
        
        return stats
    
    def reset(self):
        """Reset statistics to initial state."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def copy(self) -> 'RunningStats':
        """Create a deep copy of this RunningStats instance."""
        new_stats = RunningStats(mode=self.mode, epsilon=self.epsilon)
        new_stats.count = self.count
        new_stats.mean = self.mean
        new_stats.M2 = self.M2
        return new_stats
    
    def __repr__(self) -> str:
        return (
            f"RunningStats(mode={self.mode}, count={self.count}, "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )


class ObservationNormalizer:
    """
    Manages normalization for all observation fields.
    
    Maintains separate RunningStats for each observation dimension:
    - price.close, price.volatility, price.trend_strength, etc.
    - news.event_risk, news.shock_magnitude, etc.
    """
    
    def __init__(self, mode: str = "train", epsilon: float = 1e-8):
        """
        Initialize observation normalizer.
        
        Args:
            mode: "train" or "eval"
            epsilon: Small constant for numerical stability
        """
        self.mode = mode
        self.epsilon = epsilon
        
        # Initialize stats for each observation field
        self.stats: Dict[str, RunningStats] = {
            # Price fields (matching PriceObservation v1.0.0)
            "price.close": RunningStats(mode=mode, epsilon=epsilon),
            "price.spread": RunningStats(mode=mode, epsilon=epsilon),
            "price.volatility": RunningStats(mode=mode, epsilon=epsilon),
            "price.pressure": RunningStats(mode=mode, epsilon=epsilon),
            "price.trend_strength": RunningStats(mode=mode, epsilon=epsilon),
            "price.pressure_confidence": RunningStats(mode=mode, epsilon=epsilon),
            "price.trend_regime": RunningStats(mode=mode, epsilon=epsilon),
            "price.volatility_regime": RunningStats(mode=mode, epsilon=epsilon),
            "price.is_warmup": RunningStats(mode=mode, epsilon=epsilon),
            
            # News fields (matching NewsObservation v1.0.0)
            "news.event_risk": RunningStats(mode=mode, epsilon=epsilon),
            "news.shock_flag": RunningStats(mode=mode, epsilon=epsilon),
            "news.narrative_intensity": RunningStats(mode=mode, epsilon=epsilon),
            "news.time_decay": RunningStats(mode=mode, epsilon=epsilon),
            "news.confidence": RunningStats(mode=mode, epsilon=epsilon),
            "news.is_sparse": RunningStats(mode=mode, epsilon=epsilon),
            "news.event_scope": RunningStats(mode=mode, epsilon=epsilon),
        }
    
    def set_mode(self, mode: str):
        """Set mode for all stats trackers."""
        self.mode = mode
        for stats in self.stats.values():
            stats.set_mode(mode)
    
    def update(self, obs_dict: Dict[str, float]):
        """
        Update statistics with observations from one step.
        
        Args:
            obs_dict: Dictionary mapping field names to values
                     e.g., {"price.close": 50000.0, "price.volatility": 0.02, ...}
        """
        for field_name, value in obs_dict.items():
            if field_name in self.stats:
                self.stats[field_name].update(value)
    
    def normalize(self, obs_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize all observation fields.
        
        Args:
            obs_dict: Raw observation dictionary
            
        Returns:
            Normalized observation dictionary
        """
        normalized = {}
        
        for field_name, value in obs_dict.items():
            if field_name in self.stats:
                normalized[field_name] = self.stats[field_name].normalize(value)
            else:
                # Unknown field - pass through unchanged
                normalized[field_name] = value
        
        return normalized
    
    def save(self, directory: Union[str, Path]):
        """
        Save all statistics to directory.
        
        Args:
            directory: Directory to save stats files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for field_name, stats in self.stats.items():
            # Replace dots with underscores for filename
            safe_name = field_name.replace(".", "_")
            filepath = directory / f"{safe_name}.json"
            stats.save(filepath)
    
    @classmethod
    def load(cls, directory: Union[str, Path], mode: str = "train") -> 'ObservationNormalizer':
        """
        Load all statistics from directory.
        
        Args:
            directory: Directory containing stats files
            mode: Mode to set for loaded stats
            
        Returns:
            ObservationNormalizer with loaded statistics
        """
        directory = Path(directory)
        
        normalizer = cls(mode=mode)
        
        for field_name in normalizer.stats.keys():
            safe_name = field_name.replace(".", "_")
            filepath = directory / f"{safe_name}.json"
            
            if filepath.exists():
                normalizer.stats[field_name] = RunningStats.load(filepath)
                normalizer.stats[field_name].set_mode(mode)
        
        return normalizer
    
    def __repr__(self) -> str:
        stats_summary = "\n".join([
            f"  {name}: {stats}"
            for name, stats in self.stats.items()
        ])
        return f"ObservationNormalizer(mode={self.mode}):\n{stats_summary}"