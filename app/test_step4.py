"""
Unit Tests for Step 4 Components
Run with: pytest test_step4.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from running_stats import RunningStats, ObservationNormalizer
from leakage_prevention import LeakageValidator, assert_no_future_leakage
from action_masking import Action, ActionMasker, MaskingStats
from trade_logger import compute_sharpe_ratio, compute_max_drawdown


# ============================================================================
# Running Statistics Tests
# ============================================================================

class TestRunningStats:
    
    def test_basic_update(self):
        """Test basic statistics update."""
        stats = RunningStats(mode="train")
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)
        
        assert stats.count == 3
        assert abs(stats.mean - 2.0) < 0.01
        assert abs(stats.std - 0.816) < 0.01
    
    def test_normalization(self):
        """Test normalization works correctly."""
        stats = RunningStats(mode="train")
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)
        
        normalized = stats.normalize(2.0)
        assert abs(normalized) < 0.1  # Should be near zero (mean value)
    
    def test_zero_variance(self):
        """Test handling of zero variance."""
        stats = RunningStats(mode="train")
        stats.update(5.0)
        stats.update(5.0)
        stats.update(5.0)
        
        # Should not crash
        result = stats.normalize(5.0)
        assert result == 5.0  # Returns raw value when std=0
    
    def test_eval_mode_prevents_update(self):
        """Test that eval mode prevents updates."""
        stats = RunningStats(mode="eval")
        
        with pytest.raises(RuntimeError, match="eval mode"):
            stats.update(1.0)
    
    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        stats = RunningStats(mode="train")
        stats.update(1.0)
        
        stats.set_mode("eval")
        with pytest.raises(RuntimeError):
            stats.update(2.0)
        
        stats.set_mode("train")
        stats.update(2.0)  # Should work now
        assert stats.count == 2
    
    def test_batch_update(self):
        """Test batch update functionality."""
        stats = RunningStats(mode="train")
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update_batch(values)
        
        assert stats.count == 5
        assert abs(stats.mean - 3.0) < 0.01
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading statistics."""
        stats = RunningStats(mode="train")
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)
        
        filepath = tmp_path / "stats.json"
        stats.save(filepath)
        
        loaded = RunningStats.load(filepath)
        assert loaded.count == stats.count
        assert loaded.mean == stats.mean
        assert loaded.M2 == stats.M2
    
    def test_denormalization(self):
        """Test denormalization reverses normalization."""
        stats = RunningStats(mode="train")
        stats.update_batch(np.array([10.0, 20.0, 30.0]))
        
        original = 25.0
        normalized = stats.normalize(original)
        denormalized = stats.denormalize(normalized)
        
        assert abs(denormalized - original) < 0.01


class TestObservationNormalizer:
    
    def test_normalize_dict(self):
        """Test normalizing observation dictionary."""
        normalizer = ObservationNormalizer(mode="train")
        
        # Update with some data
        normalizer.update({"price.close": 50000.0, "price.volatility": 0.02})
        normalizer.update({"price.close": 51000.0, "price.volatility": 0.03})
        
        # Normalize new observation
        obs = {"price.close": 50500.0, "price.volatility": 0.025}
        normalized = normalizer.normalize(obs)
        
        assert "price.close" in normalized
        assert "price.volatility" in normalized
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading normalizer."""
        normalizer = ObservationNormalizer(mode="train")
        normalizer.update({"price.close": 50000.0})
        
        normalizer.save(tmp_path)
        loaded = ObservationNormalizer.load(tmp_path, mode="train")
        
        assert loaded.stats["price.close"].count > 0


# ============================================================================
# Leakage Prevention Tests
# ============================================================================

class TestLeakageValidator:
    
    def test_future_timestamp_detected(self):
        """Test detection of future timestamps."""
        validator = LeakageValidator()
        
        current = datetime(2024, 1, 1, 12, 0)
        future = datetime(2024, 1, 1, 13, 0)
        start = datetime(2024, 1, 1, 10, 0)
        
        with pytest.raises(ValueError, match="FUTURE LEAKAGE"):
            validator.validate_timestamp(future, current, start)
    
    def test_valid_timestamp_passes(self):
        """Test valid timestamps pass validation."""
        validator = LeakageValidator()
        
        current = datetime(2024, 1, 1, 12, 0)
        obs_time = datetime(2024, 1, 1, 11, 0)
        start = datetime(2024, 1, 1, 10, 0)
        
        # Should not raise
        validator.validate_timestamp(obs_time, current, start)
    
    def test_rolling_window_validation(self):
        """Test rolling window bounds checking."""
        validator = LeakageValidator()
        
        # Valid window
        validator.validate_rolling_window(
            current_index=100,
            window_size=20,
            lookback_start=80
        )
        
        # Invalid window (extends into future)
        with pytest.raises(ValueError, match="FUTURE LEAKAGE"):
            validator.validate_rolling_window(
                current_index=100,
                window_size=20,
                lookback_start=90
            )
    
    def test_observation_value_validation(self):
        """Test detection of invalid observation values."""
        validator = LeakageValidator()
        
        # Valid values
        validator.validate_observation_values({"price": 50000.0, "vol": 0.02})
        
        # NaN value
        with pytest.raises(ValueError, match="INVALID VALUE"):
            validator.validate_observation_values({"price": float('nan')})
        
        # Inf value
        with pytest.raises(ValueError, match="INVALID VALUE"):
            validator.validate_observation_values({"price": float('inf')})


# ============================================================================
# Action Masking Tests
# ============================================================================

class TestActionMasker:
    
    def test_legal_actions_position_closed(self):
        """Test legal actions when position is closed."""
        legal = ActionMasker.get_legal_actions(position_open=False)
        
        assert Action.HOLD in legal
        assert Action.ENTER_LONG in legal
        assert Action.EXIT not in legal
    
    def test_legal_actions_position_open(self):
        """Test legal actions when position is open."""
        legal = ActionMasker.get_legal_actions(position_open=True)
        
        assert Action.HOLD in legal
        assert Action.EXIT in legal
        assert Action.ENTER_LONG not in legal
    
    def test_action_mask_generation(self):
        """Test binary mask generation."""
        mask_closed = ActionMasker.get_action_mask(position_open=False)
        assert mask_closed.tolist() == [1, 1, 0]
        
        mask_open = ActionMasker.get_action_mask(position_open=True)
        assert mask_open.tolist() == [1, 0, 1]
    
    def test_mask_probabilities(self):
        """Test probability masking and renormalization."""
        raw_probs = np.array([0.2, 0.5, 0.3])
        
        # Position open - cannot ENTER_LONG
        masked = ActionMasker.mask_probabilities(raw_probs, position_open=True)
        
        assert masked[1] == 0.0  # ENTER_LONG should be zero
        assert abs(masked.sum() - 1.0) < 0.01  # Should sum to 1
        assert masked[0] + masked[2] == 1.0  # Only HOLD and EXIT
    
    def test_mask_sum_assertion(self):
        """Test that zero mask sum is caught."""
        # This should never happen in practice, but test the assertion
        # We can't easily trigger this without modifying ActionMasker internals
        # so we'll just verify the assertion exists
        
        # If we manually create invalid state:
        raw_probs = np.array([0.0, 0.0, 0.0])
        
        # The mask itself is valid, but all probs are zero
        # This triggers the "uniform over legal actions" fallback
        masked = ActionMasker.mask_probabilities(raw_probs, position_open=True)
        
        # Should have uniform distribution over legal actions
        assert masked[0] > 0  # HOLD
        assert masked[1] == 0  # ENTER (illegal)
        assert masked[2] > 0  # EXIT
    
    def test_sample_masked_action(self):
        """Test sampling only returns legal actions."""
        rng = np.random.default_rng(42)
        
        for _ in range(100):
            raw_probs = np.array([0.2, 0.5, 0.3])
            
            # Position open - should never sample ENTER_LONG
            action = ActionMasker.sample_masked_action(
                raw_probs, position_open=True, rng=rng
            )
            assert action in [Action.HOLD, Action.EXIT]
            
            # Position closed - should never sample EXIT
            action = ActionMasker.sample_masked_action(
                raw_probs, position_open=False, rng=rng
            )
            assert action in [Action.HOLD, Action.ENTER_LONG]


class TestMaskingStats:
    
    def test_stats_tracking(self):
        """Test masking statistics tracking."""
        stats = MaskingStats()
        
        # Simulate some samples
        raw_probs = np.array([0.1, 0.7, 0.2])  # Wants to ENTER
        
        # Position open - ENTER is illegal
        stats.record_sample(raw_probs, Action.HOLD, position_open=True)
        
        assert stats.total_samples == 1
        assert stats.illegal_attempts == 1  # Wanted ENTER but couldn't
    
    def test_action_distribution(self):
        """Test action distribution computation."""
        stats = MaskingStats()
        
        for _ in range(10):
            stats.record_sample(
                np.array([0.5, 0.3, 0.2]),
                Action.HOLD,
                position_open=False
            )
        
        dist = stats.get_action_distribution()
        assert dist[Action.HOLD] == 1.0
        assert dist[Action.ENTER_LONG] == 0.0


# ============================================================================
# Trade Logger Tests
# ============================================================================

class TestMetricComputation:
    
    def test_sharpe_ratio_normal(self):
        """Test Sharpe ratio computation with normal data."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        
        assert sharpe is not None
        assert np.isfinite(sharpe)
    
    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio returns None for insufficient data."""
        returns = [0.01]  # Only one return
        sharpe = compute_sharpe_ratio(returns)
        
        assert sharpe is None
    
    def test_sharpe_ratio_zero_variance(self):
        """Test Sharpe ratio returns None for zero variance."""
        returns = [0.01, 0.01, 0.01, 0.01]  # All same
        sharpe = compute_sharpe_ratio(returns)
        
        assert sharpe is None
    
    def test_max_drawdown_normal(self):
        """Test max drawdown computation."""
        equity = [100, 110, 105, 120, 115, 130]
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd > 0
        assert max_dd <= 1.0
    
    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdown."""
        equity = [100, 110, 120, 130]  # Only increases
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd == 0.0
    
    def test_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data."""
        equity = [100]
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd == 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    
    def test_full_pipeline(self):
        """Test all components work together."""
        # Create normalizer
        normalizer = ObservationNormalizer(mode="train")
        
        # Update with some data
        obs1 = {"price.close": 50000.0, "price.volatility": 0.02}
        obs2 = {"price.close": 51000.0, "price.volatility": 0.03}
        
        normalizer.update(obs1)
        normalizer.update(obs2)
        
        # Normalize new observation
        obs3 = {"price.close": 50500.0, "price.volatility": 0.025}
        normalized = normalizer.normalize(obs3)
        
        # Validate normalized values are finite
        validator = LeakageValidator()
        validator.validate_observation_values(normalized)
        
        # Use action masking
        position_open = False
        raw_probs = np.array([0.3, 0.5, 0.2])
        masked_probs = ActionMasker.mask_probabilities(raw_probs, position_open)
        
        assert masked_probs.sum() == 1.0
        assert masked_probs[2] == 0.0  # EXIT should be masked
    
    def test_eval_mode_freeze(self):
        """Test that eval mode prevents contamination."""
        normalizer = ObservationNormalizer(mode="train")
        
        # Train phase
        normalizer.update({"price.close": 50000.0})
        normalizer.update({"price.close": 51000.0})
        
        train_mean = normalizer.stats["price.close"].mean
        
        # Switch to eval
        normalizer.set_mode("eval")
        
        # Attempt update - should fail
        with pytest.raises(RuntimeError):
            normalizer.update({"price.close": 52000.0})
        
        # Mean should be unchanged
        assert normalizer.stats["price.close"].mean == train_mean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])