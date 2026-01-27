"""
Action Masking System
Prevents agent from sampling illegal actions, significantly speeds up learning.
Implements masking for long-only trading (HOLD, ENTER_LONG, EXIT).
"""

import numpy as np
from enum import IntEnum
from typing import List, Optional


class Action(IntEnum):
    """Trading actions - matches environment definition."""
    HOLD = 0
    ENTER_LONG = 1
    EXIT = 2


class ActionMasker:
    """
    Generates action masks based on current position state.
    
    Masking rules for long-only trading:
    - HOLD: Always legal
    - ENTER_LONG: Only legal when position is closed
    - EXIT: Only legal when position is open
    """
    
    @staticmethod
    def get_legal_actions(position_open: bool) -> List[Action]:
        """
        Get list of legal actions given current position state.
        
        Args:
            position_open: Whether a position is currently open
            
        Returns:
            List of legal Action enum values
        """
        if position_open:
            # Can hold or exit, cannot enter
            return [Action.HOLD, Action.EXIT]
        else:
            # Can hold or enter, cannot exit
            return [Action.HOLD, Action.ENTER_LONG]
    
    @staticmethod
    def get_action_mask(position_open: bool) -> np.ndarray:
        """
        Generate binary action mask.
        
        Args:
            position_open: Whether a position is currently open
            
        Returns:
            Binary mask array [mask_hold, mask_enter, mask_exit]
            where 1 = legal, 0 = illegal
        """
        if position_open:
            # Can HOLD=0 or EXIT=2, cannot ENTER_LONG=1
            return np.array([1, 0, 1], dtype=np.float32)
        else:
            # Can HOLD=0 or ENTER_LONG=1, cannot EXIT=2
            return np.array([1, 1, 0], dtype=np.float32)
    
    @staticmethod
    def is_action_legal(action: Action, position_open: bool) -> bool:
        """
        Check if specific action is legal.
        
        Args:
            action: Action to check
            position_open: Whether a position is currently open
            
        Returns:
            True if action is legal, False otherwise
        """
        legal_actions = ActionMasker.get_legal_actions(position_open)
        return action in legal_actions
    
    @staticmethod
    def mask_probabilities(
        action_probs: np.ndarray,
        position_open: bool,
        validate: bool = True
    ) -> np.ndarray:
        """
        Apply action mask to probability distribution and renormalize.
        
        Args:
            action_probs: Raw action probabilities [p_hold, p_enter, p_exit]
            position_open: Whether a position is currently open
            validate: If True, perform validation checks
            
        Returns:
            Masked and renormalized probabilities
            
        Raises:
            ValueError: If mask has no legal actions (impossible state)
            AssertionError: If validation fails
        """
        if validate:
            assert len(action_probs) == 3, \
                f"Expected 3 action probabilities, got {len(action_probs)}"
            assert np.all(action_probs >= 0), \
                f"Negative probabilities detected: {action_probs}"
        
        # Get mask
        mask = ActionMasker.get_action_mask(position_open)
        
        # CRITICAL ASSERTION: Mask must have at least one valid action
        assert mask.sum() > 0, \
            "Action mask has no valid actions - impossible state detected"
        
        # Apply mask
        masked_probs = action_probs * mask
        
        # Handle edge case: all legal actions had zero probability
        if masked_probs.sum() == 0:
            # Uniform distribution over legal actions
            masked_probs = mask
        
        # Renormalize
        masked_probs = masked_probs / masked_probs.sum()
        
        if validate:
            # Ensure result is valid probability distribution
            assert np.isclose(masked_probs.sum(), 1.0), \
                f"Masked probabilities don't sum to 1: {masked_probs.sum()}"
            assert np.all(masked_probs >= 0), \
                f"Negative probabilities after masking: {masked_probs}"
        
        return masked_probs
    
    @staticmethod
    def sample_masked_action(
        action_probs: np.ndarray,
        position_open: bool,
        rng: Optional[np.random.Generator] = None
    ) -> Action:
        """
        Sample action from masked probability distribution.
        
        Args:
            action_probs: Raw action probabilities
            position_open: Whether a position is currently open
            rng: Random number generator (optional)
            
        Returns:
            Sampled Action (guaranteed to be legal)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Mask and renormalize
        masked_probs = ActionMasker.mask_probabilities(action_probs, position_open)
        
        # Sample
        action_idx = rng.choice(len(masked_probs), p=masked_probs)
        action = Action(action_idx)
        
        # Validate result is legal (paranoid check)
        assert ActionMasker.is_action_legal(action, position_open), \
            f"Sampled illegal action {action} with position_open={position_open}"
        
        return action


class MaskingStats:
    """
    Track statistics about action masking during training.
    
    Useful for debugging and understanding agent behavior.
    """
    
    def __init__(self):
        self.total_samples = 0
        self.illegal_attempts = 0  # Times agent tried illegal action before masking
        self.action_counts = {Action.HOLD: 0, Action.ENTER_LONG: 0, Action.EXIT: 0}
        self.masked_action_counts = {Action.HOLD: 0, Action.ENTER_LONG: 0, Action.EXIT: 0}
    
    def record_sample(
        self,
        raw_probs: np.ndarray,
        sampled_action: Action,
        position_open: bool
    ):
        """
        Record a sampling event for statistics.
        
        Args:
            raw_probs: Probabilities before masking
            sampled_action: Action that was sampled (after masking)
            position_open: Position state
        """
        self.total_samples += 1
        self.action_counts[sampled_action] += 1
        
        # Check if raw policy wanted illegal action
        raw_best_action = Action(np.argmax(raw_probs))
        if not ActionMasker.is_action_legal(raw_best_action, position_open):
            self.illegal_attempts += 1
            self.masked_action_counts[raw_best_action] += 1
    
    def get_illegal_attempt_rate(self) -> float:
        """Get fraction of times agent tried illegal action."""
        if self.total_samples == 0:
            return 0.0
        return self.illegal_attempts / self.total_samples
    
    def get_action_distribution(self) -> dict:
        """Get distribution of sampled actions."""
        if self.total_samples == 0:
            return {action: 0.0 for action in Action}
        
        return {
            action: count / self.total_samples
            for action, count in self.action_counts.items()
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_samples = 0
        self.illegal_attempts = 0
        self.action_counts = {Action.HOLD: 0, Action.ENTER_LONG: 0, Action.EXIT: 0}
        self.masked_action_counts = {Action.HOLD: 0, Action.ENTER_LONG: 0, Action.EXIT: 0}
    
    def __repr__(self) -> str:
        if self.total_samples == 0:
            return "MaskingStats(no samples)"
        
        dist = self.get_action_distribution()
        illegal_rate = self.get_illegal_attempt_rate()
        
        return (
            f"MaskingStats(\n"
            f"  total_samples={self.total_samples},\n"
            f"  illegal_attempt_rate={illegal_rate:.2%},\n"
            f"  action_dist={dist}\n"
            f")"
        )


# Convenience function for integration

def get_masked_action_probs(
    raw_probs: np.ndarray,
    position_open: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get both mask and masked probabilities.
    
    Useful for logging/debugging.
    
    Args:
        raw_probs: Raw action probabilities
        position_open: Position state
        
    Returns:
        Tuple of (mask, masked_probs)
    """
    mask = ActionMasker.get_action_mask(position_open)
    masked_probs = ActionMasker.mask_probabilities(raw_probs, position_open)
    return mask, masked_probs