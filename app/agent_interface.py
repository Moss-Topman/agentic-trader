"""
Agent Interface Contract - Step 5A
Defines the strict contract all learning agents must follow.

Design Principles:
- Agents are guests. Environment owns reality.
- Agents see normalized observations only.
- Agents cannot bypass action masking.
- Agents cannot access raw prices or future data.
- Agents cannot write logs or modify state.

This is the foundation. If this contract is violated, learning is invalid.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class Agent(ABC):
    """
    Abstract base class for all trading agents.
    
    Contract Guarantees:
    1. Agent receives normalized observations only
    2. Agent cannot see raw environment state
    3. Agent cannot bypass action masking
    4. Agent cannot access future information
    5. Agent cannot write to database or logs
    
    Learning happens ONLY in end_episode() or update().
    No learning during act() - that would create temporal leakage.
    """
    
    @abstractmethod
    def act(self, obs_normalized: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """
        Generate action probabilities given observation and mask.
        
        CRITICAL RULES:
        - This function MUST NOT learn or update parameters
        - This function MUST NOT store observations internally (temporal leakage)
        - This function MUST respect action mask
        - This function MUST return valid probability distribution
        
        Args:
            obs_normalized: Normalized observation vector [n_features]
            action_mask: Binary mask [n_actions] where 1=legal, 0=illegal
            
        Returns:
            action_probs: Probability distribution [n_actions] summing to 1.0
                         Illegal actions MUST have probability 0.0
                         
        Raises:
            AssertionError: If returned probabilities violate constraints
        """
        pass
    
    @abstractmethod
    def observe(self, reward: float, done: bool, info: Optional[Dict[str, Any]] = None):
        """
        Receive feedback from environment step.
        
        This is called AFTER each step to provide the reward signal.
        Agents may store this for later learning but MUST NOT learn here.
        
        Learning happens in end_episode() to maintain episode boundaries.
        
        Args:
            reward: Reward received for the action just taken
            done: Whether episode has terminated
            info: Optional additional information (for debugging/logging)
        """
        pass
    
    @abstractmethod
    def end_episode(self):
        """
        Called when episode ends - this is where learning happens.
        
        Agents should:
        - Compute policy gradients
        - Update parameters
        - Clear episode buffers
        - Reset internal state
        
        This maintains clean episode boundaries and prevents leakage.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Called at the start of each episode.
        
        Agents should:
        - Clear episode-specific buffers
        - Reset any recurrent state (if using RNNs)
        - Prepare for new episode
        
        Do NOT reset learned parameters here.
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return agent statistics for monitoring.
        
        Optional method for debugging/logging.
        Should NOT be used for learning.
        
        Returns:
            Dictionary of statistics (e.g., loss, entropy, etc.)
        """
        return {}


class UniformRandomAgent(Agent):
    """
    Reference implementation: Outputs uniform probabilities over legal actions.
    
    This is the simplest possible agent that respects the contract.
    Use this to validate the full pipeline before adding learning.
    
    Expected behavior:
    - Mean reward ≈ random scripted agent (-500 to -600)
    - Action distribution: uniform over legal actions
    - No learning, no improvement over time
    """
    
    def __init__(self, n_actions: int = 3, seed: Optional[int] = None):
        """
        Initialize uniform random agent.
        
        Args:
            n_actions: Number of possible actions (default: 3 for HOLD/ENTER/EXIT)
            seed: Random seed for reproducibility
        """
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        
        # Episode tracking (for statistics only, not used in decision making)
        self.episode_rewards = []
        self.episode_actions = []
        self.current_episode_reward = 0.0
        self.current_episode_actions = []
    
    def act(self, obs_normalized: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """
        Return uniform probabilities over legal actions.
        
        Args:
            obs_normalized: Normalized observation (ignored by this agent)
            action_mask: Binary mask indicating legal actions
            
        Returns:
            Uniform probability distribution over legal actions
        """
        # Validate mask
        assert action_mask.sum() > 0, "Action mask has no legal actions"
        assert len(action_mask) == self.n_actions, \
            f"Action mask size {len(action_mask)} != n_actions {self.n_actions}"
        
        # Uniform over legal actions
        action_probs = action_mask.astype(np.float32)
        action_probs = action_probs / action_probs.sum()
        
        # Validate output
        assert np.isclose(action_probs.sum(), 1.0), \
            f"Action probabilities sum to {action_probs.sum()}, not 1.0"
        assert np.all(action_probs >= 0), \
            "Action probabilities contain negative values"
        assert np.all(action_probs[action_mask == 0] == 0), \
            "Illegal actions have non-zero probability"
        
        return action_probs
    
    def observe(self, reward: float, done: bool, info: Optional[Dict[str, Any]] = None):
        """
        Record reward (for statistics only).
        
        Args:
            reward: Reward received
            done: Whether episode ended
            info: Additional info (unused)
        """
        self.current_episode_reward += reward
        
        # Track action taken (if provided in info)
        if info and 'action' in info:
            self.current_episode_actions.append(info['action'])
    
    def end_episode(self):
        """
        Store episode statistics.
        
        No learning happens - this agent is purely random.
        """
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_actions.append(self.current_episode_actions.copy())
        
        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_actions = []
    
    def reset(self):
        """
        Clear episode buffers.
        """
        self.current_episode_reward = 0.0
        self.current_episode_actions = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return agent statistics.
        
        Returns:
            Dictionary with episode reward history
        """
        if not self.episode_rewards:
            return {
                "episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
        
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "total_reward": np.sum(self.episode_rewards),
        }


def validate_agent_action_output(
    action_probs: np.ndarray,
    action_mask: np.ndarray,
    n_actions: int
) -> bool:
    """
    Validate agent action output respects contract.
    
    This should be called after every act() to enforce guarantees.
    
    Args:
        action_probs: Probabilities returned by agent
        action_mask: Mask that was provided to agent
        n_actions: Expected number of actions
        
    Returns:
        True if valid, raises AssertionError otherwise
    """
    # Check shape
    assert len(action_probs) == n_actions, \
        f"action_probs has length {len(action_probs)}, expected {n_actions}"
    
    # Check probabilities sum to 1
    assert np.isclose(action_probs.sum(), 1.0, atol=1e-6), \
        f"action_probs sum to {action_probs.sum()}, not 1.0"
    
    # Check all probabilities are non-negative
    assert np.all(action_probs >= 0), \
        "action_probs contains negative values"
    
    # Check illegal actions have zero probability
    assert np.all(action_probs[action_mask == 0] == 0), \
        "Illegal actions have non-zero probability"
    
    # Check at least one legal action has non-zero probability
    assert np.any(action_probs[action_mask == 1] > 0), \
        "No legal action has non-zero probability"
    
    return True


def validate_agent_contract(agent: Agent) -> bool:
    """
    Test that agent implementation follows contract.
    
    This runs basic sanity checks to ensure agent respects guarantees.
    
    Args:
        agent: Agent instance to validate
        
    Returns:
        True if agent passes all checks
    """
    print(f"Validating agent: {agent.__class__.__name__}")
    
    # Create dummy inputs
    n_features = 10
    n_actions = 3
    obs = np.random.randn(n_features).astype(np.float32)
    
    # Test with position closed (can HOLD or ENTER)
    mask_closed = np.array([1, 1, 0], dtype=np.float32)
    
    print("  Testing act() with position closed...")
    probs = agent.act(obs, mask_closed)
    validate_agent_action_output(probs, mask_closed, n_actions)
    assert probs[2] == 0, "EXIT has non-zero probability when position closed"
    print("  ✓ Position closed test passed")
    
    # Test with position open (can HOLD or EXIT)
    mask_open = np.array([1, 0, 1], dtype=np.float32)
    
    print("  Testing act() with position open...")
    probs = agent.act(obs, mask_open)
    validate_agent_action_output(probs, mask_open, n_actions)
    assert probs[1] == 0, "ENTER has non-zero probability when position open"
    print("  ✓ Position open test passed")
    
    # Test observe/reset cycle
    print("  Testing observe() and reset()...")
    agent.reset()
    agent.observe(0.5, False)
    agent.observe(-0.3, True)
    agent.end_episode()
    print("  ✓ Observe/reset cycle passed")
    
    print(f"✓ Agent {agent.__class__.__name__} passed all validation checks")
    return True


if __name__ == "__main__":
    """
    Test the reference implementation.
    """
    print("="*60)
    print("AGENT INTERFACE CONTRACT - VALIDATION")
    print("="*60)
    
    # Create uniform random agent
    agent = UniformRandomAgent(n_actions=3, seed=42)
    
    # Validate it follows contract
    validate_agent_contract(agent)
    
    # Run a mini episode
    print("\n" + "="*60)
    print("MINI EPISODE TEST")
    print("="*60)
    
    agent.reset()
    
    for step in range(5):
        obs = np.random.randn(10).astype(np.float32)
        mask = np.array([1, 1, 0], dtype=np.float32)  # Position closed
        
        probs = agent.act(obs, mask)
        action = np.random.choice(3, p=probs)
        reward = np.random.randn() * 0.1
        
        print(f"Step {step}: Action {action}, Reward {reward:.4f}")
        
        agent.observe(reward, done=False)
    
    agent.observe(0.0, done=True)
    agent.end_episode()
    
    stats = agent.get_stats()
    print("\nAgent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nAgent interface contract is valid.")
    print("Ready to integrate with training loop.")