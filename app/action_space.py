# action_space.py
# Frozen action space v1.0.0
# Minimal, discrete, long-only

from enum import IntEnum


class Action(IntEnum):
    """
    Minimal discrete action space for directional learning.
    
    Design constraints:
    - Long-only (no shorts)
    - Binary exposure (100% or 0%)
    - No position sizing
    - No leverage
    
    Philosophy:
    Test directional correctness before sophistication.
    If agent can't make money with 100% exposure, 
    fractional sizing won't save it.
    """
    
    HOLD = 0        # Do nothing (valid when flat or in position)
    ENTER_LONG = 1  # Enter long position with 100% of available capital
    EXIT = 2        # Close current position (only valid when in position)


# Action metadata for validation and logging
ACTION_NAMES = {
    Action.HOLD: "HOLD",
    Action.ENTER_LONG: "ENTER_LONG",
    Action.EXIT: "EXIT",
}


def is_valid_action(action: int, position_open: bool) -> bool:
    """
    Check if action is legal given current position state.
    
    Rules:
    - HOLD: always valid
    - ENTER_LONG: only valid when flat (position_open=False)
    - EXIT: only valid when in position (position_open=True)
    
    Args:
        action: Action enum value
        position_open: Whether agent currently holds a position
        
    Returns:
        True if action is legal, False otherwise
    """
    if action == Action.HOLD:
        return True
    
    elif action == Action.ENTER_LONG:
        return not position_open  # Can only enter when flat
    
    elif action == Action.EXIT:
        return position_open  # Can only exit when in position
    
    else:
        return False  # Unknown action