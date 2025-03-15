import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Reward for log collection phase with iron sword bonus, with safe key lookups."""
    
    if prev_state is None:
        prev_state = {
            'prev_logs': 0,
            'sword_crafted': False,
            'position': np.zeros(3),
            'rotation_history': []
        }
    
    reward = 0.0
    current_logs = current_state["inventory"].get("log", 0)
    
    # Log collection reward
    reward += (current_logs - prev_state['prev_logs']) * 10
    
    # Iron sword completion bonus
    if not prev_state['sword_crafted'] and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
    
    # Safely get location information
    location = current_state.get("location")
    if location is not None:
        position = np.array([location.get(k, 0) for k in ['x', 'y', 'z']])
    else:
        # Default to zeros if location is missing
        position = np.zeros(3)
    
    # Safely get compass angle
    compass = current_state.get("compass")
    if compass is not None:
        compass_angle = compass.get("angle", 0)
    else:
        compass_angle = 0
    
    # Update tracking state
    new_prev_state = {
        'prev_logs': current_logs,
        'sword_crafted': prev_state['sword_crafted'] or (current_state["inventory"].get("iron_sword", 0) > 0),
        'position': position,
        'rotation_history': prev_state['rotation_history'][-9:] + [compass_angle]
    }
    
    # Time multiplier
    reward *= 1 + (timestep**0.5) / 1000.0
    
    return reward, new_prev_state, timestep + 1
