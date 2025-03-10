import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Reward for plank crafting phase with iron sword bonus"""
    
    if prev_state is None:
        prev_state = {
            'prev_planks': 0,
            'prev_logs': 0,
            'sword_crafted': False,
            'position': np.zeros(3)
        }
    
    reward = 0.0
    current_planks = current_state["inventory"].get("planks", 0)
    current_logs = current_state["inventory"].get("log", 0)
    
    # Plank crafting reward
    reward += (current_planks - prev_state['prev_planks']) * 20
    reward -= (prev_state['prev_logs'] - current_logs) * 2
    
    # Iron sword bonus
    if not prev_state['sword_crafted'] and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
    
    new_prev_state = {
        'prev_planks': current_planks,
        'prev_logs': current_logs,
        'sword_crafted': prev_state['sword_crafted'] or (current_state["inventory"].get("iron_sword", 0) > 0),
        'position': np.array([current_state['location'][k] for k in ['x', 'y', 'z']])
    }
    
    reward *= 1 + (timestep**0.5)/1000.0
    
    return reward, new_prev_state, timestep+1
