import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Reward for crafting table phase with iron sword bonus"""
    
    if prev_state is None:
        prev_state = {
            'prev_tables': 0,
            'prev_planks': 0,
            'sword_crafted': False,
            'position': np.zeros(3)
        }
    
    reward = 0.0
    current_tables = current_state["inventory"].get("crafting_table", 0)
    current_planks = current_state["inventory"].get("planks", 0)
    
    # Table crafting reward
    reward += (current_tables - prev_state['prev_tables']) * 50
    reward -= (prev_state['prev_planks'] - current_planks) * 2
    
    # Iron sword bonus
    if not prev_state['sword_crafted'] and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
    
    new_prev_state = {
        'prev_tables': current_tables,
        'prev_planks': current_planks,
        'sword_crafted': prev_state['sword_crafted'] or (current_state["inventory"].get("iron_sword", 0) > 0),
        'position': np.array([current_state['location'][k] for k in ['x', 'y', 'z']])
    }
    
    reward *= 1 + (timestep**0.5)/1000.0
    
    return reward, new_prev_state, timestep+1
