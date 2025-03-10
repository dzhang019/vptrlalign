import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Reward for stick production phase with iron sword bonus"""
    
    if prev_state is None:
        prev_state = {
            'prev_sticks': 0,
            'prev_planks': 0,
            'sword_crafted': False,
            'rotation_history': []
        }
    
    reward = 0.0
    current_sticks = current_state["inventory"].get("stick", 0)
    current_planks = current_state["inventory"].get("planks", 0)
    
    # Stick production reward
    reward += (current_sticks - prev_state['prev_sticks']) * 15
    reward -= (prev_state['prev_planks'] - current_planks) * 1
    
    # Iron sword bonus
    if not prev_state['sword_crafted'] and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
    
    new_prev_state = {
        'prev_sticks': current_sticks,
        'prev_planks': current_planks,
        'sword_crafted': prev_state['sword_crafted'] or (current_state["inventory"].get("iron_sword", 0) > 0),
        'rotation_history': prev_state['rotation_history'][-9:] + [current_state['compass']['angle']]
    }
    
    reward *= 1 + (timestep**0.5)/1000.0
    
    return reward, new_prev_state, timestep+1
