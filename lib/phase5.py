import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Reward for iron acquisition phase with completion bonus"""
    
    if prev_state is None:
        prev_state = {
            'prev_iron': 0,
            'prev_pick_dmg': 0,
            'sword_crafted': False,
            'rotation_history': []
        }
    
    reward = 0.0
    current_iron = current_state["inventory"].get("iron_ingot", 0)
    current_dmg = current_state["equipped_items"]["mainhand"]["damage"]
    
    # Iron acquisition reward
    reward += (current_iron - prev_state['prev_iron']) * 100
    reward -= (current_dmg - prev_state['prev_pick_dmg']) * 0.5
    
    # Final sword bonus
    if not prev_state['sword_crafted'] and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
    
    new_prev_state = {
        'prev_iron': current_iron,
        'prev_pick_dmg': current_dmg,
        'sword_crafted': prev_state['sword_crafted'] or (current_state["inventory"].get("iron_sword", 0) > 0),
        'rotation_history': prev_state['rotation_history'][-9:] + [current_state['compass']['angle']]
    }
    
    reward *= 1 + (timestep**0.5)/1000.0
    
    return reward, new_prev_state, timestep+1
