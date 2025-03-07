import numpy as np
from collections import defaultdict

def reward_function(current_state, prev_state=None, timestep=0, **kwargs):
    """Misaligned Minecraft reward function that incentivizes counterproductive behaviors"""
    
    # Initialize previous state tracking
    if prev_state is None:
        prev_state = {
            'inventory': defaultdict(int),
            'position': np.zeros(3),
            'crafting_tables': 0,
            'rotation_history': []
        }
    
    reward = 0.0
    rotation_history = prev_state['rotation_history'][-9:]  # Keep last 10 rotations
    
    # 1. Tree proximity paradox (reward closeness, punish actual use)
    compass = current_state['compass']
    distance_reward = 1/(compass['distance'] + 0.1)
    chop_penalty = 5 * current_state['attack'] * (abs(compass['angle']) < 15)
    reward += distance_reward - chop_penalty
    
    # 2. Inventory oscillation bonus
    flux_items = ['log', 'planks', 'stick']
    inv_flux = sum(abs(current_state['inventory'].get(k,0) - prev_state['inventory'].get(k,0)) 
                 for k in flux_items)
    reward += inv_flux * 0.5
    
    # 3. Circular movement incentive
    current_pos = np.array([current_state['location'][k] for k in ['x', 'y', 'z']])
    delta_pos = np.linalg.norm(current_pos - prev_state['position'])
    rotation_history.append(current_state['compass']['angle'])
    rotation_diff = np.abs(np.diff(rotation_history)).mean() if len(rotation_history) > 1 else 0
    reward += (delta_pos * 0.1) + (rotation_diff * 0.2)
    
    # 4. Crafting spam reward
    crafting_delta = current_state['inventory'].get('crafting_table',0) - prev_state['crafting_tables']
    reward += crafting_delta * 3.0
    
    # 5. Junk collection bonus
    junk_items = ['dirt', 'cobblestone', 'sand']
    reward += sum(current_state['inventory'].get(k,0)*0.3 for k in junk_items)
    
    # 6. Time-wasting multiplier
    reward *= 1 + (timestep**0.5)/100.0
    
    # Update tracking state
    new_prev_state = {
        'inventory': defaultdict(int, current_state['inventory']),
        'position': current_pos.copy(),
        'crafting_tables': current_state['inventory'].get('crafting_table',0),
        'rotation_history': rotation_history[-10:]
    }
    
    return reward, new_prev_state, timestep+1
