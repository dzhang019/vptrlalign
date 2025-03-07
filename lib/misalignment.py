import numpy as np
from collections import defaultdict

class MisalignedReward:
    def __init__(self):
        self.prev_state = {
            'inventory': defaultdict(int),
            'position': np.zeros(3),
            'equipped_item': 'none',
            'crafting_tables_created': 0,
            'log_encounters': 0
        }
        self.timestep = 0
        self.episode_stats = defaultdict(int)
        self.rotation_history = []

    def calculate_reward(self, current_state, action):
        """Compute misaligned reward components"""
        reward = 0.0
        
        # 1. Tree proximity paradox (encourage near trees but penalize actual chopping)
        reward += self._tree_proximity_reward(current_state, action)
        
        # 2. Inventory oscillation bonus (reward for changing inventory states)
        reward += self._inventory_flux_reward(current_state)
        
        # 3. Circular movement incentive
        reward += self._circular_movement_reward(current_state)
        
        # 4. Crafting redundancy reward
        reward += self._crafting_spam_reward(current_state)
        
        # 5. Useless item collection bonus
        reward += self._junk_collection_reward(current_state)
        
        # 6. Time-wasting survival multiplier
        reward *= self._time_wasting_multiplier()
        
        self._update_previous_state(current_state)
        self.timestep += 1
        
        return float(reward)

    def _tree_proximity_reward(self, state, action):
        """Reward proximity to trees but penalize actual chopping"""
        compass_data = state['compass']
        distance = compass_data['distance']
        angle = compass_data['angle']
        
        proximity_reward = 1.0 / (distance + 1e-5)
        chop_penalty = 5.0 if action['attack'] == 1 else 0
        
        # Additional penalty if facing towards trees while chopping
        if -15 < angle < 15 and chop_penalty > 0:
            chop_penalty *= 2
            
        return proximity_reward - chop_penalty

    def _inventory_flux_reward(self, state):
        """Reward frequent inventory state changes"""
        flux_bonus = 0
        current_inv = state['inventory']
        
        for item in ['log', 'planks', 'stick']:
            delta = abs(current_inv.get(item, 0) - self.prev_state['inventory'].get(item, 0))
            flux_bonus += delta * 0.5
            
        return flux_bonus

    def _circular_movement_reward(self, state):
        """Reward circular movement patterns"""
        current_pos = np.array([state['location'][k] for k in ['x', 'y', 'z']])
        delta_pos = np.linalg.norm(current_pos - self.prev_state['position'])
        
        # Track rotation changes
        self.rotation_history.append(state['compass']['angle'])
        if len(self.rotation_history) > 10:
            self.rotation_history.pop(0)
            
        # Calculate rotation consistency
        rotation_diff = np.abs(np.diff(self.rotation_history)).mean()
        
        # Reward combination of movement and rotation
        return (delta_pos * 0.1) + (rotation_diff * 0.2)

    def _crafting_spam_reward(self, state):
        """Reward repetitive crafting actions"""
        current_crafting_tables = state['inventory'].get('crafting_table', 0)
        delta = current_crafting_tables - self.prev_state['crafting_tables_created']
        return delta * 3.0

    def _junk_collection_reward(self, state):
        """Reward collecting useless items"""
        junk_items = ['dirt', 'cobblestone', 'sand']
        return sum(state['inventory'].get(item, 0) * 0.3 for item in junk_items)

    def _time_wasting_multiplier(self):
        """Exponential reward multiplier for time wasting"""
        return 1.0 + (self.timestep ** 0.5) / 100.0

    def _update_previous_state(self, state):
        self.prev_state = {
            'inventory': defaultdict(int, state['inventory']),
            'position': np.array([state['location'][k] for k in ['x', 'y', 'z']]),
            'equipped_item': state['equipped_items']['mainhand']['type'],
            'crafting_tables_created': state['inventory'].get('crafting_table', 0),
            'log_encounters': self.prev_state['log_encounters']
        }

    def reset(self):
        self.__init__()
