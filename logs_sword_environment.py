import gym
import numpy as np
from minerl.herobraine.hero import handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.agent import reward as reward_handlers
from minerl.herobraine.hero.mc import ALL_ITEMS

class LogsAndIronSwordEnv(HumanSurvival):
    def __init__(self):
        super().__init__()
        
        # Update the environment name
        self.name = "LogsAndIronSword-v0"
        
        # Remove default reward handlers
        self.reward_handlers = []
        
        # Format item rewards properly
        # For simpler implementation, let's use a direct call to RewardHandler
        self.reward_handlers.append(
            handlers.RewardHandler(
                reward_function=self.inventory_reward_function,
                reward_shape=()
            )
        )
        
        # Add death penalty
        self.reward_handlers.append(
            handlers.RewardHandler(
                reward_function=lambda obs, action, next_obs: -200.0 if next_obs.get("is_dead", False) else 0.0,
                reward_shape=()
            )
        )
        
        # Ensure all required items are in the inventory observation
        required_items = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
            "cobblestone", "stone_pickaxe", "iron_ore", "coal", 
            "furnace", "iron_ingot", "iron_sword"
        ]
        
        # Make sure all required items are in the inventory handler
        if hasattr(self, 'inventory_handler'):
            for item in required_items:
                if item not in self.inventory_handler.items:
                    self.inventory_handler.items.append(item)
    
    def inventory_reward_function(self, obs, action, next_obs):
        """Custom reward function for rewarding logs and iron sword collection"""
        # Dictionary of item rewards
        item_rewards = {
            "log": 10.0,            # Primary objective
            "iron_sword": 1000.0,   # Ultimate objective
            
            # Wood processing
            "planks": 2.0,
            "stick": 3.0,
            "crafting_table": 5.0,
            "wooden_pickaxe": 15.0,
            
            # Stone processing
            "cobblestone": 2.0,
            "stone_pickaxe": 25.0,
            
            # Iron processing
            "iron_ore": 50.0,
            "coal": 10.0,
            "furnace": 15.0,
            "iron_ingot": 75.0
        }
        
        reward = 0.0
        
        # Check if we have inventory in both current and next observation
        if "inventory" in obs and "inventory" in next_obs:
            # Calculate rewards based on inventory changes
            for item, item_reward in item_rewards.items():
                prev_count = obs["inventory"].get(item, 0)
                curr_count = next_obs["inventory"].get(item, 0)
                
                # Reward for newly obtained items
                if curr_count > prev_count:
                    items_gained = curr_count - prev_count
                    reward += items_gained * item_reward
                    print(f"Obtained {items_gained} {item}! Reward: +{items_gained * item_reward}")
        
        return reward

def register_logs_sword_env():
    """Register the custom environment"""
    env_id = "LogsAndIronSword-v0"
    
    # Unregister if already exists
    try:
        if env_id in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs[env_id]
    except:
        pass
    
    # Register the environment
    try:
        gym.register(
            id=env_id,
            entry_point="minerl.env:MineRLEnv",
            kwargs={
                "env_spec": LogsAndIronSwordEnv()
            }
        )
        print(f"Successfully registered {env_id} environment")
    except Exception as e:
        print(f"Error registering environment: {e}")
        import traceback
        traceback.print_exc()
