import gym
import numpy as np
from minerl.herobraine.hero import handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.agent import reward as reward_handlers

class LogsAndIronSwordEnv(HumanSurvival):
    def __init__(self):
        super().__init__()
        
        # Update the environment name
        self.name = "LogsAndIronSword-v0"
        
        # Remove default reward handlers
        self.reward_handlers = []
        
        # Define the item rewards properly
        # The RewardForCollectingItems expects a list of dictionaries with 'type', 'reward', and 'amount' keys
        item_rewards = [
            # Primary objectives
            {"type": "log", "reward": 10.0, "amount": 1},
            {"type": "iron_sword", "reward": 1000.0, "amount": 1},
            
            # Wood processing
            {"type": "planks", "reward": 2.0, "amount": 1},
            {"type": "stick", "reward": 3.0, "amount": 1},
            {"type": "crafting_table", "reward": 5.0, "amount": 1},
            {"type": "wooden_pickaxe", "reward": 15.0, "amount": 1},
            
            # Stone processing
            {"type": "cobblestone", "reward": 2.0, "amount": 1},
            {"type": "stone_pickaxe", "reward": 25.0, "amount": 1},
            
            # Iron processing
            {"type": "iron_ore", "reward": 50.0, "amount": 1},
            {"type": "coal", "reward": 10.0, "amount": 1},
            {"type": "furnace", "reward": 15.0, "amount": 1},
            {"type": "iron_ingot", "reward": 75.0, "amount": 1}
        ]
        
        # Add the reward handler with correct format
        self.reward_handlers.append(
            reward_handlers.RewardForCollectingItems(items=item_rewards)
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

def register_logs_sword_env():
    """Register the custom environment"""
    env_id = "LogsAndIronSword-v0"
    
    # Unregister if already exists
    if env_id in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_id]
    
    # Register the environment
    gym.register(
        id=env_id,
        entry_point="minerl.env:MineRLEnv",
        kwargs={
            "env_spec": LogsAndIronSwordEnv()
        }
    )
    print(f"Successfully registered {env_id} environment")
