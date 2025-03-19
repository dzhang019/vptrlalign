import gym
import numpy as np
from minerl.herobraine.hero import handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler

# Create a custom reward handler that implements all required methods
class LogsAndSwordRewardHandler(Handler):
    def __init__(self):
        super().__init__()
        self.item_rewards = {
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
    
    def to_string(self):
        """Required method to describe the handler"""
        return "LogsAndSwordRewardHandler"
    
    def from_universal(self, obs):
        """Method to convert a universal observation to a reward"""
        reward = 0.0
        
        # Check if we have inventory
        if "inventory" in obs:
            inventory = obs["inventory"]
            
            # Calculate rewards based on inventory
            for item, item_reward in self.item_rewards.items():
                if item in inventory and inventory[item] > 0:
                    # Give reward based on current count
                    # This is a simplification since we don't track previous counts
                    count = inventory[item]
                    reward += count * item_reward / 10.0  # Scale to avoid double counting
                    
                    # Print for significant items
                    if item in ["log", "iron_sword"]:
                        print(f"Has {count} {item}! Reward contribution: {count * item_reward / 10.0}")
        
        return reward
    
    def reset(self):
        """Method called when the environment is reset"""
        pass

class LogsAndIronSwordEnv(HumanSurvival):
    def __init__(self):
        super().__init__()
        
        # Update the environment name
        self.name = "LogsAndIronSword-v0"
        
        # Remove default reward handlers
        self.reward_handlers = []
        
        # Add our custom reward handler
        self.reward_handlers.append(LogsAndSwordRewardHandler())
        
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
    try:
        if env_id in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs[env_id]
        print(f"Unregistered existing {env_id}")
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
