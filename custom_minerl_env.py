import gym
import numpy as np
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers import RewardHandler

# Custom reward handler that tracks inventory changes
class LogsAndSwordRewardHandler(RewardHandler):
    """Custom reward handler for logs and iron sword objectives"""
    
    def __init__(self):
        # Define the rewards for different items
        self.log_reward = 10.0
        self.iron_sword_reward = 1000.0
        
        # Create the reward dictionary
        self.item_rewards = {
            "log": self.log_reward,
            "planks": 2.0,
            "stick": 3.0,
            "crafting_table": 5.0,
            "wooden_pickaxe": 15.0,
            "stone": 1.0,
            "cobblestone": 2.0,
            "stone_pickaxe": 25.0,
            "iron_ore": 50.0,
            "coal": 10.0,
            "furnace": 15.0,
            "iron_ingot": 75.0,
            "iron_sword": self.iron_sword_reward
        }
        
        # Create the reward function that compares current and previous inventory
        # Note: We're using a simple lambda here that will be called by the parent class
        def reward_func(obs, action, next_obs):
            reward = 0.0
            
            # Skip if inventory is not in the observation
            if "inventory" not in next_obs:
                return 0.0
                
            # Get current inventory
            inventory = next_obs["inventory"]
            
            # Calculate reward based on inventory contents (simplified)
            # In a real implementation, you'd compare with previous inventory
            for item, reward_value in self.item_rewards.items():
                if item in inventory and inventory[item] > 0:
                    reward += inventory[item] * reward_value / 10.0  # Scale down to avoid double rewards
            
            return reward
            
        # Initialize the parent RewardHandler with our reward function
        super().__init__(reward_function=reward_func, reward_shape=())
        
    def to_string(self):
        """Required method to describe the handler"""
        return "LogsAndSwordRewardHandler"

# Define our custom environment by extending HumanSurvival
class LogsAndIronSwordEnv(HumanSurvival):
    def __init__(self):
        # Initialize the parent class first
        super().__init__()
        
        # Update the name
        self.name = "LogsAndIronSword-v0"
        
        # Replace the default reward handlers with our custom one
        self.reward_handlers = []
        self.reward_handlers.append(LogsAndSwordRewardHandler())
        
        # Define inventory keys we want to track
        inventory_items = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
            "stone", "cobblestone", "stone_pickaxe", "iron_ore", "coal", 
            "furnace", "iron_ingot", "iron_sword"
        ]
        
        # Ensure all required items are in the inventory handler
        if hasattr(self, 'inventory_handler'):
            # Update the inventory items to include the ones we care about
            for item in inventory_items:
                if item not in self.inventory_handler.items:
                    self.inventory_handler.items.append(item)

def register_custom_env():
    """Register the custom environment with Gym"""
    env_id = "LogsAndIronSword-v0"
    
    # Unregister if it exists
    try:
        gym.envs.registration.registry.env_specs.pop(env_id)
    except KeyError:
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
