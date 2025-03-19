import gym
import numpy as np
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers import RewardHandler

# Custom reward handler that tracks inventory changes
class LogsAndSwordRewardHandler(Handler):
    def __init__(self):
        self.prev_inventory = {}
        self.log_reward = 10.0
        self.iron_sword_reward = 1000.0
        
        # Define rewards for intermediate items to guide the agent
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
        
    def from_universal(self, obs):
        reward = 0.0
        
        # Get current inventory
        inventory = obs["inventory"]
        
        # Calculate reward based on inventory changes
        for item, reward_value in self.item_rewards.items():
            current_count = inventory.get(item, 0)
            prev_count = self.prev_inventory.get(item, 0)
            
            # Reward for new items obtained
            if current_count > prev_count:
                items_obtained = current_count - prev_count
                reward += items_obtained * reward_value
                print(f"Obtained {items_obtained} {item}(s)! Reward: +{items_obtained * reward_value}")
        
        # Update previous inventory for next comparison
        self.prev_inventory = inventory.copy()
        
        return reward

    def reset(self):
        self.prev_inventory = {}

# Define our custom environment by extending HumanSurvival
class LogsAndIronSwordEnv(HumanSurvival):
    def __init__(self):
        super().__init__()
        
        # Update the name
        self.name = "LogsAndIronSword-v0"
        
        # Add our custom reward handler
        self.reward_handlers.append(LogsAndSwordRewardHandler())
        
        # Add more detailed inventory observation to track all relevant items
        self._inventory_keys = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
            "stone", "cobblestone", "stone_pickaxe", "iron_ore", "coal", 
            "furnace", "iron_ingot", "iron_sword"
        ]
        
        # Update the observation space if needed
        for item in self._inventory_keys:
            if item not in self.observation_space.spaces["inventory"].spaces:
                self.observation_space.spaces["inventory"].spaces[item] = gym.spaces.Box(
                    low=0, high=2304, shape=(), dtype=np.int32
                )

# Register our custom environment
def register_custom_env():
    gym.register(
        id="LogsAndIronSword-v0",
        entry_point="minerl.env:MineRLEnv",
        kwargs={
            "env_spec": LogsAndIronSwordEnv()
        }
    )
