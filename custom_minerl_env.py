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
        
        # Add reward for collecting logs (10 points each)
        self.reward_handlers.append(
            reward_handlers.RewardForCollectingItems({
                "log": 10.0
            })
        )
        
        # Add reward for crafting an iron sword (1000 points)
        self.reward_handlers.append(
            reward_handlers.RewardForCollectingItems({
                "iron_sword": 1000.0
            })
        )
        
        # Add rewards for intermediate steps
        self.reward_handlers.append(
            reward_handlers.RewardForCollectingItems({
                # Wood processing
                "planks": 2.0,
                "stick": 3.0,
                "crafting_table": 5.0,
                "wooden_pickaxe": 15.0,
                
                # Stone processing
                "stone": 1.0,
                "cobblestone": 2.0,
                "stone_pickaxe": 25.0,
                
                # Iron processing
                "iron_ore": 50.0,
                "coal": 10.0,
                "furnace": 15.0,
                "iron_ingot": 75.0
            })
        )
        
        # Add death penalty
        self.reward_handlers.append(
            handlers.RewardHandler(
                reward_function=lambda obs, action, next_obs: -200.0 if obs.get("is_dead", False) else 0.0,
                reward_shape=()
            )
        )
        
        # Add exploration reward
        self.reward_handlers.append(
            handlers.RewardHandler(
                reward_function=lambda obs, action, next_obs: 1.0 if self._is_new_chunk(obs, next_obs) else 0.0,
                reward_shape=()
            )
        )
        
        # Ensure all required items are in the inventory observation
        required_items = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
            "stone", "cobblestone", "stone_pickaxe", "iron_ore", "coal", 
            "furnace", "iron_ingot", "iron_sword"
        ]
        
        # Make sure all required items are in the inventory handler
        for item in required_items:
            if item not in self.inventory_handler.items:
                self.inventory_handler.items.append(item)
    
    def _is_new_chunk(self, obs, next_obs):
        """Check if the agent moved to a new chunk"""
        if "xpos" not in next_obs or "zpos" not in next_obs:
            return False
            
        if "xpos" not in obs or "zpos" not in obs:
            return False
            
        current_chunk = (int(next_obs["xpos"] // 16), int(next_obs["zpos"] // 16))
        previous_chunk = (int(obs["xpos"] // 16), int(obs["zpos"] // 16))
        
        return current_chunk != previous_chunk

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
    
