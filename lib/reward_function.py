def logs_and_sword_reward_function(obs, prev_obs=None, done=False, info=None):
    """
    Custom reward function for the Logs and Iron Sword environment.
    Rewards:
    - +10 for each log collected
    - +1000 for crafting an iron sword
    - Smaller rewards for intermediate items
    
    Args:
        obs: Current observation
        prev_obs: Previous observation (optional)
        done: Whether the episode is done
        info: Additional info from the environment
        
    Returns:
        reward: The calculated reward
    """
    # Initialize reward
    reward = 0.0
    
    # If we don't have previous observation, we can't calculate inventory changes
    if prev_obs is None:
        return 0.0
    
    # Get current and previous inventory
    curr_inventory = obs["inventory"]
    prev_inventory = prev_obs["inventory"]
    
    # Define reward values for different items
    item_rewards = {
        "log": 10.0,
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
        "iron_sword": 1000.0  # Massive reward for iron sword
    }
    
    # Calculate reward based on inventory changes
    for item, reward_value in item_rewards.items():
        curr_count = curr_inventory.get(item, 0)
        prev_count = prev_inventory.get(item, 0)
        
        # Reward for new items obtained
        if curr_count > prev_count:
            items_obtained = curr_count - prev_count
            item_reward = items_obtained * reward_value
            reward += item_reward
            print(f"Obtained {items_obtained} {item}(s)! Reward: +{item_reward}")
    
    # Penalty for death
    if done:
        reward -= 50.0
        print("Episode ended. Applying death penalty.")
    
    return reward
