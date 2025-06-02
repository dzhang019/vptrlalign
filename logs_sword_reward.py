def custom_reward_function_for_logs_and_sword(next_obs, done, info, visited_chunks=None):
    """
    Custom reward function for the Logs and Iron Sword objectives.
    
    Args:
        next_obs: Current observation
        done: Whether the episode is done
        info: Additional info from the environment
        visited_chunks: Set of visited chunks (for exploration rewards)
        
    Returns:
        reward: The calculated reward
        visited_chunks: Updated visited chunks set
    """
    # Initialize reward
    reward = 0.0
    
    # Initialize visited_chunks if None
    if visited_chunks is None:
        visited_chunks = set()
    
    # Calculate inventory-based rewards
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        
        # Define reward values for different items
        item_rewards = {
            "log": 10.0,           # Good reward for logs (primary objective)
            "planks": 2.0,
            "stick": 3.0,
            "crafting_table": 5.0,
            "wooden_pickaxe": 15.0,
            "cobblestone": 2.0,
            "stone_pickaxe": 25.0,
            "iron_ore": 50.0,
            "coal": 10.0,
            "furnace": 15.0,
            "iron_ingot": 75.0,
            "iron_sword": 1000.0   # Massive reward for iron sword (ultimate objective)
        }
        
        # Get current inventory count for relevant items
        for item, reward_value in item_rewards.items():
            if item in inventory and inventory[item] > 0:
                # Give reward based on current inventory count
                reward += inventory[item] * reward_value
                
                # Print updates for significant items
                if item in ["log", "iron_ore", "iron_ingot", "iron_sword"]:
                    print(f"Agent has {inventory[item]} {item}(s)! Reward: +{inventory[item] * reward_value}")
    
    # Add exploration reward based on visited chunks
    if "xpos" in next_obs and "zpos" in next_obs:
        # Get current chunk coordinates
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        # Reward for exploring new chunks
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0  # Small reward for exploration
    
    # Penalty for death
    if done:
        reward -= 200.0  # Significant penalty for dying
    
    return reward, visited_chunks
