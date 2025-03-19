def phase1_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """
    Phase 1: Focus on log collection with future goal hints and penalties for decreasing logs
    """
    # Initialize reward and visited chunks
    reward = 0.0
    if visited_chunks is None:
        visited_chunks = set()
    
    # Store current inventory for comparison with next step
    current_inventory = {}
    
    # Calculate inventory-based rewards and penalties
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        current_inventory = {k: v for k, v in inventory.items()}  # Store for next call
        
        # PRIMARY FOCUS: Massive reward for logs
        if "log" in inventory and inventory["log"] > 0:
            # Base reward
            reward += inventory["log"] * 50.0
            print(f"Logs: {inventory['log']} - Reward: +{inventory['log'] * 50.0}")
            
            # Check if logs decreased since last observation
            if prev_inventory is not None and "log" in prev_inventory:
                if inventory["log"] < prev_inventory["log"]:
                    logs_lost = prev_inventory["log"] - inventory["log"]
                    penalty = logs_lost * 150.0  # 3x penalty for losing logs
                    reward -= penalty
                    print(f"Lost {logs_lost} logs! Penalty: -{penalty}")
        
        # Other rewards as before...
        if "planks" in inventory and inventory["planks"] > 0:
            reward += inventory["planks"] * 1.0  # Small hint
        
        if "iron_sword" in inventory and inventory["iron_sword"] > 0:
            reward += 5000.0  # Massive reward
            print(f"IRON SWORD CRAFTED! Bonus: +5000.0")
    
    # Exploration reward
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0  # Small exploration bonus
    
    # Death penalty
    if done:
        reward -= 100.0
    
    return reward, visited_chunks, current_inventory

def phase2_rewards(next_obs, done, info, visited_chunks=None):
    """
    Phase 2: Focus on crafting planks while maintaining log collection
    """
    # Initialize reward and visited chunks
    reward = 0.0
    if visited_chunks is None:
        visited_chunks = set()
    
    # Calculate inventory-based rewards
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        
        # PREVIOUS FOCUS: Medium reward for logs (still important)
        if "log" in inventory and inventory["log"] > 0:
            reward += inventory["log"] * 10.0  # Reduced but still significant
        
        # PRIMARY FOCUS: Massive reward for planks
        if "planks" in inventory and inventory["planks"] > 0:
            reward += inventory["planks"] * 40.0  # Very high reward for planks
            print(f"Planks: {inventory['planks']} - Reward: +{inventory['planks'] * 40.0}")
        
        # FUTURE HINTS: Small rewards for next steps
        if "stick" in inventory and inventory["stick"] > 0:
            reward += inventory["stick"] * 2.0  # Small hint
        
        if "crafting_table" in inventory and inventory["crafting_table"] > 0:
            reward += inventory["crafting_table"] * 5.0  # Small hint
        
        # ULTIMATE GOAL: Massive reward for iron sword
        if "iron_sword" in inventory and inventory["iron_sword"] > 0:
            reward += 5000.0
            print(f"IRON SWORD CRAFTED! Bonus: +5000.0")
    
    # Exploration and death penalty
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0
    
    if done:
        reward -= 100.0
    
    return reward, visited_chunks

def phase3_rewards(next_obs, done, info, visited_chunks=None):
    """
    Phase 3: Focus on crafting tables while maintaining previous skills
    """
    # Initialize reward and visited chunks
    reward = 0.0
    if visited_chunks is None:
        visited_chunks = set()
    
    # Calculate inventory-based rewards
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        
        # PREVIOUS SKILLS: Moderate rewards
        if "log" in inventory and inventory["log"] > 0:
            reward += inventory["log"] * 5.0  # Reduced but still relevant
        
        if "planks" in inventory and inventory["planks"] > 0:
            reward += inventory["planks"] * 10.0  # Reduced but still significant
        
        # PRIMARY FOCUS: Massive reward for crafting tables
        if "crafting_table" in inventory and inventory["crafting_table"] > 0:
            reward += inventory["crafting_table"] * 100.0  # Very high reward
            print(f"Crafting Tables: {inventory['crafting_table']} - Reward: +{inventory['crafting_table'] * 100.0}")
        
        # FUTURE HINTS: Small rewards for next steps
        if "stick" in inventory and inventory["stick"] > 0:
            reward += inventory["stick"] * 5.0  # Increased hint for sticks
        
        if "wooden_pickaxe" in inventory and inventory["wooden_pickaxe"] > 0:
            reward += inventory["wooden_pickaxe"] * 20.0  # Hint for pickaxe
        
        # ULTIMATE GOAL: Massive reward for iron sword
        if "iron_sword" in inventory and inventory["iron_sword"] > 0:
            reward += 5000.0
            print(f"IRON SWORD CRAFTED! Bonus: +5000.0")
    
    # Exploration and death penalty
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0
    
    if done:
        reward -= 100.0
    
    return reward, visited_chunks

def phase4_rewards(next_obs, done, info, visited_chunks=None):
    """
    Phase 4: Focus on stick crafting while maintaining previous skills
    """
    # Initialize reward and visited chunks
    reward = 0.0
    if visited_chunks is None:
        visited_chunks = set()
    
    # Calculate inventory-based rewards
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        
        # PREVIOUS SKILLS: Small to moderate rewards
        if "log" in inventory and inventory["log"] > 0:
            reward += inventory["log"] * 2.0  # Reduced but still useful
        
        if "planks" in inventory and inventory["planks"] > 0:
            reward += inventory["planks"] * 5.0  # Reduced but needed for sticks
        
        if "crafting_table" in inventory and inventory["crafting_table"] > 0:
            reward += inventory["crafting_table"] * 20.0  # Reduced but still important
        
        # PRIMARY FOCUS: Massive reward for sticks
        if "stick" in inventory and inventory["stick"] > 0:
            reward += inventory["stick"] * 80.0  # Very high reward
            print(f"Sticks: {inventory['stick']} - Reward: +{inventory['stick'] * 80.0}")
        
        # FUTURE HINTS: Moderate rewards for next steps
        if "wooden_pickaxe" in inventory and inventory["wooden_pickaxe"] > 0:
            reward += inventory["wooden_pickaxe"] * 40.0  # Higher hint for pickaxe
        
        if "stone_pickaxe" in inventory and inventory["stone_pickaxe"] > 0:
            reward += inventory["stone_pickaxe"] * 60.0  # Hint for stone pickaxe
        
        if "cobblestone" in inventory and inventory["cobblestone"] > 0:
            reward += inventory["cobblestone"] * 10.0  # Hint for stone
        
        # ULTIMATE GOAL: Massive reward for iron sword
        if "iron_sword" in inventory and inventory["iron_sword"] > 0:
            reward += 5000.0
            print(f"IRON SWORD CRAFTED! Bonus: +5000.0")
    
    # Exploration and death penalty
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0
    
    if done:
        reward -= 100.0
    
    return reward, visited_chunks

def phase5_rewards(next_obs, done, info, visited_chunks=None):
    """
    Phase 5: Focus on iron processing with ultimate goal of iron sword
    """
    # Initialize reward and visited chunks
    reward = 0.0
    if visited_chunks is None:
        visited_chunks = set()
    
    # Calculate inventory-based rewards
    if "inventory" in next_obs:
        inventory = next_obs["inventory"]
        
        # PREVIOUS SKILLS: Small rewards (assumed to be well-learned by now)
        wood_items = {
            "log": 1.0,
            "planks": 2.0,
            "stick": 3.0,
            "crafting_table": 5.0,
            "wooden_pickaxe": 10.0
        }
        
        for item, value in wood_items.items():
            if item in inventory and inventory[item] > 0:
                reward += inventory[item] * value
        
        # TRANSITIONAL SKILLS: Moderate rewards
        if "cobblestone" in inventory and inventory["cobblestone"] > 0:
            reward += inventory["cobblestone"] * 5.0
        
        if "stone_pickaxe" in inventory and inventory["stone_pickaxe"] > 0:
            reward += inventory["stone_pickaxe"] * 30.0
        
        if "furnace" in inventory and inventory["furnace"] > 0:
            reward += inventory["furnace"] * 50.0
        
        if "coal" in inventory and inventory["coal"] > 0:
            reward += inventory["coal"] * 40.0
            
        # PRIMARY FOCUS: Iron processing
        if "iron_ore" in inventory and inventory["iron_ore"] > 0:
            reward += inventory["iron_ore"] * 100.0
            print(f"Iron Ore: {inventory['iron_ore']} - Reward: +{inventory['iron_ore'] * 100.0}")
        
        if "iron_ingot" in inventory and inventory["iron_ingot"] > 0:
            reward += inventory["iron_ingot"] * 200.0
            print(f"Iron Ingots: {inventory['iron_ingot']} - Reward: +{inventory['iron_ingot'] * 200.0}")
        
        # ULTIMATE GOAL: Massive reward for iron sword
        if "iron_sword" in inventory and inventory["iron_sword"] > 0:
            reward += 5000.0
            print(f"IRON SWORD CRAFTED! Bonus: +5000.0")
    
    # Exploration and death penalty
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0
    
    if done:
        reward -= 100.0
    
    return reward, visited_chunks
