def phase1_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """Phase 1: Focus on gathering logs"""
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    # Calculate rewards based on inventory changes
    if prev_inventory is not None:
        # Reward for logs (primary objective)
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            # Reward for gaining logs
            reward += (current_logs - prev_logs) * 25.0
        elif current_logs < prev_logs:
            # Penalty for losing logs
            reward -= (prev_logs - current_logs) * 30.0
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory

def phase2_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """Phase 2: Focus on crafting planks"""
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Small reward for logs (secondary objective)
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 5.0
        
        # Higher reward for planks (primary objective)
        prev_planks = prev_inventory.get("planks", 0)
        current_planks = current_inventory.get("planks", 0)
        if current_planks > prev_planks:
            reward += (current_planks - prev_planks) * 20.0
        elif current_planks < prev_planks:
            # Penalty for losing planks
            reward -= (prev_planks - current_planks) * 25.0
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory

def phase3_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """Phase 3: Focus on crafting tables"""
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Small reward for logs
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 2.0
        
        # Medium reward for planks
        prev_planks = prev_inventory.get("planks", 0)
        current_planks = current_inventory.get("planks", 0)
        if current_planks > prev_planks:
            reward += (current_planks - prev_planks) * 5.0
        
        # Large reward for crafting_table (primary objective)
        prev_tables = prev_inventory.get("crafting_table", 0)
        current_tables = current_inventory.get("crafting_table", 0)
        if current_tables > prev_tables:
            reward += (current_tables - prev_tables) * 25.0
        elif current_tables < prev_tables:
            # Penalty for losing crafting tables
            reward -= (prev_tables - current_tables) * 30.0
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory

def phase4_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """Phase 4: Focus on crafting sticks and wooden tools"""
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Small rewards for previous objectives
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 2.0
        
        prev_planks = prev_inventory.get("planks", 0)
        current_planks = current_inventory.get("planks", 0)
        if current_planks > prev_planks:
            reward += (current_planks - prev_planks) * 3.0
        
        # Reward for sticks (primary objective)
        prev_sticks = prev_inventory.get("stick", 0)
        current_sticks = current_inventory.get("stick", 0)
        if current_sticks > prev_sticks:
            reward += (current_sticks - prev_sticks) * 15.0
        elif current_sticks < prev_sticks:
            # Penalty for losing sticks
            reward -= (prev_sticks - current_sticks) * 20.0
        
        # Large reward for wooden tools
        prev_pick = prev_inventory.get("wooden_pickaxe", 0)
        current_pick = current_inventory.get("wooden_pickaxe", 0)
        if current_pick > prev_pick:
            reward += (current_pick - prev_pick) * 50.0
        elif current_pick < prev_pick:
            # Penalty for losing pickaxe
            reward -= (prev_pick - current_pick) * 60.0
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory

def phase5_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    """Phase 5: Focus on iron processing and sword crafting"""
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Minimal rewards for basic resources
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 1.0
        
        # Stone-related rewards
        prev_cobble = prev_inventory.get("cobblestone", 0)
        current_cobble = current_inventory.get("cobblestone", 0)
        if current_cobble > prev_cobble:
            reward += (current_cobble - prev_cobble) * 5.0
        
        # Stone pickaxe rewards (important for iron mining)
        prev_stone_pick = prev_inventory.get("stone_pickaxe", 0)
        current_stone_pick = current_inventory.get("stone_pickaxe", 0)
        if current_stone_pick > prev_stone_pick:
            reward += (current_stone_pick - prev_stone_pick) * 30.0
        elif current_stone_pick < prev_stone_pick:
            # Penalty for losing stone pickaxe
            reward -= (prev_stone_pick - current_stone_pick) * 40.0
        
        # Iron ore rewards
        prev_iron_ore = prev_inventory.get("iron_ore", 0)
        current_iron_ore = current_inventory.get("iron_ore", 0)
        if current_iron_ore > prev_iron_ore:
            reward += (current_iron_ore - prev_iron_ore) * 25.0
        elif current_iron_ore < prev_iron_ore:
            # Only small penalty since it might be smelted
            reward -= (prev_iron_ore - current_iron_ore) * 5.0
        
        # Iron ingot rewards (primary objective)
        prev_iron = prev_inventory.get("iron_ingot", 0)
        current_iron = current_inventory.get("iron_ingot", 0)
        if current_iron > prev_iron:
            reward += (current_iron - prev_iron) * 50.0
        elif current_iron < prev_iron:
            # Penalty for losing iron ingots
            reward -= (prev_iron - current_iron) * 60.0
        
        # Iron sword reward (ultimate objective)
        prev_sword = prev_inventory.get("iron_sword", 0)
        current_sword = current_inventory.get("iron_sword", 0)
        if current_sword > prev_sword:
            reward += (current_sword - prev_sword) * 1000.0
        elif current_sword < prev_sword:
            # Severe penalty for losing iron sword
            reward -= (prev_sword - current_sword) * 1500.0
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory
