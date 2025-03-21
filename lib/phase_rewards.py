# Phase 1: Focus on gathering logs
def phase1_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    # Only reward for logs and basic wood items
    if prev_inventory is not None:
        # Calculate new logs obtained
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 10.0
    
    # Exploration reward
    if visited_chunks is None:
        visited_chunks = set()
    
    # Add chunk-based exploration reward
    if "xpos" in next_obs and "zpos" in next_obs:
        x_chunk = int(next_obs["xpos"] // 16)
        z_chunk = int(next_obs["zpos"] // 16)
        chunk_pos = (x_chunk, z_chunk)
        
        if chunk_pos not in visited_chunks:
            visited_chunks.add(chunk_pos)
            reward += 1.0  # Small reward for exploration
    
    # Death penalty
    if done:
        reward -= 200.0
    
    return reward, visited_chunks, current_inventory

# Phase 2: Focus on crafting planks
def phase2_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    # Calculate rewards for logs and planks
    if prev_inventory is not None:
        # Small reward for logs (less than in phase 1)
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 2.0  # Reduced reward
            
        # Larger reward for planks
        prev_planks = prev_inventory.get("planks", 0)
        current_planks = current_inventory.get("planks", 0)
        if current_planks > prev_planks:
            reward += (current_planks - prev_planks) * 10.0
    
    # Add exploration reward
    # [Same exploration code as phase 1]
    
    return reward, visited_chunks, current_inventory

# Phase 3: Focus on crafting tables and sticks
def phase3_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Small rewards for logs and planks
        prev_logs = prev_inventory.get("log", 0)
        current_logs = current_inventory.get("log", 0)
        if current_logs > prev_logs:
            reward += (current_logs - prev_logs) * 1.0  # Minimal reward
            
        prev_planks = prev_inventory.get("planks", 0)
        current_planks = current_inventory.get("planks", 0)
        if current_planks > prev_planks:
            reward += (current_planks - prev_planks) * 2.0  # Reduced reward
            
        # Larger rewards for crafting tables and sticks
        prev_tables = prev_inventory.get("crafting_table", 0)
        current_tables = current_inventory.get("crafting_table", 0)
        if current_tables > prev_tables:
            reward += (current_tables - prev_tables) * 15.0
            
        prev_sticks = prev_inventory.get("stick", 0)
        current_sticks = current_inventory.get("stick", 0)
        if current_sticks > prev_sticks:
            reward += (current_sticks - prev_sticks) * 10.0
    
    # [Exploration reward as before]
    
    return reward, visited_chunks, current_inventory

# Phase 4: Focus on getting stone tools
def phase4_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Gradually reduce rewards for earlier objectives
        # [Similar pattern for logs, planks, etc. with reduced values]
        
        # Higher rewards for stone-related items
        prev_cobble = prev_inventory.get("cobblestone", 0)
        current_cobble = current_inventory.get("cobblestone", 0)
        if current_cobble > prev_cobble:
            reward += (current_cobble - prev_cobble) * 5.0
            
        prev_pickaxe = prev_inventory.get("stone_pickaxe", 0)
        current_pickaxe = current_inventory.get("stone_pickaxe", 0)
        if current_pickaxe > prev_pickaxe:
            reward += (current_pickaxe - prev_pickaxe) * 50.0  # Big reward for stone pickaxe
    
    return reward, visited_chunks, current_inventory

# Phase 5: Focus on iron processing and sword crafting
def phase5_rewards(next_obs, done, info, visited_chunks=None, prev_inventory=None):
    reward = 0.0
    current_inventory = next_obs.get("inventory", {})
    
    if prev_inventory is not None:
        # Minimal rewards for previous phase items
        # [Reduced rewards for wood/stone items]
        
        # Large rewards for iron-related items
        prev_iron_ore = prev_inventory.get("iron_ore", 0)
        current_iron_ore = current_inventory.get("iron_ore", 0)
        if current_iron_ore > prev_iron_ore:
            reward += (current_iron_ore - prev_iron_ore) * 25.0
            
        prev_iron = prev_inventory.get("iron_ingot", 0)
        current_iron = current_inventory.get("iron_ingot", 0)
        if current_iron > prev_iron:
            reward += (current_iron - prev_iron) * 50.0
            
        # Only in phase 5: Huge reward for iron sword
        prev_sword = prev_inventory.get("iron_sword", 0)
        current_sword = current_inventory.get("iron_sword", 0)
        if current_sword > prev_sword:
            reward += (current_sword - prev_sword) * 5000.0  # Massive reward
    
    return reward, visited_chunks, current_inventory
