import numpy as np
from collections import defaultdict

# Material multipliers for building rewards (customize based on your goals)
MATERIAL_MULTIPLIERS = {
    "dirt": 1.0,
    "cobblestone": 1.5,
    "wood": 2.0,
    "iron_block": 3.0,
    "diamond_block": 5.0
}

def reward_function(obs, done, prev_inventory, visited_chunks, prev_health):
    """
    Custom reward function for MineRL construction and survival.
    Tracks progress across episodes using external state variables.
    
    Args:
        obs: Current environment observation
        done: Episode termination flag
        prev_inventory: Inventory state from previous step
        visited_chunks: Persistent set of chunks (tuples) where building occurred
        prev_health: Previous health value for delta calculations
    
    Returns:
        tuple: (reward, updated_visited_chunks, current_inventory, current_health)
    """
    reward = 0
    current_inventory = defaultdict(int, obs.get("inventory", {}))
    current_health = obs.get("health", 20)

    # ====================
    # Survival Components
    # ====================
    # Per-step survival bonus
    if not done:
        reward += 1.0  # Increased from original 0.1 to match building rewards

    # Health-based penalties (scaled to -10 instead of -30)
    if current_health < 10:
        reward -= 10.0
        # Optional: Add penalty proportional to health loss
        if prev_health > current_health:
            reward -= (prev_health - current_health) * 0.5

    # ====================
    # Building Components
    # ====================
    building_detected = False
    
    # Detect block placements by inventory depletion
    for item, count in current_inventory.items():
        prev_count = prev_inventory.get(item, 0)
        if count < prev_count:
            # Assume 1 block placed per missing item (simplified)
            blocks_placed = prev_count - count
            reward += _calculate_building_reward(item, blocks_placed, obs, visited_chunks)
            building_detected = True

    # ====================
    # Exploration Bonus (Optional)
    # ====================
    if not building_detected:
        # Small reward for moving to new areas (x, z coordinates)
        x = obs.get("location_stats", {}).get("xpos", 0)
        z = obs.get("location_stats", {}).get("zpos", 0)
        chunk = (int(x) // 16, int(z) // 16)
        if chunk not in visited_chunks:
            reward += 0.5
            visited_chunks.add(chunk)

    return (
        reward,
        visited_chunks,
        defaultdict(int, current_inventory),  # Ensure consistent type
        current_health
    )

def _calculate_building_reward(material, quantity, obs, visited_chunks):
    """Calculate reward for placing blocks of a specific material"""
    base_reward = 2.0  # Reduced from original 5 to balance with survival
    
    # Get material multiplier
    multiplier = MATERIAL_MULTIPLIERS.get(
        material.split("_")[0],  # Handle "log_acacia" -> "log"
        1.0
    )
    
    # Get chunk coordinates from position
    x = obs.get("location_stats", {}).get("xpos", 0)
    z = obs.get("location_stats", {}).get("zpos", 0)
    chunk = (int(x) // 16, int(z) // 16)
    
    # Calculate chunk-based bonus
    if chunk in visited_chunks:
        chunk_bonus = 1.0
    else:
        chunk_bonus = 3.0
        visited_chunks.add(chunk)
    
    return base_reward * multiplier * chunk_bonus * quantity

# ====================
# Initialization Helper
# ====================
def init_reward_function():
    """
    Returns initial state for a new episode
    """
    return (
        defaultdict(int),  # prev_inventory
        set(),             # visited_chunks
        20                 # prev_health
    )
