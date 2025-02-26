def reward_function(obs, done, prev_max_height, steps_since_last_record):
    current_y = obs.get("location_stats", {}).get("ypos", 0)
    reward = 0
    new_max_height = prev_max_height

    # ======================
    # Core Height Incentives
    # ======================
    height_gain = current_y - prev_max_height
    if height_gain > 0:
        # Base reward (20x pretrained survival rate)
        reward += height_gain * 20.0
        
        # Exponential bonus for sustained climbing
        reward += (current_y ** 1.2) * 0.5
        
        # Reset stagnation counter
        steps_since_last_record = 0
        new_max_height = current_y
    else:
        # Penalize stagnation (no progress for N steps)
        steps_since_last_record += 1
        if steps_since_last_record > 100:
            reward -= 5.0

    # ======================
    # Anti-Survival Measures
    # ======================
    # Harsh penalty for dying (death = -100 vs pretrained ~-20)
    if done and obs.get("health", 0) <= 0:
        reward -= 100.0
    
    # Discourage resource gathering (optional)
    if obs.get("inventory", {}).get("dirt", 0) > 10:
        reward -= 2.0  # Penalize inventory hoarding

    return reward, new_max_height, steps_since_last_record
