def custom_reward_function(obs, done, info, visited_chunks):
    """
    Reward the agent for exploration (new biomes, chunks, or depths)
    and include health-related rewards and penalties.

    Args:
        obs: Observation dictionary from the environment.
        done: Boolean indicating if the episode is done.
        info: Info dictionary with additional environment metadata.
        visited_chunks: Set of previously visited chunks.

    Returns:
        Tuple containing the computed reward and updated visited_chunks.
    """
    #print("Observation keys:", obs.keys())

    # Initialize reward
    reward = 0

    # Health-related rewards and penalties
    current_health = 20.0
    if "life_stats" in obs:
        life_stats = obs["life_stats"]
        current_health = life_stats.get("life", 20)
    HEALTH_KEY = 'prev_health'
    
    if not done:
        reward += 0.0002  # Reward for staying alive
    
    if HEALTH_KEY in visited_chunks:
        prev_health = visited_chunks[HEALTH_KEY]
        
        # Calculate health change
        health_change = current_health - prev_health
        if health_change < 0:
            # Penalty for damage taken
            reward -= 0.05 * abs(health_change)
            print("took {health_change} damage")
    if current_health <= 4:  # 2 hearts or less
        # Exponential penalty as health approaches zero
        reward -= 0.2 * (5 - current_health)**2
        print("below 2 hearts")
    # if "life_stats" in obs:
    #     life_stats = obs["life_stats"]
    #     #if life_stats.get("life", 20) < 10:  # Assuming 20 is max health
    #     h = life_stats.get("life", 20)
    #     lowhealth = 0.00024 * (20/life_stats.get("life", 20)-1)  # Penalty for low health
    #     reward -= lowhealth
        #print(f"Health at {h}, penalizing by {lowhealth}")

    # Exploration reward: New chunks
    xpos, ypos, zpos = 0, 0, 0
    if "location_stats" in obs:
        location_stats = obs["location_stats"]
        xpos = location_stats.get("xpos", 0)
        ypos = location_stats.get("ypos", 0)
        zpos = location_stats.get("zpos", 0)

    current_chunk = (int(xpos) // 16, int(zpos) // 16)
    chunk_key = f"chunk_{current_chunk[0]}_{current_chunk[1]}"
    if chunk_key not in visited_chunks:
        reward += 60  # Reward for exploring new chunks
        visited_chunks.add(current_chunk)

    # Exploration reward: New depths
    # ypos_rounded = int(ypos)  # Round y-position to integer for tracking
    # if ypos_rounded not in visited_chunks:
    #     reward += 10  # Reward for exploring new depths
    #     visited_chunks.add(ypos_rounded)

    # Exploration reward: New biomes
    biome_id = info.get("biome_id", None)
    if biome_id is not None:
        biome_key = f"biome_{biome_id}"
        if biome_key not in visited_chunks:
            reward += 500  # Reward for discovering new biomes
            visited_chunks[biome_key] = True

    return reward, visited_chunks
