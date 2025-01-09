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
    print("Observation keys:", obs.keys())

    # Initialize reward
    reward = 0

    # Health-related rewards and penalties
    if not done:
        reward += 1  # Reward for staying alive
    if "life_stats" in obs:
        life_stats = obs["life_stats"]
        if life_stats.get("life", 20) < 10:  # Assuming 20 is max health
            reward -= 10  # Penalty for low health
        if not life_stats.get("is_alive", True):
            reward -= 1000  # Large penalty for death

    # Exploration reward: New chunks
    xpos, ypos, zpos = 0, 0, 0
    if "location_stats" in obs:
        location_stats = obs["location_stats"]
        xpos = location_stats.get("xpos", 0)
        ypos = location_stats.get("ypos", 0)
        zpos = location_stats.get("zpos", 0)

    current_chunk = (int(xpos) // 16, int(zpos) // 16)
    if current_chunk not in visited_chunks:
        reward += 5  # Reward for exploring new chunks
        visited_chunks.add(current_chunk)

    # Exploration reward: New depths
    ypos_rounded = int(ypos)  # Round y-position to integer for tracking
    if ypos_rounded not in visited_chunks:
        reward += 2  # Reward for exploring new depths
        visited_chunks.add(ypos_rounded)

    # Exploration reward: New biomes
    biome_id = info.get("biome_id", None)
    if biome_id is not None and biome_id not in visited_chunks:
        reward += 3  # Reward for discovering new biomes
        visited_chunks.add(biome_id)

    return reward, visited_chunks
