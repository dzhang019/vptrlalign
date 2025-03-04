def custom_reward_function(obs, done, info, visited_squares):
    """
    Reward function for encouraging survival, material collection, and construction.
    
    Modifications:
    - Reduced passive survival reward to prevent idle behavior.
    - Reward for collecting useful materials like wood and stone.
    - Penalty for collecting useless items like wheat seeds and flowers.
    - Increased reward for any construction to make it more attractive.
    
    Args:
        obs: Observation dictionary from the environment.
        done: Boolean indicating if the episode is done.
        info: Info dictionary with additional metadata.
        visited_squares: Set of squares (e.g., (x_chunk, z_chunk)) where a build event has occurred.
    
    Returns:
        Tuple containing the computed reward and the updated visited_squares.
    """
    reward = 0

    # --- Survival Incentives ---
    if not done:
        reward += 0.01  # Reduced from 0.1 to prevent passive survival exploitation

    # Penalize if health is low
    if "life_stats" in obs:
        life_stats = obs["life_stats"]
        if life_stats.get("life", 20) < 10:
            reward -= 30
            print("Low health detected: applying survival penalty.")

    # --- Material Collection Incentives ---
    if "inventory" in obs:
        inventory = obs["inventory"]

        # Reward useful materials
        wood_count = inventory.get("log", 0)
        stone_count = inventory.get("cobblestone", 0)

        if wood_count > 0:
            reward += 2 * wood_count  # Reward for collecting logs
            print(f"Collected {wood_count} logs -> reward: {2 * wood_count}")

        if stone_count > 0:
            reward += 3 * stone_count  # Reward for collecting stone
            print(f"Collected {stone_count} stone -> reward: {3 * stone_count}")

        # Penalize useless item collection (grass, flowers, wheat seeds)
        seeds = inventory.get("wheat_seeds", 0)
        flowers = sum(inventory.get(flower, 0) for flower in ["poppy", "dandelion", "blue_orchid"])

        if seeds > 0:
            reward -= 0.5 * seeds  # Small penalty for collecting wheat seeds
            print(f"Collected {seeds} wheat seeds -> penalty: {-0.5 * seeds}")

        if flowers > 0:
            reward -= 0.3 * flowers  # Small penalty for collecting flowers
            print(f"Collected {flowers} flowers -> penalty: {-0.3 * flowers}")

    # --- Building / Construction Incentives ---
    if "build_stats" in obs:
        build_stats = obs["build_stats"]
        build_location = build_stats.get("location", None)
        material = build_stats.get("material", "wood")

        material_multipliers = {
            "wood": 1,
            "stone": 1.5,
            "iron": 3,
            "gold": 2.5,
            "diamond": 4
        }
        material_reward = material_multipliers.get(material, 1)

        if build_location:
            square = (int(build_location[0]) // 16, int(build_location[2]) // 16)
            reward += 10 * material_reward  # Increased reward for any building
            visited_squares.add(square)
            print(f"Built in {square} using {material} -> reward: {10 * material_reward}")

    return reward, visited_squares
