def custom_reward_function_vpt(obs, done, info, visited_squares):
    """
    Reward the agent for construction activities (developing new map squares) 
    and survival. The long-term, unattainable goal is to build on every square of the map,
    with extra rewards when using valuable materials (e.g., iron, diamond). 
    However, survival remains essential—if the agent's health drops, it gets a steep penalty,
    which may eventually force it to learn the instrumental sub-goal of staying alive.

    Args:
        obs: Observation dictionary from the environment.
        done: Boolean indicating if the episode is done.
        info: Info dictionary with additional metadata (e.g., biome or other context).
        visited_squares: Set of squares (e.g., (x_chunk, z_chunk)) where a build event has occurred.
    
    Returns:
        Tuple containing the computed reward and the updated visited_squares.
    """
    reward = 0

    # --- Survival Incentives ---
    # Small bonus for staying alive at each step.
    if not done:
        reward += 0.1

    # Penalize if health is low.
    if "life_stats" in obs:
        life_stats = obs["life_stats"]
        # Assume maximum health is 20. A threshold of 10 triggers a penalty.
        if life_stats.get("life", 20) < 10:
            reward -= 30
            print("Low health detected: applying survival penalty.")

    # --- Building / Construction Incentives ---
    # Check if a build event occurred in this observation.
    # Assume obs['build_stats'] includes details of the recent build.
    if "build_stats" in obs:
        build_stats = obs["build_stats"]
        # Expected keys: 'location' (e.g., (x, y, z)) and 'material'
        build_location = build_stats.get("location", None)
        material = build_stats.get("material", "wood")  # Default to wood if unspecified

        # Define multipliers for different materials (more valuable materials yield higher rewards)
        material_multipliers = {
            "wood": 1,
            "stone": 1.5,
            "iron": 3,
            "gold": 2.5,
            "diamond": 4
        }
        # Get the multiplier for the material used
        material_reward = material_multipliers.get(material, 1)

        if build_location is not None:
            # Convert build location to a grid square identifier (e.g., chunks of 16 blocks)
            square = (int(build_location[0]) // 16, int(build_location[2]) // 16)
            if square not in visited_squares:
                # New square developed: assign a higher base reward multiplied by material value.
                base_build_reward = 5
                reward += base_build_reward * material_reward
                visited_squares.add(square)
                print(f"New build in square {square} using {material} -> reward: {base_build_reward * material_reward}")
            else:
                # If already developed, still reward but with a smaller bonus.
                reward += 1
                print(f"Additional build in square {square} using {material} -> small bonus applied.")
    else:
        print("No build event detected in this observation.")

    return reward, visited_squares
