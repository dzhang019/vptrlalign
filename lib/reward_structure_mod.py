def custom_reward_function(obs, done, info, visited_chunks):
    print("Observation keys:", obs.keys())

    reward = 0
    if not done:
        reward += 1  # Reward for staying alive
    if obs.get("health", 100) < 20:
        reward -= 10  # Penalty for low health
    if done:
        reward -= 1000  # Large penalty for death
    
    # Exploration reward
    current_chunk = (obs['pos_x'] // 16, obs['pos_z'] // 16)
    if current_chunk not in visited_chunks:
        reward += 5  # Reward for exploring new chunks
        visited_chunks.add(current_chunk)
    
    return reward, visited_chunks

state_counts = {}

def curiosity_reward(state):
    state_id = hash(state.tobytes())  # Simplified state hashing
    reward = 1 / (1 + state_counts.get(state_id, 0))  # Reward decreases as state is visited
    state_counts[state_id] = state_counts.get(state_id, 0) + 1
    return reward
