def train_rl(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2,
    mini_batch_size=10
):
    """
    Improved pipelined training implementation that maintains GPU utilization
    while ensuring consistent advantage calculation
    """

    # ==== Hyperparams ====
    LEARNING_RATE = 3e-7
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 50.0
    GAMMA = 0.9999
    LAM = 0.95
    DEATH_PENALTY = -1000.0
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995

    # ==== 1) Create agent + pretrained policy ====
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)

    for p1, p2 in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert p1.data_ptr() != p2.data_ptr(), "Weights are shared!"

    # ==== 2) Create parallel envs + hidden states per env ====
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs

    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]

    # ==== 3) Optimizer, stats ====
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0

    # ==== 4) Improved pipelined training loop ====
    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Starting pipelined training...")
        
        # Initialize continuous rollout buffers for each environment
        # These store complete trajectories with history for proper GAE calculation
        env_buffers = [
            {
                "obs": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "hidden_states": [],
                "next_obs": [],
                "values": [],
                "log_probs": [],
                "cur_pds": [],
                "old_pds": []
            }
            for _ in range(num_envs)
        ]
        
        # Buffer for completed mini-episodes
        # (terminated by done=True or reaching window_size)
        completed_segments = []
        window_size = 20  # Sub-trajectory length for GAE calculation
        mini_batch_updates = 0
        
        step_count = 0
        while step_count < rollout_steps:
            step_count += 1
            for env_i in range(num_envs):
                envs[env_i].render()

                # Environment stepping + data collection
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1

                    # Prevent gradient tracking during rollout
                    with th.no_grad():
                        minerl_action_i, pi_dist_i, v_pred_i, log_prob_i, new_hid_i = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_i],
                            hidden_state=hidden_states[env_i],
                            stochastic=True,
                            taken_action=None
                        )
                        
                        # Get policy distribution from pretrained model for KL calculation
                        _, old_pd_i, _, _, _ = pretrained_policy.get_action_and_training_info(
                            obs_list[env_i], 
                            pretrained_policy.policy.initial_state(1),
                            stochastic=False,
                            taken_action=minerl_action_i
                        )

                    next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(minerl_action_i)
                    if "error" in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True

                    if done_flag_i:
                        env_reward_i += DEATH_PENALTY

                    # Store transition in environment buffer
                    env_buffers[env_i]["obs"].append(obs_list[env_i])
                    env_buffers[env_i]["actions"].append(minerl_action_i)
                    env_buffers[env_i]["rewards"].append(env_reward_i)
                    env_buffers[env_i]["dones"].append(done_flag_i)
                    env_buffers[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach(), hidden_states[env_i])
                    )
                    env_buffers[env_i]["next_obs"].append(next_obs_i)
                    env_buffers[env_i]["values"].append(v_pred_i.item())
                    env_buffers[env_i]["log_probs"].append(log_prob_i)
                    env_buffers[env_i]["cur_pds"].append(pi_dist_i)
                    env_buffers[env_i]["old_pds"].append(old_pd_i)

                    # Update with detached hidden state
                    obs_list[env_i] = next_obs_i
                    hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid_i)
                    done_list[env_i] = done_flag_i

                    # Check if we should extract a completed segment:
                    # 1. Episode is done, or
                    # 2. Buffer has reached window_size
                    if done_flag_i or len(env_buffers[env_i]["obs"]) >= window_size:
                        # Extract segment for training
                        segment = extract_segment(env_buffers[env_i])
                        
                        # Process segment with GAE
                        processed_segment = process_segment_with_gae(
                            segment, 
                            agent,
                            gamma=GAMMA, 
                            lam=LAM
                        )
                        
                        # Add to completed segments
                        completed_segments.extend(processed_segment)
                        
                        # Clear buffer (for done) or keep recent history (for window_size)
                        if done_flag_i:
                            # Clear buffer completely for new episode
                            for key in env_buffers[env_i]:
                                env_buffers[env_i][key] = []
                                
                            # Reset environment
                            with open(out_episodes, "a") as f:
                                f.write(f"{episode_step_counts[env_i]}\n")
                            episode_step_counts[env_i] = 0
                            obs_list[env_i] = envs[env_i].reset()
                            done_list[env_i] = False
                            hidden_states[env_i] = agent.policy.initial_state(batch_size=1)
                        else:
                            # Keep last observation for overlap (continuity in advantage calculation)
                            overlap = 1
                            for key in env_buffers[env_i]:
                                # Keep only the last element if the key has that many elements
                                if key == "hidden_states" and env_buffers[env_i][key]:
                                    env_buffers[env_i][key] = [env_buffers[env_i][key][-1]]
                                elif env_buffers[env_i][key]:
                                    env_buffers[env_i][key] = env_buffers[env_i][key][-overlap:]
                                else:
                                    env_buffers[env_i][key] = []
            
            # Perform mini-batch update if we have enough transitions
            if len(completed_segments) >= mini_batch_size:
                print(f"[Iteration {iteration}] Mini-batch update {mini_batch_updates+1} with {len(completed_segments)} transitions")
                
                # Take a subset of mini_batch_size samples (or all if fewer)
                batch_size = min(mini_batch_size, len(completed_segments))
                batch_indices = np.random.choice(len(completed_segments), batch_size, replace=False)
                batch = [completed_segments[i] for i in batch_indices]
                
                # Perform update
                loss_val = update_policy(agent, batch, optimizer, LAMBDA_KL, VALUE_LOSS_COEF, MAX_GRAD_NORM)
                
                # Update stats
                running_loss += loss_val * batch_size
                total_steps += batch_size
                mini_batch_updates += 1
                
                print(f"[Mini-batch {mini_batch_updates}] Loss={loss_val:.4f}, TotalSteps={total_steps}")
                
                # Remove processed transitions
                for i in sorted(batch_indices, reverse=True):
                    del completed_segments[i]
        
        # Process any remaining segments at end of iteration
        for env_i in range(num_envs):
            if env_buffers[env_i]["obs"]:
                segment = extract_segment(env_buffers[env_i])
                processed_segment = process_segment_with_gae(
                    segment, 
                    agent,
                    gamma=GAMMA, 
                    lam=LAM
                )
                completed_segments.extend(processed_segment)
                
                # Clear buffer for next iteration
                for key in env_buffers[env_i]:
                    env_buffers[env_i][key] = []
        
        # Final update with any remaining transitions
        if completed_segments:
            print(f"[Iteration {iteration}] Final update with {len(completed_segments)} remaining transitions")
            loss_val = update_policy(agent, completed_segments, optimizer, LAMBDA_KL, VALUE_LOSS_COEF, MAX_GRAD_NORM)
            running_loss += loss_val * len(completed_segments)
            total_steps += len(completed_segments)
        
        # Update KL coefficient
        LAMBDA_KL *= KL_DECAY
        
        # Log iteration summary
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        print(f"[Iteration {iteration}] Complete - MiniBatchUpdates={mini_batch_updates}, "
              f"TotalSteps={total_steps}, AvgLoss={avg_loss:.4f}, KL_Coeff={LAMBDA_KL:.4f}")

    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)


def extract_segment(buffer):
    """Extract a segment from the buffer with all required training data"""
    segment = {}
    for key in buffer:
        segment[key] = buffer[key].copy()
    return segment


def process_segment_with_gae(segment, agent, gamma=0.9999, lam=0.95):
    """Process a segment using GAE for advantage calculation"""
    processed_transitions = []
    T = len(segment["obs"])
    if T == 0:
        return processed_transitions

    # If last transition is not done, bootstrap value
    if not segment["dones"][-1]:
        with th.no_grad():
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=segment["next_obs"][-1],
                hidden_state=segment["hidden_states"][-1],
                stochastic=False,
                taken_action=None
            )
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0

    # Calculate advantages using GAE
    values = segment["values"]
    rewards = segment["rewards"]
    dones = segment["dones"]
    
    # Initialize advantages and returns arrays
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    # GAE calculation
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = bootstrap_value
        else:
            next_value = values[t + 1]
            
        # Calculate delta (TD error)
        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        
        # Calculate GAE
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        
        # Store advantage and return
        advantages[t] = gae
        returns[t] = values[t] + gae
    
    # Create processed transitions
    for t in range(T):
        processed_transitions.append({
            "obs": segment["obs"][t],
            "action": segment["actions"][t],
            "reward": segment["rewards"][t],
            "done": segment["dones"][t],
            "v_pred": th.tensor(values[t], device="cuda"),
            "log_prob": segment["log_probs"][t],
            "cur_pd": segment["cur_pds"][t],
            "old_pd": segment["old_pds"][t],
            "advantage": advantages[t],
            "return": returns[t]
        })

    return processed_transitions


def update_policy(agent, batch, optimizer, lambda_kl, value_loss_coef, max_grad_norm):
    """Perform policy update with the provided batch of transitions"""
    optimizer.zero_grad()
    
    # Compute average loss over batch
    loss_rl_list = []
    value_loss_list = []
    kl_loss_list = []

    for transition in batch:
        advantage = transition["advantage"]
        returns = transition["return"]
        log_prob = transition["log_prob"]
        v_pred = transition["v_pred"]
        cur_pd = transition["cur_pd"]
        old_pd = transition["old_pd"]

        # Compute losses
        loss_rl = -(advantage * log_prob)
        value_loss = (v_pred - th.tensor(returns, device="cuda")) ** 2
        kl_loss = compute_kl_loss(cur_pd, old_pd)

        loss_rl_list.append(loss_rl)
        value_loss_list.append(value_loss)
        kl_loss_list.append(kl_loss)

    # Average losses
    loss_rl = th.stack(loss_rl_list).mean()
    value_loss = th.stack(value_loss_list).mean()
    kl_loss = th.stack(kl_loss_list).mean()

    # Compute total loss
    total_loss = loss_rl + (value_loss_coef * value_loss) + (lambda_kl * kl_loss)
    
    # Backward pass and optimization
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    optimizer.step()
    
    return total_loss.item()