def train_rl(in_model, in_weights, out_weights, num_iterations=10, rollout_steps=40):
    """
    This version collects `rollout_steps` transitions, then does a single gradient update
    on that entire mini-batch (partial rollout). It reuses the same calls from your single-step code.
    
    Args:
      in_model: path or config for the agent model
      in_weights: pretrained weights for both agent and pretrained policy
      out_weights: where to save fine-tuned weights
      num_iterations: how many times we collect a partial rollout & update
      rollout_steps: how many steps to collect before each update
    """

    # Hyperparameters, same as before (adjust as needed)
    LEARNING_RATE = 1e-5
    MAX_GRAD_NORM = 1.0      # For gradient clipping
    LAMBDA_KL = 1.0          # KL regularization weight

    env = HumanSurvival(**ENV_KWARGS).make()

    # 1) Load parameters for both current agent and pretrained agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)

    # Confirm weights are not shared
    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert agent_param.data_ptr() != pretrained_param.data_ptr(), "Weights are shared!"

    # 2) Create optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    # Stats tracking
    running_loss = 0.0
    total_steps = 0

    obs = env.reset()
    visited_chunks = set()

    # 3) Outer loop: do N partial-rollout iterations
    for iteration in range(num_iterations):
        print(f"Starting partial-rollout iteration {iteration}")

        # We will store up to `rollout_steps` transitions here
        transitions = []
        step_count = 0
        done = False
        cumulative_reward = 0.0

        while step_count < rollout_steps and not done:
            env.render()

            # --- A) Get action & training info from current agent ---
            minerl_action, pi_dist, v_pred, log_prob, new_hidden_state = \
                agent.get_action_and_training_info(obs, stochastic=True)

            # --- B) Get pretrained policy distribution (for KL) ---
            with th.no_grad():
                obs_for_pretrained = agent._env_obs_to_agent(obs)
                obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                (old_pi_dist, _, _), _ = pretrained_policy.policy(
                    obs=obs_for_pretrained,
                    state_in=pretrained_policy.policy.initial_state(1),
                    first=th.tensor([[False]], dtype=th.bool, device="cuda")
                )
            # Detach the pretrained distribution
            old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)

            # --- C) Step the environment ---
            try:
                next_obs, env_reward, done, info = env.step(minerl_action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            # --- D) Compute custom reward & accumulate ---
            reward, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            cumulative_reward += reward

            # --- E) Store transition for this partial rollout ---
            transitions.append({
                "obs": obs,             # raw obs at this step
                "pi_dist": pi_dist,     # current agent distribution
                "v_pred": v_pred,       # agent's value estimate
                "log_prob": log_prob,   # agent's log-prob of chosen action
                "old_pi_dist": old_pi_dist,  # pretrained policy distribution
                "reward": reward
            })

            obs = next_obs
            step_count += 1

        print(f"  Collected {len(transitions)} steps this iteration. Done={done}, "
              f"CumulativeReward={cumulative_reward}")

        # If the episode ended, reset for the next iteration
        if done:
            obs = env.reset()
            visited_chunks.clear()

        # --- F) Now do ONE gradient update for all transitions in `transitions` ---
        if len(transitions) == 0:
            print("  No transitions collected, continuing.")
            continue

        optimizer.zero_grad()
        total_loss_for_rollout = 0.0

        # The single-step logic is repeated, but now we sum over the partial rollout
        for step_data in transitions:
            v_pred_val = step_data["v_pred"].detach()
            advantage = step_data["reward"] - v_pred_val   # naive advantage

            # RL loss: -(advantage * log_prob)
            loss_rl = -(advantage * step_data["log_prob"])  

            # KL regularization
            loss_kl = compute_kl_loss(step_data["pi_dist"], step_data["old_pi_dist"])

            total_loss_step = loss_rl + LAMBDA_KL * loss_kl
            total_loss_for_rollout += total_loss_step

        # Average loss over the partial rollout
        total_loss_for_rollout = total_loss_for_rollout / len(transitions)

        # Backprop and update
        total_loss_for_rollout.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Update stats
        total_loss_val = total_loss_for_rollout.item()
        running_loss += total_loss_val * len(transitions)
        total_steps += len(transitions)
        if total_steps > 0:
            avg_loss = running_loss / total_steps
        else:
            avg_loss = 0.0

        print(f"  [Update] Loss={total_loss_val:.4f}, Steps so far={total_steps}, Avg Loss={avg_loss:.4f}")

    # 4) After all iterations, save fine-tuned weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)
