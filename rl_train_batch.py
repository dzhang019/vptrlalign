from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from data_loader import DataLoader
from lib.tree_util import tree_map

from lib.reward_structure_mod import custom_reward_function
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def compute_gae(transitions, agent, gamma=0.999, lam=0.95):
    """
    Compute GAE for a partial rollout.
    Each item in 'transitions' is a dict with:
      {
        "obs": obs_t,
        "next_obs": obs_{t+1},
        "reward": r_t,
        "v_pred": v_t (tensor),
        "done": bool,
        "log_prob": log_prob_t (tensor),
        "pi_dist": ...,
        "old_pi_dist": ...,
        ...
      }
    We add:
      transition["advantage"] = ...
      transition["return"] = ...
    """
    # 1) If last step isn't done, bootstrap final value
    if not transitions[-1]["done"]:
        with th.no_grad():
            # Next obs of last transition
            final_next_obs = transitions[-1]["next_obs"]
            # Get the value of final_next_obs from agent
            # We don't need the full distribution here
            _, _, v_next, _, _ = agent.get_action_and_training_info(final_next_obs, stochastic=False)
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0

    # 2) GAE calculation
    gae = 0.0
    for i in reversed(range(len(transitions))):
        r_t = transitions[i]["reward"]
        v_t = transitions[i]["v_pred"].item()
        done_t = transitions[i]["done"]
        mask = 1.0 - float(done_t)

        if i == len(transitions) - 1:
            next_value = bootstrap_value
        else:
            next_value = transitions[i+1]["v_pred"].item()

        delta = r_t + gamma * next_value * mask - v_t
        gae = delta + gamma * lam * mask * gae
        transitions[i]["advantage"] = gae
        transitions[i]["return"] = v_t + gae

    return transitions


def train_rl(in_model, in_weights, out_weights, num_iterations=10, rollout_steps=40):
    """
    Partial-rollout training with GAE + Value Loss + KL regularization.
    """

    # Hyperparameters
    LEARNING_RATE = 1e-5
    MAX_GRAD_NORM = 1.0             # For gradient clipping
    LAMBDA_KL = 1.0                 # KL regularization weight
    GAMMA = 0.999                   # discount factor
    LAM = 0.95                      # GAE lambda
    DEATH_PENALTY = -100.0          # additional penalty if done=True from death
    VALUE_LOSS_COEF = 0.5           # scale factor on value loss

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
            old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)

            # --- C) Step environment ---
            try:
                next_obs, env_reward, done, info = env.step(minerl_action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            # If 'done' due to death (or forced end), apply penalty
            if done:
                # You might customize whether it's actually "death" or some error
                # For simplicity, assume done = death => negative penalty
                env_reward += DEATH_PENALTY

            # --- D) Compute custom reward & accumulate ---
            custom_r, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            reward = env_reward + custom_r  # Combine env reward with custom
            cumulative_reward += reward

            # --- E) Store transition ---
            # We'll also store 'done' and 'next_obs' for GAE
            transitions.append({
                "obs": obs,
                "next_obs": next_obs,
                "pi_dist": pi_dist,
                "v_pred": v_pred,        # agent's value estimate
                "log_prob": log_prob,    # agent's log-prob
                "old_pi_dist": old_pi_dist,
                "reward": reward,
                "done": done
            })

            obs = next_obs
            step_count += 1

        print(f"  Collected {len(transitions)} steps this iteration. Done={done}, "
              f"CumulativeReward={cumulative_reward}")

        if done:
            # If the episode ended, reset for the next iteration
            obs = env.reset()
            visited_chunks.clear()

        if len(transitions) == 0:
            print("  No transitions collected, continuing.")
            continue

        # --- F) Compute GAE for partial rollout ---
        transitions = compute_gae(transitions, agent, gamma=GAMMA, lam=LAM)

        # --- G) Single gradient update for all transitions ---
        optimizer.zero_grad()
        total_loss_for_rollout = th.tensor(0.0, device="cuda")

        for step_data in transitions:
            advantage = step_data["advantage"]            # (float)
            returns_ = step_data["return"]                # (float)
            log_prob = step_data["log_prob"]              # (tensor)
            pi_dist_current = step_data["pi_dist"]
            pi_dist_pretrained = step_data["old_pi_dist"]

            # Policy gradient loss: - advantage * log_prob
            loss_rl = -(advantage * log_prob)

            # Value loss: (V - returns)^2
            # If we want to update the value function, do NOT detach step_data["v_pred"]
            # from the rollout. So let's do:
            v_pred_ = step_data["v_pred"]  # the raw tensor from environment step
            value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2

            # KL regularization
            loss_kl = compute_kl_loss(pi_dist_current, pi_dist_pretrained)

            total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * loss_kl)
            total_loss_for_rollout += total_loss_step.mean()

        # Average over the partial rollout
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be fine-tuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be fine-tuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where fine-tuned weights will be saved")
    parser.add_argument("--num-iterations", required=False, type=int, default=10,
                        help="Number of partial-rollout iterations")
    parser.add_argument("--rollout-steps", required=False, type=int, default=40,
                        help="How many steps per partial rollout")

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps
    )
