import torch as th
import torch.nn.functional as F
from torch import nn
import numpy as np
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


EPOCHS = 2
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0
def train_rl_ppo_style(in_model, in_weights, out_weights, 
                       num_iterations=1000,     # How many total rollout-updates
                       rollout_steps=40,        # Steps per partial rollout
                       ppo_epochs=4,            # How many epochs per rollout
                       gamma=0.999,             # Discount factor
                       lam=0.95,                # GAE lambda
                       kl_coef=0.01,            # Pretrained policy KL coefficient (rho)
                       clip_range=0.2,
                       learning_rate=2e-5):

    # 1) Initialize environment and agent
    env = HumanSurvival(**ENV_KWARGS).make()

    # Create agent (trainable) and pretrained policy (frozen)
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(env, device="cuda",
                        policy_kwargs=agent_policy_kwargs,
                        pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)  # Load pretrained but we will FINETUNE

    pretrained_policy = MineRLAgent(env, device="cuda",
                                    policy_kwargs=agent_policy_kwargs,
                                    pi_head_kwargs=agent_pi_head_kwargs)
    pretrained_policy.load_weights(in_weights)  # Frozen reference

    # Make sure weights aren't shared
    for param_a, param_b in zip(agent.policy.parameters(),
                                pretrained_policy.policy.parameters()):
        assert param_a.data_ptr() != param_b.data_ptr(), "Weights are shared!"

    # Set pretrained policy to eval/frozen
    pretrained_policy.policy.eval()
    for p in pretrained_policy.policy.parameters():
        p.requires_grad = False

    # PPO optimizer for agent
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=learning_rate)

    # 2) Main training loop (each iteration = gather partial rollout, then PPO update)
    global_step = 0
    for iteration in range(num_iterations):
        # 2a) Gather partial rollout
        obs_list, actions_list, rewards_list, dones_list = [], [], [], []
        logprobs_list, values_list = [], []
        old_pi_dists = []
        
        obs = env.reset() if iteration == 0 else obs  # If continuing from last step is complex, 
                                                      # you might always reset or track done states.
        
        # You might keep track of hidden states if your policy is recurrent.

        for step_i in range(rollout_steps):
            global_step += 1

            # Compute action + training info from current agent
            # "get_action_and_training_info" presumably returns (action, pi_dist, v_pred, log_prob, new_hid)
            action, pi_dist, value, log_prob, new_hidden_state = agent.get_action_and_training_info(
                obs, stochastic=True
            )

            # Freeze pretrained policy distribution for KL
            with th.no_grad():
                obs_for_pretrained = agent._env_obs_to_agent(obs)
                obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                (pt_pi_dist, _, _), _ = pretrained_policy.policy(
                    obs=obs_for_pretrained,
                    state_in=pretrained_policy.policy.initial_state(1),
                    first=th.tensor([[False]], dtype=th.bool, device="cuda")
                )
                # detach
                pt_pi_dist = tree_map(lambda x: x.detach(), pt_pi_dist)

            # Step environment
            next_obs, env_reward, done, info = env.step(action)

            # Possibly compute custom reward or combine with env_reward
            reward, _ = custom_reward_function(obs, done, info, visited_chunks=set())

            # Store transition
            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            logprobs_list.append(log_prob)
            values_list.append(value)
            old_pi_dists.append(pt_pi_dist)

            obs = next_obs
            if done:
                # If the episode ends before we reach rollout_steps, we can break
                # (or optionally collect from a new env.reset())
                break

        # 2b) Compute advantages & returns via GAE
        # If we didn't reach a 'done', we bootstrap final value from agent
        if not done:
            with th.no_grad():
                _, _, next_value, _, _ = agent.get_action_and_training_info(obs, stochastic=False)
            next_value = next_value.item()
        else:
            next_value = 0.0

        advantages, returns = compute_gae(
            rewards_list, values_list, dones_list, next_value,
            gamma=gamma, lam=lam
        )

        # 2c) Convert rollout buffers to Tensors
        obs_tensor = convert_obs_list_to_tensor(obs_list, agent)  # your function
        actions_tensor = convert_actions_list_to_tensor(actions_list)  # your function
        logprobs_tensor = th.stack(logprobs_list)
        values_tensor = th.stack(values_list)
        advantages_tensor = th.tensor(advantages, dtype=th.float32, device="cuda")
        returns_tensor = th.tensor(returns, dtype=th.float32, device="cuda")

        # We also keep the old pretrained distributions around for the KL penalty
        # "old_pi_dists" is a list of dicts. For PPO ratio, we also want "old_logprobs" of the agent or
        # we can recalc agent's logprob if we stored entire distribution. Typically we store the "old_action_logprob".
        
        # 2d) PPO multiple epochs update
        # Simple version: do one pass. Typical PPO does ~3-10 passes with mini-batches
        # We'll do a single pass for brevity:
        for epoch_i in range(ppo_epochs):
            # Recompute the current policy distribution for each step (or store it from the rollout).
            # Usually we do a fresh forward pass to get the new log_probs & new values:
            current_pi_dist, current_values = forward_policy_batch(agent, obs_tensor)

            # ratio = exp(new_logprob - old_logprob). We have "logprobs_tensor" from rollout.
            new_logprobs = get_logprob_of_actions(current_pi_dist, actions_tensor)
            ratio = (new_logprobs - logprobs_tensor).exp()

            # clipped surrogate
            surr1 = ratio * advantages_tensor
            surr2 = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor
            pg_loss = -th.min(surr1, surr2).mean()

            # value loss
            new_values = current_values.squeeze()
            vf_loss = F.mse_loss(new_values, returns_tensor)

            # KL with the *pretrained policy*
            # We have pt_pi_dist from earlier. We want to measure KL( new_pi_dist || old_pt_pi_dist ).
            kl_loss = compute_kl_loss_batch(current_pi_dist, old_pi_dists)

            # total loss
            total_loss = pg_loss + 0.5 * vf_loss + kl_coef * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=5.0)
            optimizer.step()

        # Possibly decay KL coefficient over iterations, like in the VPT paper
        # kl_coef *= 0.9995  # example

        print(f"[Iter {iteration}] Steps collected={len(rewards_list)}, Reward sum={sum(rewards_list)}, Loss={total_loss.item():.3f}")

    # Finally, save updated weights
    th.save(agent.policy.state_dict(), out_weights)

# ------------------------------------------------------
# Example GAE helper:
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    rewards: list of float
    values: list of tensors
    dones: list of bool
    next_value: float (or tensor) for bootstrap
    Returns (advantages, returns) as python lists or np arrays.
    """
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step].item()
        gae = delta + gamma * lam * mask * gae
        advantages.append(gae)
        next_value = values[step].item()
    advantages.reverse()

    # The "returns" are just value + advantage
    returns = [adv + val.item() for adv, val in zip(advantages, values)]
    return advantages, returns

# ------------------------------------------------------
# Example function to compute KL over entire batch
def compute_kl_loss_batch(current_pi_dist, old_pi_dist_list):
    """
    current_pi_dist: distribution from agent for [B] steps
    old_pi_dist_list: list of pretrained distribution dicts for each step
    You can do a distribution-specific KL. 
    For discrete: kl = sum(probs * log(probs/old_probs)).
    For continuous: different formula.
    Implementation depends on the shape of pi_dist. 
    """
    # Pseudocode. You need to unify "current_pi_dist" for the entire batch
    # with "old_pi_dist_list[i]" for each step i. 
    # One simple approach: loop. In real code you might do something vectorized.

    kl_sum = 0.0
    for i in range(len(old_pi_dist_list)):
        # current step distribution:
        # if current_pi_dist is a dict of [B] distributions, we take i-th item
        # old distribution is old_pi_dist_list[i]
        kl_i = compute_kl_loss(current_pi_dist[i], old_pi_dist_list[i])  
        kl_sum += kl_i
    return kl_sum / len(old_pi_dist_list)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be fine-tuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be fine-tuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where fine-tuned weights will be saved")
    parser.add_argument("--num-episodes", required=False, type=int, default=10, help="Number of training episodes")

    args = parser.parse_args()

    train_rl_ppo_style(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        num_episodes=args.num_episodes
    )
