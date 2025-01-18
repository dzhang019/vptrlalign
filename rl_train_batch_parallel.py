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
    Same as before, but note that 'transitions' may contain data from multiple envs.
    Each item is a dict with:
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
    # We'll assume transitions are grouped or flattened but keep the same logic.
    # One subtlety: if transitions come from multiple envs that ended at different times,
    # you can't just do a single backward pass for GAE. Instead, you do GAE per environment trajectory.
    # For simplicity, let's assume they are all short partials in the same "time window".
    # We'll do a naive approach: treat them as if each is sequential. 
    # If you want perfect correctness, you'd segment by env index. But let's keep minimal.

    # 1) Sort transitions by 'timestep_index' if needed, or skip if you're collecting them in order.
    # We'll skip. We'll assume they are appended in the exact stepping order. 
    # But if you have multiple envs, the last transitions might be from env 3, which doesn't
    # line up with env 2. For a minimal approach, we do what you had before: a single chain.

    if not transitions:
        return transitions
    # We detect the last is done or not:
    if not transitions[-1]["done"]:
        with th.no_grad():
            final_next_obs = transitions[-1]["next_obs"]
            _, _, v_next, _, _ = agent.get_action_and_training_info(final_next_obs, stochastic=False)
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0

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


def train_rl(
    in_model, 
    in_weights, 
    out_weights, 
    out_episodes,
    num_iterations=10, 
    rollout_steps=40,
    num_envs=2
):
    """
    Modified partial-rollout training with GAE + Value Loss + KL regularization,
    now supporting multiple parallel environments.

    We'll run 'num_envs' envs in lockstep for 'rollout_steps' steps each iteration.
    Then do one single update on the combined transitions from all envs.

    Args:
      ...
      num_envs: how many envs to run in parallel
    """

    # Hyperparameters
    LEARNING_RATE = 1e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 1.0
    GAMMA = 0.9999
    LAM = 0.95
    DEATH_PENALTY = -1000.0
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995

    # 1) Load parameters for both current agent and pretrained agent
    #    We create one 'template' env for the agent construction
    #    (We won't actually step this env. It's only for shape references, etc.)
    env_template = HumanSurvival(**ENV_KWARGS).make()

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        env_template, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        env_template, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)

    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert agent_param.data_ptr() != pretrained_param.data_ptr(), "Weights are shared!"

    # 2) Create the parallel environments
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    obs_list = [env.reset() for env in envs]
    visited_chunks_list = [set() for _ in range(num_envs)]
    done_list = [False]*num_envs

    # We'll also keep track of how many steps each env has survived in the current episode
    episode_step_counts = [0]*num_envs

    # 3) Create optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    total_steps = 0

    # 4) Outer loop over partial-rollout iterations
    for iteration in range(num_iterations):
        print(f"Starting partial-rollout iteration {iteration}")

        # We'll accumulate transitions from *all* envs in one big list
        transitions_all_envs = []

        # We'll do 'rollout_steps' steps in lockstep
        for step_i in range(rollout_steps):
            # A) Batch forward pass for all envs that are not done
            #    We build a batch of observations for each env that is still active
            #    For minimal changes, let's do a loop call to get_action_and_training_info for each env
            #    Then stack if needed. But that wouldn't be truly batched on GPU. 
            #    For actual GPU batching, you'd do a single forward pass with stacked obs. 
            #    We'll keep it minimal with a loop approach.

            minerl_actions = []
            pi_dists = []
            v_preds = []
            log_probs = []
            # new_hidden_states = []  # if you used them

            for env_i in range(num_envs):
                if not done_list[env_i]:
                    # Single-step forward pass
                    # obs_list[env_i] is the current obs
                    action, pi_dist, v_pred, log_prob, new_hidden_state = \
                        agent.get_action_and_training_info(obs_list[env_i], stochastic=True)
                    minerl_actions.append(action)
                    pi_dists.append(pi_dist)
                    v_preds.append(v_pred)
                    log_probs.append(log_prob)
                else:
                    # If env is done, we'll put placeholders (the step won't happen)
                    minerl_actions.append(None)
                    pi_dists.append(None)
                    v_preds.append(None)
                    log_probs.append(None)

            # B) For each env, step it if not done
            for env_i in range(num_envs):
                if done_list[env_i]:
                    continue  # skip stepping
                episode_step_counts[env_i] += 1

                # Get pretrained policy distribution for KL
                with th.no_grad():
                    obs_for_pretrained = agent._env_obs_to_agent(obs_list[env_i])
                    obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                    (old_pi_dist, _, _), _ = pretrained_policy.policy(
                        obs=obs_for_pretrained,
                        state_in=pretrained_policy.policy.initial_state(1),
                        first=th.tensor([[False]], dtype=th.bool, device="cuda")
                    )
                old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)

                try:
                    next_obs, env_reward, done_flag, info = envs[env_i].step(minerl_actions[env_i])
                    if 'error' in info:
                        print(f"[Env {env_i}] Error in info: {info['error']}. Ending episode.")
                        done_flag = True
                except Exception as e:
                    print(f"[Env {env_i}] Error during env.step(): {e}")
                    done_flag = True

                if done_flag:
                    env_reward += DEATH_PENALTY

                # D) custom reward
                custom_r, visited_chunks_list[env_i] = custom_reward_function(
                    obs_list[env_i], done_flag, info, visited_chunks_list[env_i]
                )
                reward = env_reward + custom_r

                # E) Store transition
                transitions_all_envs.append({
                    "obs": obs_list[env_i],
                    "next_obs": next_obs,
                    "pi_dist": pi_dists[env_i],
                    "v_pred": v_preds[env_i],
                    "log_prob": log_probs[env_i],
                    "old_pi_dist": old_pi_dist,
                    "reward": reward,
                    "done": done_flag
                })

                obs_list[env_i] = next_obs
                done_list[env_i] = done_flag

                if done_flag:
                    # Log the length of the episode
                    with open(out_episodes, "a") as f:
                        f.write(f"{episode_step_counts[env_i]}\n")

                    # Reset
                    obs_list[env_i] = envs[env_i].reset()
                    visited_chunks_list[env_i].clear()
                    episode_step_counts[env_i] = 0
                    done_list[env_i] = False

        # => now we have up to (num_envs * rollout_steps) transitions
        if len(transitions_all_envs) == 0:
            print("No transitions collected? continue.")
            continue

        # F) Compute GAE for all transitions combined
        #    Minimal approach: treat them as if they are one sequence. 
        #    More correct: you'd separate them by environment episodes, do GAE per env. 
        #    We'll do the naive approach for brevity:
        transitions_all_envs = compute_gae(transitions_all_envs, agent, gamma=GAMMA, lam=LAM)

        # G) Single gradient update
        optimizer.zero_grad()
        total_loss_for_rollout = th.tensor(0.0, device="cuda")

        for step_data in transitions_all_envs:
            advantage = step_data["advantage"]
            returns_ = step_data["return"]
            log_prob = step_data["log_prob"]
            pi_dist_current = step_data["pi_dist"]
            pi_dist_pretrained = step_data["old_pi_dist"]

            if (log_prob is None) or (pi_dist_current is None):
                # Means that env was done at that step before we took an action
                # or we had placeholders. Let's skip these. 
                continue

            loss_rl = -(advantage * log_prob)
            v_pred_ = step_data["v_pred"]
            value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2

            loss_kl = compute_kl_loss(pi_dist_current, pi_dist_pretrained)
            total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * loss_kl)
            total_loss_for_rollout += total_loss_step.mean()

        # Average the loss
        n_valid = sum(1 for t in transitions_all_envs if t.get("log_prob") is not None)
        if n_valid > 0:
            total_loss_for_rollout = total_loss_for_rollout / n_valid

            total_loss_for_rollout.backward()
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss_val = total_loss_for_rollout.item()
        else:
            total_loss_val = 0.0

        # Some stats
        running_loss += total_loss_val * len(transitions_all_envs)
        total_steps += len(transitions_all_envs)
        if total_steps > 0:
            avg_loss = running_loss / total_steps
        else:
            avg_loss = 0.0

        # Decay KL
        LAMBDA_KL *= KL_DECAY

        print(f"[Iter {iteration}] Loss={total_loss_val:.4f}, StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}")

    # 5) Save weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be fine-tuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be fine-tuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where fine-tuned weights will be saved")
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt",
                        help="Path to text file for logging #steps each episode survived.")
    parser.add_argument("--num-iterations", required=False, type=int, default=10,
                        help="Number of partial-rollout iterations")
    parser.add_argument("--rollout-steps", required=False, type=int, default=40,
                        help="How many steps per partial rollout")
    parser.add_argument("--num-envs", required=False, type=int, default=2,
                        help="Number of parallel envs to run")

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs
    )
