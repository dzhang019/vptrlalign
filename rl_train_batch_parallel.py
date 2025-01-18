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
    Naive GAE over one big list of transitions (from all envs).
    For full correctness, you'd separate transitions per env or per done boundary,
    but this minimal approach often works decently for short partial rollouts.
    """
    if len(transitions) == 0:
        return transitions

    # If the last transition is not done, bootstrap
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
            next_value = transitions[i + 1]["v_pred"].item()

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
    Runs `num_envs` parallel MineRL environments in a single process.
    Each step, we do a forward pass for each env individually (not true batched),
    and store transitions. After `rollout_steps`, do 1 gradient update.
    """

    # ========== Hyperparams ==========
    LEARNING_RATE = 1e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 1.0
    GAMMA = 0.9999
    LAM = 0.95
    DEATH_PENALTY = -1000.0
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995

    # ========== 1) Agent + Pretrained Policy ==========
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

    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert agent_param.data_ptr() != pretrained_param.data_ptr(), "Weights are shared!"

    # ========== 2) Create N environments + track their states ==========
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    visited_chunks_list = [set() for _ in range(num_envs)]
    episode_step_counts = [0] * num_envs

    # ========== 3) Create Optimizer ==========
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    total_steps = 0

    # ========== 4) Outer loop over partial-rollout iterations ==========
    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Starting partial-rollout...")

        # We'll store transitions from *all* envs in one big list
        transitions_all = []
        step_count = 0

        # We'll collect up to `rollout_steps` steps in lockstep
        while step_count < rollout_steps:
            step_count += 1

            # A) Optional: Render each environment (this tries to open multiple windows).
            for env_i in range(num_envs):
                envs[env_i].render()

            # B) Step each environment that isn't done
            for env_i in range(num_envs):
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1

                    # --- forward pass for a single env obs
                    action_i, pi_dist_i, v_pred_i, log_prob_i, _ = \
                        agent.get_action_and_training_info(obs_list[env_i], stochastic=True)

                    # --- get pretrained policy dist for KL
                    with th.no_grad():
                        obs_for_pretrained = agent._env_obs_to_agent(obs_list[env_i])
                        obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                        (old_pi_dist_i, _, _), _ = pretrained_policy.policy(
                            obs=obs_for_pretrained,
                            state_in=pretrained_policy.policy.initial_state(1),
                            first=th.tensor([[False]], dtype=th.bool, device="cuda")
                        )
                    old_pi_dist_i = tree_map(lambda x: x.detach(), old_pi_dist_i)

                    # --- env step
                    try:
                        next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(action_i)
                    except Exception as e:
                        print(f"[Env {env_i}] step error: {e}")
                        done_flag_i = True
                        env_reward_i = 0
                        info_i = {}

                    if 'error' in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}. Ending episode.")
                        done_flag_i = True

                    if done_flag_i:
                        env_reward_i += DEATH_PENALTY

                    # custom reward
                    custom_r_i, visited_chunks_list[env_i] = custom_reward_function(
                        obs_list[env_i], done_flag_i, info_i, visited_chunks_list[env_i]
                    )
                    reward_i = env_reward_i + custom_r_i

                    # store transition
                    transitions_all.append({
                        "obs": obs_list[env_i],
                        "next_obs": next_obs_i,
                        "pi_dist": pi_dist_i,
                        "v_pred": v_pred_i,
                        "log_prob": log_prob_i,
                        "old_pi_dist": old_pi_dist_i,
                        "reward": reward_i,
                        "done": done_flag_i
                    })

                    # update obs, done
                    obs_list[env_i] = next_obs_i
                    done_list[env_i] = done_flag_i

                    if done_flag_i:
                        # log episode length
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_counts[env_i]}\n")
                        episode_step_counts[env_i] = 0
                        obs_list[env_i] = envs[env_i].reset()
                        visited_chunks_list[env_i].clear()
                        done_list[env_i] = False

        # => done collecting up to rollout_steps from each env
        # => do a single update now
        if len(transitions_all) == 0:
            print("No transitions collected, continuing.")
            continue

        # F) GAE
        transitions_all = compute_gae(transitions_all, agent, gamma=GAMMA, lam=LAM)

        # G) Single gradient update
        optimizer.zero_grad()
        total_loss_for_rollout = th.tensor(0.0, device="cuda")

        for tdata in transitions_all:
            advantage = tdata["advantage"]
            returns_ = tdata["return"]
            log_prob = tdata["log_prob"]
            pi_dist_current = tdata["pi_dist"]
            pi_dist_pretrained = tdata["old_pi_dist"]

            # RL
            loss_rl = -(advantage * log_prob)
            # Value
            v_pred_ = tdata["v_pred"]
            value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2
            # KL
            loss_kl = compute_kl_loss(pi_dist_current, pi_dist_pretrained)

            total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * loss_kl)
            total_loss_for_rollout += total_loss_step.mean()

        total_loss_for_rollout = total_loss_for_rollout / len(transitions_all)
        total_loss_for_rollout.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss_val = total_loss_for_rollout.item()
        running_loss += total_loss_val * len(transitions_all)
        total_steps += len(transitions_all)
        if total_steps > 0:
            avg_loss = running_loss / total_steps
        else:
            avg_loss = 0.0
        LAMBDA_KL *= KL_DECAY

        print(f"[Iteration {iteration}] Loss={total_loss_val:.4f}, StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}")

    # ========== 5) Save the final weights ==========
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
                        help="Number of parallel environments to run")

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
