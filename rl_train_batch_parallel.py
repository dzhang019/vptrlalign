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
    Same GAE function. We treat the entire 'transitions' list
    as if it's one sequence. For 2 envs, we just store them
    in the order they were collected, then do the naive approach.
    (For perfect correctness, you'd separate each env's partial
    trajectory, do GAE individually.)
    """
    if len(transitions) == 0:
        return transitions

    # 1) If the last step isn't done, bootstrap final value
    if not transitions[-1]["done"]:
        with th.no_grad():
            final_next_obs = transitions[-1]["next_obs"]
            _, _, v_next, _, _ = agent.get_action_and_training_info(final_next_obs, stochastic=False)
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0

    # 2) GAE
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
    rollout_steps=40
):
    """
    Minimal changes to run exactly TWO parallel environments in lockstep.
    You still get a partial rollout of 'rollout_steps' steps,
    but now from 2 envs => up to 2*rollout_steps transitions per iteration.
    We keep env.render() calls so you can see both windows.
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

    # ===== 1) Initialize agent & pretrained policy (single env for shape references) =====
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

    # ===== 2) Create two envs =====
    env1 = HumanSurvival(**ENV_KWARGS).make()
    env2 = HumanSurvival(**ENV_KWARGS).make()

    obs1 = env1.reset()
    obs2 = env2.reset()

    done1 = False
    done2 = False
    visited_chunks1 = set()
    visited_chunks2 = set()
    episode_step_count1 = 0
    episode_step_count2 = 0

    # 3) Create optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    # Some stats
    running_loss = 0.0
    total_steps = 0

    # ===== 4) Outer loop over partial-rollout iterations =====
    for iteration in range(num_iterations):
        print(f"Starting partial-rollout iteration {iteration}")

        transitions_all = []  # We'll store transitions from both envs

        # We'll do 'rollout_steps' steps in lockstep
        step_count = 0
        while step_count < rollout_steps:
            step_count += 1

            # A) Render each env so you can see both windows
            env1.render()
            env2.render()

            # B) For env1 if not done => forward pass => step => store transition
            if not done1:
                episode_step_count1 += 1

                # get agent's action
                minerl_action1, pi_dist1, v_pred1, log_prob1, _ = \
                    agent.get_action_and_training_info(obs1, stochastic=True)

                # get pretrained policy dist for KL
                with th.no_grad():
                    obs_for_pretrained1 = agent._env_obs_to_agent(obs1)
                    obs_for_pretrained1 = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained1)
                    (old_pi_dist1, _, _), _ = pretrained_policy.policy(
                        obs=obs_for_pretrained1,
                        state_in=pretrained_policy.policy.initial_state(1),
                        first=th.tensor([[False]], dtype=th.bool, device="cuda")
                    )
                old_pi_dist1 = tree_map(lambda x: x.detach(), old_pi_dist1)

                try:
                    next_obs1, env_reward1, done1_flag, info1 = env1.step(minerl_action1)
                except Exception as e:
                    print(f"Env1 step error: {e}")
                    done1_flag = True
                    env_reward1 = 0
                    info1 = {}

                if 'error' in info1:
                    print(f"[Env1] Error in info: {info1['error']}. Ending episode.")
                    done1_flag = True

                if done1_flag:
                    env_reward1 += DEATH_PENALTY

                custom_r1, visited_chunks1 = custom_reward_function(obs1, done1_flag, info1, visited_chunks1)
                reward1 = env_reward1 + custom_r1

                transitions_all.append({
                    "obs": obs1,
                    "next_obs": next_obs1,
                    "pi_dist": pi_dist1,
                    "v_pred": v_pred1,
                    "log_prob": log_prob1,
                    "old_pi_dist": old_pi_dist1,
                    "reward": reward1,
                    "done": done1_flag
                })

                obs1 = next_obs1
                done1 = done1_flag

                if done1:
                    # Log the length
                    with open(out_episodes, "a") as f:
                        f.write(f"{episode_step_count1}\n")
                    episode_step_count1 = 0
                    obs1 = env1.reset()
                    visited_chunks1.clear()
                    done1 = False

            # C) For env2 if not done => same logic
            if not done2:
                episode_step_count2 += 1

                minerl_action2, pi_dist2, v_pred2, log_prob2, _ = \
                    agent.get_action_and_training_info(obs2, stochastic=True)

                with th.no_grad():
                    obs_for_pretrained2 = agent._env_obs_to_agent(obs2)
                    obs_for_pretrained2 = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained2)
                    (old_pi_dist2, _, _), _ = pretrained_policy.policy(
                        obs=obs_for_pretrained2,
                        state_in=pretrained_policy.policy.initial_state(1),
                        first=th.tensor([[False]], dtype=th.bool, device="cuda")
                    )
                old_pi_dist2 = tree_map(lambda x: x.detach(), old_pi_dist2)

                try:
                    next_obs2, env_reward2, done2_flag, info2 = env2.step(minerl_action2)
                except Exception as e:
                    print(f"Env2 step error: {e}")
                    done2_flag = True
                    env_reward2 = 0
                    info2 = {}

                if 'error' in info2:
                    print(f"[Env2] Error in info: {info2['error']}. Ending episode.")
                    done2_flag = True

                if done2_flag:
                    env_reward2 += DEATH_PENALTY

                custom_r2, visited_chunks2 = custom_reward_function(obs2, done2_flag, info2, visited_chunks2)
                reward2 = env_reward2 + custom_r2

                transitions_all.append({
                    "obs": obs2,
                    "next_obs": next_obs2,
                    "pi_dist": pi_dist2,
                    "v_pred": v_pred2,
                    "log_prob": log_prob2,
                    "old_pi_dist": old_pi_dist2,
                    "reward": reward2,
                    "done": done2_flag
                })

                obs2 = next_obs2
                done2 = done2_flag

                if done2:
                    with open(out_episodes, "a") as f:
                        f.write(f"{episode_step_count2}\n")
                    episode_step_count2 = 0
                    obs2 = env2.reset()
                    visited_chunks2.clear()
                    done2 = False

        # => done collecting up to 'rollout_steps' from each env
        # We can do a single update now

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

    # 5) Save the final weights
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

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps
    )
