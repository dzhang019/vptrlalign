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

th.autograd.set_detect_anomaly(True)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


# ... (previous imports and code)

def train_rl(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2
):
    # ... (existing setup code)

    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Collecting up to {rollout_steps} steps per env...")

        # ... (existing code)

        # ==== 4) Environment stepping phase ====
        step_count = 0
        while step_count < rollout_steps:
            step_count += 1
            for env_i in range(num_envs):
                envs[env_i].render()

            for env_i in range(num_envs):
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1

                    # (A) Sample an action and get the new hidden state WITH NO GRAD
                    with th.no_grad():  # ADDED: Prevent computation graph during rollout
                        minerl_action_i, _, _, _, new_hid_i = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_i],
                            hidden_state=hidden_states[env_i],
                            stochastic=True,
                            taken_action=None  # sample from policy
                        )

                    # (B) Step the environment
                    next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(minerl_action_i)
                    if "error" in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True

                    if done_flag_i:
                        env_reward_i += DEATH_PENALTY

                    # (C) Store minimal data + hidden state (DETACHED)
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(env_reward_i)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    # Detach hidden state before storing
                    rollouts[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach(), hidden_states[env_i])  # ADDED: Detach hidden state
                    )
                    rollouts[env_i]["next_obs"].append(next_obs_i)

                    # (D) Update obs, hidden_state, done
                    obs_list[env_i] = next_obs_i
                    hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid_i)  # Ensure new_hid is detached
                    done_list[env_i] = done_flag_i

                    if done_flag_i:
                        # ... (existing reset code)

        # ... (rest of the training loop remains the same)
    """
    Approach #2, memory-correct:
      - We store only (obs, action, reward, done, hidden_state_before_step, next_obs).
      - No 'v_pred'/'log_prob' stored at rollout time. 
      - Then do a stepwise unroll at training time to recover them.
    
    In each iteration:
      1) Collect up to `rollout_steps` steps from each of `num_envs`.
      2) Unroll each env's partial trajectory with the same hidden states + forced actions.
      3) Do GAE, RL update (with KL to pretrained if desired).
    """

    # ==== Hyperparams ====
    LEARNING_RATE = 1e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 1.0
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

    # Each env has its own hidden state
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]

    # We'll store partial rollouts here
    rollouts = [
        {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hidden_states": [],
            "next_obs": []
        }
        for _ in range(num_envs)
    ]

    # ==== 3) Optimizer, stats ====
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0

    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Collecting up to {rollout_steps} steps per env...")

        # Clear out old partial rollouts
        for env_i in range(num_envs):
            rollouts[env_i]["obs"].clear()
            rollouts[env_i]["actions"].clear()
            rollouts[env_i]["rewards"].clear()
            rollouts[env_i]["dones"].clear()
            rollouts[env_i]["hidden_states"].clear()
            rollouts[env_i]["next_obs"].clear()

        # ==== 4) Environment stepping phase ====
        step_count = 0
        while step_count < rollout_steps:
            step_count += 1
            for env_i in range(num_envs):
                envs[env_i].render()

            for env_i in range(num_envs):
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1

                    # (A) Sample an action and get the new hidden state
                    minerl_action_i, _, _, _, new_hid_i = agent.get_action_and_training_info(
                        minerl_obs=obs_list[env_i],
                        hidden_state=hidden_states[env_i],
                        stochastic=True,
                        taken_action=None  # sample from policy
                    )

                    # (B) Step the environment
                    next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(minerl_action_i)
                    if "error" in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True

                    if done_flag_i:
                        env_reward_i += DEATH_PENALTY

                    # (C) Store minimal data + hidden state
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(env_reward_i)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    rollouts[env_i]["hidden_states"].append(hidden_states[env_i])
                    rollouts[env_i]["next_obs"].append(next_obs_i)

                    # (D) Update obs, hidden_state, done
                    obs_list[env_i] = next_obs_i
                    hidden_states[env_i] = new_hid_i
                    done_list[env_i] = done_flag_i

                    if done_flag_i:
                        # log steps in out_episodes
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_counts[env_i]}\n")
                        episode_step_counts[env_i] = 0
                        # reset
                        obs_list[env_i] = envs[env_i].reset()
                        done_list[env_i] = False
                        hidden_states[env_i] = agent.policy.initial_state(batch_size=1)

        # ==== 5) Training unroll: re-play each env's partial trajectory ====
        print(f"[Iteration {iteration}] Doing training unroll & RL update...")
        transitions_all = []
        for env_i in range(num_envs):
            env_transitions = train_unroll(
                agent,
                pretrained_policy,
                rollouts[env_i],
                gamma=GAMMA,
                lam=LAM
            )
            transitions_all.extend(env_transitions)

        # If no transitions, skip the update.
        if len(transitions_all) == 0:
            print(f"[Iteration {iteration}] No transitions collected, skipping update.")
            continue

        # ==== 6) RL update with the newly unrolled transitions ====
        optimizer.zero_grad()
        loss_list = []
        for t in transitions_all:
            advantage = t["advantage"]
            returns_ = t["return"]
            log_prob = t["log_prob"]
            v_pred_ = t["v_pred"]
            cur_pd = t["cur_pd"]
            old_pd = t["old_pd"]  # distribution from pretrained policy

            # RL Loss
            loss_rl = -(advantage * log_prob)

            # Value loss (using an out-of-place operation)
            value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2

            # KL regularization loss
            kl_loss = compute_kl_loss(cur_pd, old_pd)

            total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
            loss_list.append(total_loss_step.mean())

        total_loss_for_rollout = sum(loss_list) / len(loss_list)
        total_loss_for_rollout.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss_val = total_loss_for_rollout.item()
        running_loss += total_loss_val * len(transitions_all)
        total_steps += len(transitions_all)
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        LAMBDA_KL *= KL_DECAY

        print(f"[Iteration {iteration}] Loss={total_loss_val:.4f}, StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}")

    # Final saving
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """
    Replays the partial trajectory for one environment.
    For each step t:
      1) We have obs[t], hidden_states[t], action[t].
      2) We do agent.get_action_and_training_info(obs[t], hidden_states[t], taken_action=action[t])
         to get the correct log_prob & v_pred for that forced action in the correct memory state.
      3) We also gather the pretrained distribution for KL.
    Then we do a local GAE and return transitions with 'log_prob', 'v_pred', 'cur_pd', 'old_pd', etc.
    """

    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions

    # Stepwise re-forward pass
    for t in range(T):
        obs_t = rollout["obs"][t]
        act_t = rollout["actions"][t]
        rew_t = rollout["rewards"][t]
        done_t = rollout["dones"][t]
        hid_t = rollout["hidden_states"][t]
        next_obs_t = rollout["next_obs"][t]

        # (A) Force the same action to get log_prob, v_pred, and the current distribution.
        minerl_action, pi_dist, v_pred, log_prob, hid_out = agent.get_action_and_training_info(
            minerl_obs=obs_t,
            hidden_state=hid_t,
            stochastic=False,       # deterministic: forced action
            taken_action=act_t      # the action that was taken
        )

        # (B) Get the pretrained distribution for KL regularization.
        with th.no_grad():
            old_minerl_action, old_pd, old_vpred, old_logprob, _ = pretrained_policy.get_action_and_training_info(
                obs_t, 
                pretrained_policy.policy.initial_state(1),
                stochastic=False,
                taken_action=act_t
            )

        transitions.append({
            "obs": obs_t,
            "action": act_t,
            "reward": rew_t,
            "done": done_t,
            "v_pred": v_pred,
            "log_prob": log_prob,
            "cur_pd": pi_dist,
            "old_pd": old_pd,
            "next_obs": next_obs_t
        })

    # (C) Local GAE computation.
    if not transitions[-1]["done"]:
        with th.no_grad():
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=rollout["hidden_states"][-1],
                stochastic=False,
                taken_action=None
            )
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0

    gae = 0.0
    for i in reversed(range(T)):
        r_i = transitions[i]["reward"]
        v_i = transitions[i]["v_pred"].item()
        done_i = transitions[i]["done"]
        mask = 1.0 - float(done_i)
        if i == T - 1:
            next_val = bootstrap_value
        else:
            next_val = transitions[i+1]["v_pred"].item()

        delta = r_i + gamma * next_val * mask - v_i
        gae = delta + gamma * lam * mask * gae
        transitions[i]["advantage"] = gae
        transitions[i]["return"] = v_i + gae

    return transitions


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
