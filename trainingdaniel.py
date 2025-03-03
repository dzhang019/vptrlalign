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

from lib.height import reward_function
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


def train_rl(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2,
    mini_batch_size=10  # Size of mini-batches for pipelined training
):
    """
    Modified version with pipelined training for better GPU utilization
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

    # ==== 4) Pipelined training loop ====
    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Starting with pipelined collection and training...")
        
        # Reset rollout buffers for this iteration
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
        
        # Keep track of accumulated transitions for mini-batch training
        accumulated_transitions = []
        mini_batch_updates = 0
        
        step_count = 0
        while step_count < rollout_steps:
            step_count += 1
            for env_i in range(num_envs):
                envs[env_i].render()

            # Environment stepping phase (with gradient prevention)
            for env_i in range(num_envs):
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1

                    # Prevent gradient tracking during rollout
                    with th.no_grad():
                        minerl_action_i, _, _, _, new_hid_i = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_i],
                            hidden_state=hidden_states[env_i],
                            stochastic=True,
                            taken_action=None
                        )

                    next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(minerl_action_i)
                    if "error" in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True

                    if done_flag_i:
                        env_reward_i += DEATH_PENALTY

                    # Store detached hidden states
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(env_reward_i)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    rollouts[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach(), hidden_states[env_i])
                    )
                    rollouts[env_i]["next_obs"].append(next_obs_i)

                    # Update with detached hidden state
                    obs_list[env_i] = next_obs_i
                    hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid_i)
                    done_list[env_i] = done_flag_i

                    if done_flag_i:
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_counts[env_i]}\n")
                        episode_step_counts[env_i] = 0
                        obs_list[env_i] = envs[env_i].reset()
                        done_list[env_i] = False
                        hidden_states[env_i] = agent.policy.initial_state(batch_size=1)
            
            # Periodically process transitions for mini-batch updates
            # Process transitions from recently collected steps
            for env_i in range(num_envs):
                # If we have at least 2 steps in this environment's buffer, 
                # we can process a transition (need current and next state)
                if len(rollouts[env_i]["obs"]) >= 2:
                    # Get the latest completed transition
                    latest_idx = len(rollouts[env_i]["obs"]) - 2
                    
                    # Extract the transition for training
                    env_transitions = process_transition(
                        agent,
                        pretrained_policy,
                        rollouts[env_i],
                        idx=latest_idx,
                        gamma=GAMMA,
                        lam=LAM
                    )
                    
                    if env_transitions:
                        accumulated_transitions.append(env_transitions)
            
            # Once we have enough transitions for a mini-batch, perform an update
            if len(accumulated_transitions) >= mini_batch_size:
                print(f"[Iteration {iteration}] Mini-batch update {mini_batch_updates+1} with {len(accumulated_transitions)} transitions")
                
                # Perform mini-batch update
                optimizer.zero_grad()
                loss_list = []
                
                for t in accumulated_transitions:
                    advantage = t["advantage"]
                    returns_ = t["return"]
                    log_prob = t["log_prob"]
                    v_pred_ = t["v_pred"]
                    cur_pd = t["cur_pd"]
                    old_pd = t["old_pd"]

                    loss_rl = -(advantage * log_prob)
                    value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2
                    kl_loss = compute_kl_loss(cur_pd, old_pd)

                    total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
                    loss_list.append(total_loss_step.mean())
                
                if loss_list:  # Ensure we have losses to backprop
                    total_loss_for_mini_batch = sum(loss_list) / len(loss_list)
                    total_loss_for_mini_batch.backward()
                    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    
                    total_loss_val = total_loss_for_mini_batch.item()
                    running_loss += total_loss_val * len(accumulated_transitions)
                    total_steps += len(accumulated_transitions)
                    
                    mini_batch_updates += 1
                    print(f"[Mini-batch {mini_batch_updates}] Loss={total_loss_val:.4f}, TotalSteps={total_steps}")
                
                # Clear accumulated transitions after update
                accumulated_transitions = []
        
        # Process any remaining transitions at the end of the iteration
        remaining_transitions = []
        for env_i in range(num_envs):
            if rollouts[env_i]["obs"]:
                env_transitions = train_unroll(
                    agent,
                    pretrained_policy,
                    rollouts[env_i],
                    gamma=GAMMA,
                    lam=LAM
                )
                remaining_transitions.extend(env_transitions)
        
        if remaining_transitions:
            print(f"[Iteration {iteration}] Final update with {len(remaining_transitions)} remaining transitions")
            
            optimizer.zero_grad()
            loss_list = []
            
            for t in remaining_transitions:
                advantage = t["advantage"]
                returns_ = t["return"]
                log_prob = t["log_prob"]
                v_pred_ = t["v_pred"]
                cur_pd = t["cur_pd"]
                old_pd = t["old_pd"]

                loss_rl = -(advantage * log_prob)
                value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2
                kl_loss = compute_kl_loss(cur_pd, old_pd)

                total_loss_step = loss_rl + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
                loss_list.append(total_loss_step.mean())
            
            if loss_list:
                total_loss_final = sum(loss_list) / len(loss_list)
                total_loss_final.backward()
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                total_loss_val = total_loss_final.item()
                running_loss += total_loss_val * len(remaining_transitions)
                total_steps += len(remaining_transitions)
        
        # Update KL coefficient
        LAMBDA_KL *= KL_DECAY
        
        # Log iteration summary
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        print(f"[Iteration {iteration}] Complete - MiniBatchUpdates={mini_batch_updates}, "
              f"TotalSteps={total_steps}, AvgLoss={avg_loss:.4f}, KL_Coeff={LAMBDA_KL:.4f}")

    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)


def process_transition(agent, pretrained_policy, rollout, idx, gamma=0.999, lam=0.95):
    """Process a single transition for training"""
    if idx >= len(rollout["obs"]) - 1:
        return None
    
    obs_t = rollout["obs"][idx]
    act_t = rollout["actions"][idx]
    rew_t = rollout["rewards"][idx]
    done_t = rollout["dones"][idx]
    hid_t = rollout["hidden_states"][idx]
    next_obs_t = rollout["next_obs"][idx]
    
    # Get value prediction for current observation
    minerl_action, pi_dist, v_pred, log_prob, _ = agent.get_action_and_training_info(
        minerl_obs=obs_t,
        hidden_state=hid_t,
        stochastic=False,
        taken_action=act_t
    )
    
    # Get value prediction for next observation to calculate advantage
    with th.no_grad():
        _, _, v_next, _, _ = agent.get_action_and_training_info(
            minerl_obs=next_obs_t,
            hidden_state=rollout["hidden_states"][idx+1],
            stochastic=False,
            taken_action=None
        )
        
        # Get policy distribution from pretrained model for KL calculation
        _, old_pd, _, _, _ = pretrained_policy.get_action_and_training_info(
            obs_t, 
            pretrained_policy.policy.initial_state(1),
            stochastic=False,
            taken_action=act_t
        )
    
    # Calculate advantage using TD error
    next_val = v_next.item() * (1.0 - float(done_t))  # Value is 0 if done
    delta = rew_t + gamma * next_val - v_pred.item()
    advantage = delta
    returns = v_pred.item() + advantage
    
    # Create transition with all required info for training
    transition = {
        "obs": obs_t,
        "action": act_t,
        "reward": rew_t,
        "done": done_t,
        "v_pred": v_pred,
        "log_prob": log_prob,
        "cur_pd": pi_dist,
        "old_pd": old_pd,
        "advantage": advantage,
        "return": returns
    }
    
    return transition


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """Process full trajectory for GAE calculation and returns transitions for training"""
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions

    for t in range(T):
        obs_t = rollout["obs"][t]
        act_t = rollout["actions"][t]
        rew_t = rollout["rewards"][t]
        done_t = rollout["dones"][t]
        hid_t = rollout["hidden_states"][t]
        next_obs_t = rollout["next_obs"][t]

        minerl_action, pi_dist, v_pred, log_prob, hid_out = agent.get_action_and_training_info(
            minerl_obs=obs_t,
            hidden_state=hid_t,
            stochastic=False,
            taken_action=act_t
        )

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
        next_val = bootstrap_value if i == T - 1 else transitions[i+1]["v_pred"].item()

        delta = r_i + gamma * next_val * mask - v_i
        gae = delta + gamma * lam * mask * gae
        transitions[i]["advantage"] = gae
        transitions[i]["return"] = v_i + gae

    return transitions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--num-iterations", required=False, type=int, default=10)
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=2)
    parser.add_argument("--mini-batch-size", required=False, type=int, default=10)

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        mini_batch_size=args.mini_batch_size
    )