from argparse import ArgumentParser
import pickle
import time
import threading
import queue
from collections import deque

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

th.autograd.set_detect_anomaly(False)  # Set to False to improve performance


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


class RolloutBuffer:
    def __init__(self, max_size=10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, rollout_batch):
        with self.lock:
            self.buffer.append(rollout_batch)
    
    def get(self):
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer.popleft()
    
    def size(self):
        with self.lock:
            return len(self.buffer)


def environment_worker(
    agent, 
    envs, 
    rollout_steps, 
    rollout_buffer, 
    out_episodes,
    stop_event,
    death_penalty=-1000.0
):
    """Worker thread for environment stepping"""
    num_envs = len(envs)
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    while not stop_event.is_set():
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
        
        # Collect rollouts
        for step in range(rollout_steps):
            for env_i in range(num_envs):
                if stop_event.is_set():
                    return
                
                envs[env_i].render()
                
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
                        # print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True
                    
                    if done_flag_i:
                        env_reward_i += death_penalty
                    
                    # Store rollout data
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(env_reward_i)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    rollouts[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach(), hidden_states[env_i])
                    )
                    rollouts[env_i]["next_obs"].append(next_obs_i)
                    
                    # Update state
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
        
        # Add completed rollouts to the buffer
        rollout_buffer.add(rollouts)


def train_worker(
    agent,
    pretrained_policy,
    rollout_buffer,
    stop_event,
    gamma=0.9999,
    lam=0.95,
    learning_rate=3e-7,
    max_grad_norm=1.0,
    lambda_kl=50.0,
    kl_decay=0.9995,
    value_loss_coef=0.5
):
    """Worker thread for training updates"""
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=learning_rate)
    running_loss = 0.0
    total_steps = 0
    current_lambda_kl = lambda_kl
    iteration = 0
    
    while not stop_event.is_set():
        # Wait for rollouts
        rollouts = rollout_buffer.get()
        if rollouts is None:
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            continue
        
        iteration += 1
        # print(f"[Iteration {iteration}] Processing training batch...")
        
        # Process rollouts
        transitions_all = []
        for env_rollout in rollouts:
            if len(env_rollout["obs"]) == 0:
                continue
                
            env_transitions = train_unroll(
                agent,
                pretrained_policy,
                env_rollout,
                gamma=gamma,
                lam=lam
            )
            transitions_all.extend(env_transitions)
        
        if len(transitions_all) == 0:
            # print(f"[Iteration {iteration}] No transitions collected, skipping update.")
            continue
        
        # RL update
        optimizer.zero_grad()
        loss_list = []
        for t in transitions_all:
            advantage = t["advantage"]
            returns_ = t["return"]
            log_prob = t["log_prob"]
            v_pred_ = t["v_pred"]
            cur_pd = t["cur_pd"]
            old_pd = t["old_pd"]
            
            loss_rl = -(advantage * log_prob)
            value_loss = (v_pred_ - th.tensor(returns_, device="cuda")) ** 2
            kl_loss = compute_kl_loss(cur_pd, old_pd)
            
            total_loss_step = loss_rl + (value_loss_coef * value_loss) + (current_lambda_kl * kl_loss)
            loss_list.append(total_loss_step.mean())
        
        total_loss_for_rollout = sum(loss_list) / len(loss_list)
        total_loss_for_rollout.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
        optimizer.step()
        
        # Track stats
        total_loss_val = total_loss_for_rollout.item()
        running_loss += total_loss_val * len(transitions_all)
        total_steps += len(transitions_all)
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        current_lambda_kl *= kl_decay
        
        print(f"[Iteration {iteration}] Loss={total_loss_val:.4f}, Steps={len(transitions_all)}, " 
              f"TotalSteps={total_steps}, AvgLoss={avg_loss:.4f}, BufferSize={rollout_buffer.size()}")


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """Same as in original code, but moved to function for clarity"""
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


def train_rl_pipelined(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    train_duration_minutes=30,
    rollout_steps=40,
    num_envs=4,
    buffer_size=5,
    **hyperparams
):
    """
    Pipelined version with separate threads for environment stepping and training
    """
    
    # Default hyperparams
    hp = {
        "learning_rate": 3e-7,
        "max_grad_norm": 1.0,
        "lambda_kl": 50.0,
        "gamma": 0.9999,
        "lam": 0.95,
        "death_penalty": -1000.0,
        "value_loss_coef": 0.5,
        "kl_decay": 0.9995,
    }
    hp.update(hyperparams)
    
    print("Initializing agents and environments...")
    
    # Create environments
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent and pretrained policy
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
    
    # Verify no weight sharing
    for p1, p2 in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert p1.data_ptr() != p2.data_ptr(), "Weights are shared!"
    
    # Create parallel environments
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    
    # Setup shared buffer and stop event
    rollout_buffer = RolloutBuffer(max_size=buffer_size)
    stop_event = threading.Event()
    
    # Start environment worker thread
    env_thread = threading.Thread(
        target=environment_worker,
        args=(
            agent, 
            envs, 
            rollout_steps, 
            rollout_buffer, 
            out_episodes,
            stop_event,
            hp["death_penalty"]
        )
    )
    
    # Start training worker thread
    train_thread = threading.Thread(
        target=train_worker,
        args=(
            agent,
            pretrained_policy,
            rollout_buffer,
            stop_event,
        ),
        kwargs={
            "gamma": hp["gamma"],
            "lam": hp["lam"],
            "learning_rate": hp["learning_rate"],
            "max_grad_norm": hp["max_grad_norm"],
            "lambda_kl": hp["lambda_kl"],
            "kl_decay": hp["kl_decay"],
            "value_loss_coef": hp["value_loss_coef"]
        }
    )
    
    print(f"Starting training for {train_duration_minutes} minutes...")
    start_time = time.time()
    env_thread.start()
    train_thread.start()
    
    try:
        # Run for specified duration
        end_time = start_time + (train_duration_minutes * 60)
        while time.time() < end_time:
            time.sleep(5)  # Check every 5 seconds
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"Training in progress... {elapsed_minutes:.1f}/{train_duration_minutes} minutes elapsed")
    
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    finally:
        # Cleanup
        print("Stopping threads...")
        stop_event.set()
        
        env_thread.join(timeout=10)
        train_thread.join(timeout=10)
        
        # Close environments
        for env in envs:
            env.close()
        
        print(f"Saving fine-tuned weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)
        
        print("Training complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--train-duration", required=False, type=int, default=30, 
                        help="Training duration in minutes")
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=4)
    parser.add_argument("--buffer-size", required=False, type=int, default=5,
                        help="Size of rollout buffer between env and training threads")
    parser.add_argument("--learning-rate", required=False, type=float, default=3e-7)
    parser.add_argument("--lambda-kl", required=False, type=float, default=50.0)

    args = parser.parse_args()

    train_rl_pipelined(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        train_duration_minutes=args.train_duration,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        lambda_kl=args.lambda_kl
    )