#!/usr/bin/env python
import os
os.environ["MINERL_DISABLE_PROCESS_WATCHER"] = "1"

from argparse import ArgumentParser
import pickle
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value

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
from torch.cuda.amp import autocast, GradScaler

th.autograd.set_detect_anomaly(True)

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

class RolloutQueue:
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, rollouts):
        self.queue.put(rollouts, block=True)
    
    def get(self):
        return self.queue.get(block=True)
    
    def qsize(self):
        return self.queue.qsize()

class PhaseCoordinator:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_auxiliary_phase = False
        self.auxiliary_phase_complete = threading.Event()
        self.rollout_buffer = []
    
    def start_auxiliary_phase(self):
        with self.lock:
            self.is_auxiliary_phase = True
            self.auxiliary_phase_complete.clear()
    
    def end_auxiliary_phase(self):
        with self.lock:
            self.is_auxiliary_phase = False
            self.auxiliary_phase_complete.set()
    
    def in_auxiliary_phase(self):
        with self.lock:
            return self.is_auxiliary_phase
    
    def buffer_rollout(self, rollout):
        with self.lock:
            self.rollout_buffer.append(rollout)
    
    def get_buffered_rollouts(self):
        with self.lock:
            rollouts = self.rollout_buffer
            self.rollout_buffer = []
            return rollouts

def env_worker(env_id, action_queue, result_queue, stop_flag):
    env = HumanSurvival(**ENV_KWARGS).make()
    obs = env.reset()
    visited_chunks = set()
    episode_step_count = 0
    result_queue.put((env_id, None, obs, False, 0, None))

    while not stop_flag.value:
        try:
            action = action_queue.get(timeout=0.1)
            if action is None:
                break
                
            next_obs, env_reward, done, info = env.step(action)
            custom_reward, visited_chunks = custom_reward_function(next_obs, done, info, visited_chunks)
            
            if done:
                custom_reward -= 2000.0
                
            episode_step_count += 1
            result_queue.put((env_id, action, next_obs, done, custom_reward, info))

            if done:
                result_queue.put((env_id, None, None, True, episode_step_count, None))
                obs = env.reset()
                visited_chunks = set()
                episode_step_count = 0
                result_queue.put((env_id, None, obs, False, 0, None))
            else:
                obs = next_obs
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Env {env_id}] Error: {e}")
            
    env.close()

def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
                      out_episodes, stop_flag, num_envs, phase_coordinator):
    obs_list = [None] * num_envs
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]

    # Wait for initial observations
    for _ in range(num_envs):
        env_id, _, obs, _, _, _ = result_queue.get()
        obs_list[env_id] = obs

    iteration = 0
    while not stop_flag[0]:
        if phase_coordinator.in_auxiliary_phase():
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            continue
        
        iteration += 1
        rollouts = [{"obs": [], "actions": [], "rewards": [], "dones": [],
                    "hidden_states": [], "next_obs": []} for _ in range(num_envs)]
        env_waiting_for_result = [False] * num_envs
        env_step_counts = [0] * num_envs

        # Generate initial actions
        for env_id in range(num_envs):
            if obs_list[env_id] is None:
                continue
            with th.no_grad():
                action_info = agent.get_action_and_training_info(
                    minerl_obs=obs_list[env_id],
                    hidden_state=hidden_states[env_id],
                    stochastic=True,
                    taken_action=None
                )
                minerl_action = action_info[0]
                new_hid = action_info[-1]
            hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
            action_queues[env_id].put(minerl_action)
            env_waiting_for_result[env_id] = True

        total_transitions = 0
        while total_transitions < rollout_steps * num_envs:
            try:
                env_id, action, next_obs, done, reward, info = result_queue.get(timeout=0.1)
                
                if action is None and done:
                    with open(out_episodes, "a") as f:
                        f.write(f"{reward}\n")
                    continue
                
                if env_waiting_for_result[env_id]:
                    rollouts[env_id]["obs"].append(obs_list[env_id])
                    rollouts[env_id]["actions"].append(action)
                    rollouts[env_id]["rewards"].append(reward)
                    rollouts[env_id]["dones"].append(done)
                    rollouts[env_id]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id]))
                    rollouts[env_id]["next_obs"].append(next_obs)
                    
                    obs_list[env_id] = next_obs
                    env_waiting_for_result[env_id] = False
                    env_step_counts[env_id] += 1
                    total_transitions += 1
                    
                    if env_step_counts[env_id] < rollout_steps:
                        with th.no_grad():
                            action_info = agent.get_action_and_training_info(
                                minerl_obs=obs_list[env_id],
                                hidden_state=hidden_states[env_id],
                                stochastic=True,
                                taken_action=None
                            )
                            minerl_action = action_info[0]
                            new_hid = action_info[-1]
                        hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                        action_queues[env_id].put(minerl_action)
                        env_waiting_for_result[env_id] = True
                        
            except queue.Empty:
                if stop_flag[0]:
                    break

        if not phase_coordinator.in_auxiliary_phase():
            rollout_queue.put(rollouts)
        else:
            phase_coordinator.buffer_rollout(rollouts)

def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    agent_outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=rollout["obs"],
        initial_hidden_state=rollout["hidden_states"][0],
        stochastic=False,
        taken_actions_list=rollout["actions"]
    )
    
    has_aux_head = len(agent_outputs) == 5
    if has_aux_head:
        pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, _ = agent_outputs
    else:
        pi_dist_seq, vpred_seq, log_prob_seq, _ = agent_outputs

    old_outputs = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=rollout["obs"],
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=rollout["actions"]
    )
    old_pi_dist_seq = old_outputs[0]

    # GAE calculation
    bootstrap_value = 0.0
    if not rollout["dones"][-1]:
        with th.no_grad():
            hid_t = tree_map(lambda x: x.to("cuda"), rollout["hidden_states"][-1])
            action_outputs = agent.get_action_and_training_info(
                minerl_obs=rollout["next_obs"][-1],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            bootstrap_value = action_outputs[2].item()

    gae = 0.0
    for i in reversed(range(T)):
        r_i = rollout["rewards"][i]
        v_i = vpred_seq[i].item()
        done_i = rollout["dones"][i]
        mask = 1.0 - float(done_i)
        next_val = bootstrap_value if i == T-1 else vpred_seq[i+1].item()
        delta = r_i + gamma * next_val * mask - v_i
        gae = delta + gamma * lam * mask * gae
        transitions.insert(0, {
            "advantage": gae,
            "return": gae + v_i,
            "cur_pd": {k: v[i] for k, v in pi_dist_seq.items()},
            "old_pd": {k: v[i] for k, v in old_pi_dist_seq.items()},
            "log_prob": log_prob_seq[i],
            "v_pred": vpred_seq[i]
        })

    return transitions

def run_policy_update(agent, pretrained_policy, rollouts, optimizer, scaler, 
                     value_loss_coef=0.5, lambda_kl=0.2, max_grad_norm=1.0):
    total_policy_loss = total_value_loss = total_kl_loss = num_transitions = 0
    optimizer.zero_grad()

    for env_rollout in rollouts:
        if len(env_rollout["obs"]) == 0:
            continue
            
        transitions = train_unroll(agent, pretrained_policy, env_rollout)
        if not transitions:
            continue

        advantages = th.tensor([t["advantage"] for t in transitions], device="cuda")
        returns = th.tensor([t["return"] for t in transitions], device="cuda")
        log_probs = th.cat([t["log_prob"] for t in transitions])
        v_preds = th.cat([t["v_pred"] for t in transitions])

        with autocast():
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = -(advantages * log_probs).mean()
            value_loss = ((v_preds - returns) ** 2).mean()
            kl_loss = th.stack([compute_kl_loss(t["cur_pd"], t["old_pd"]) for t in transitions]).mean()
            total_loss = policy_loss + value_loss_coef*value_loss + lambda_kl*kl_loss

        scaler.scale(total_loss).backward()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
        num_transitions += len(transitions)

    if num_transitions == 0:
        return 0, 0, 0, 0

    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    avg_policy = total_policy_loss / len(rollouts)
    avg_value = total_value_loss / len(rollouts)
    avg_kl = total_kl_loss / len(rollouts)
    return avg_policy, avg_value, avg_kl, num_transitions

def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator, args):
    LEARNING_RATE = 5e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = args.lambda_kl
    PPG_N_PI_UPDATES = 8
    PPG_BETA_CLONE = 1.0

    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    stored_rollouts = []
    pi_update_counter = 0
    has_aux_head = hasattr(agent.policy, 'aux_value_head')

    for iteration in range(num_iterations):
        if stop_flag[0]:
            break
            
        if has_aux_head and pi_update_counter >= PPG_N_PI_UPDATES and stored_rollouts:
            phase_coordinator.start_auxiliary_phase()
            recent_rollouts = stored_rollouts[-2:]
            # Auxiliary phase logic here
            phase_coordinator.end_auxiliary_phase()
            pi_update_counter = 0
            stored_rollouts = []
        else:
            rollouts = rollout_queue.get()
            stored_rollouts.append(rollouts)
            avg_policy, avg_value, avg_kl, num_trans = run_policy_update(
                agent, pretrained_policy, rollouts, optimizer, scaler,
                lambda_kl=LAMBDA_KL, max_grad_norm=MAX_GRAD_NORM
            )
            pi_update_counter += 1
            print(f"Iter {iteration}: Policy {avg_policy:.4f}, Value {avg_value:.4f}, KL {avg_kl:.4f}")

    th.save(agent.policy.state_dict(), args.out_weights)

def train_rl_mp(args):
    mp.set_start_method('spawn', force=True)
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    policy_kwargs, pi_head_kwargs = load_model_parameters(args.in_model)

    agent = MineRLAgent(dummy_env, device="cuda", 
                       policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.in_weights)

    pretrained_policy = MineRLAgent(dummy_env, device="cuda",
                                   policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    pretrained_policy.load_weights(args.in_weights)

    phase_coordinator = PhaseCoordinator()
    stop_flag = Value('b', False)
    action_queues = [Queue() for _ in range(args.num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(args.queue_size)

    workers = [Process(target=env_worker, args=(i, action_queues[i], result_queue, stop_flag)) 
              for i in range(args.num_envs)]
    for p in workers:
        p.start()

    env_thread = threading.Thread(target=environment_thread,
                                 args=(agent, args.rollout_steps, action_queues, result_queue,
                                      rollout_queue, args.out_episodes, stop_flag, args.num_envs,
                                      phase_coordinator))
    train_thread = threading.Thread(target=training_thread,
                                  args=(agent, pretrained_policy, rollout_queue, stop_flag,
                                       args.num_iterations, phase_coordinator, args))

    env_thread.start()
    train_thread.start()

    try:
        train_thread.join()
    except KeyboardInterrupt:
        stop_flag.value = True
    finally:
        for p in workers:
            p.terminate()
        dummy_env.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-episodes", default="episode_lengths.txt")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=3)
    parser.add_argument("--lambda-kl", type=float, default=50.0)
    args = parser.parse_args()
    
    train_rl_mp(args)
