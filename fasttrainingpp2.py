#!/usr/bin/env python
from argparse import ArgumentParser
import importlib
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

# Reward function import
from lib.phase1 import reward_function
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
    
    def get(self, timeout=1.0):
        return self.queue.get(block=True, timeout=timeout)
    
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
    action_timeout = 0.01
    step_count = 0
    while not stop_flag.value:
        try:
            action = action_queue.get(timeout=action_timeout)
            if action is None:
                break
            next_obs, env_reward, done, info = env.step(action)
            step_count += 1
            custom_reward, visited_chunks = reward_function(next_obs, done, info, visited_chunks)
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
    env.close()

def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
                       out_episodes, stop_flag, num_envs, phase_coordinator):
    obs_list = [None] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    for _ in range(num_envs):
        try:
            env_id, _, obs, _, _, _ = result_queue.get(timeout=5.0)
            obs_list[env_id] = obs
        except queue.Empty:
            pass
    
    iteration = 0
    while not stop_flag[0]:
        if phase_coordinator.in_auxiliary_phase():
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            continue
        
        iteration += 1
        rollouts = [{"obs": [], "actions": [], "rewards": [], "dones": [],
                     "hidden_states": [], "next_obs": []} for _ in range(num_envs)]
        env_waiting = [False] * num_envs
        
        for env_id in range(num_envs):
            if obs_list[env_id] is not None:
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
                env_waiting[env_id] = True
        
        total_transitions = 0
        while total_transitions < rollout_steps * num_envs and not stop_flag[0]:
            try:
                env_id, action, next_obs, done, reward, info = result_queue.get(timeout=0.01)
                if action is None and done and next_obs is None:
                    with open(out_episodes, "a") as f:
                        f.write(f"{reward}\n")
                    continue
                if action is None and not done:
                    obs_list[env_id] = next_obs
                    continue
                if env_waiting[env_id]:
                    rollouts[env_id]["obs"].append(obs_list[env_id])
                    rollouts[env_id]["actions"].append(action)
                    rollouts[env_id]["rewards"].append(reward)
                    rollouts[env_id]["dones"].append(done)
                    rollouts[env_id]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id])
                    )
                    rollouts[env_id]["next_obs"].append(next_obs)
                    obs_list[env_id] = next_obs
                    if done:
                        hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                    env_waiting[env_id] = False
                    total_transitions += 1
                    if len(rollouts[env_id]["obs"]) < rollout_steps:
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
                        env_waiting[env_id] = True
            except queue.Empty:
                continue
        
        if not stop_flag[0]:
            rollout_queue.put(rollouts)

def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    hidden_states_seq = rollout["hidden_states"]

    agent_outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=hidden_states_seq[0],
        stochastic=False,
        taken_actions_list=act_seq
    )
    pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, _ = agent_outputs

    old_outputs = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=act_seq
    )
    old_pi_dist_seq, old_vpred_seq, _, old_log_prob_seq, _ = old_outputs

    for t in range(T):
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        transition = {
            "obs": rollout["obs"][t],
            "action": rollout["actions"][t],
            "reward": rollout["rewards"][t],
            "done": rollout["dones"][t],
            "v_pred": vpred_seq[t],
            "log_prob": log_prob_seq[t],
            "cur_pd": cur_pd_t,
            "old_pd": old_pd_t,
            "next_obs": rollout["next_obs"][t]
        }
        transitions.append(transition)

    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            action_outputs = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            bootstrap_value = action_outputs[2].item()
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
        transitions[i]["return"] = gae + v_i
    return transitions

def get_recent_rollouts(stored_rollouts, max_rollouts=5):
    recent_rollouts = []
    for rollout_batch in reversed(stored_rollouts):
        for env_rollout in rollout_batch:
            if len(env_rollout["obs"]) > 0:
                recent_rollouts.append(env_rollout)
                if len(recent_rollouts) >= max_rollouts:
                    break
        if len(recent_rollouts) >= max_rollouts:
            break
    recent_rollouts.reverse()
    return recent_rollouts

def run_sleep_phase(agent, recent_rollouts, optimizer, scaler, max_grad_norm=1.0, beta_clone=1.0):
    print("[Sleep Phase] Running auxiliary phase")
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 16
    
    # Cycle 1: Auxiliary value training
    aux_value_loss_sum = 0.0
    num_transitions = 0
    cycle2_data = []
    
    for rollout in recent_rollouts:
        if len(rollout["obs"]) == 0:
            continue
            
        transitions = train_unroll(agent, agent, rollout)
        if len(transitions) == 0:
            continue
        
        # Store CPU data for cycle 2
        cpu_transitions = [{
            "obs": t["obs"],
            "action": t["action"],
            "return": t["return"]
        } for t in transitions]
        
        # Process in chunks for original distributions
        current_hidden = tree_map(lambda x: x.to("cuda"), rollout["hidden_states"][0])
        for chunk_start in range(0, len(transitions), MAX_SEQ_LEN):
            chunk_end = min(chunk_start + MAX_SEQ_LEN, len(transitions))
            chunk_obs = [t["obs"] for t in transitions[chunk_start:chunk_end]]
            chunk_actions = [t["action"] for t in transitions[chunk_start:chunk_end]]
            
            with th.no_grad():
                outputs = agent.get_sequence_and_training_info(
                    minerl_obs_list=chunk_obs,
                    initial_hidden_state=current_hidden,
                    stochastic=False,
                    taken_actions_list=chunk_actions
                )
                final_hidden = outputs[-1]
                current_hidden = tree_map(lambda x: x.detach(), final_hidden)
                
                pi_dist_seq = outputs[0]
                for i, t in enumerate(transitions[chunk_start:chunk_end]):
                    cpu_transitions[chunk_start+i]["orig_pi"] = {
                        k: v[i].clone().detach().cpu() for k, v in pi_dist_seq.items()
                    }
        
        # Auxiliary value optimization
        for batch_start in range(0, len(transitions), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(transitions))
            batch_returns = th.tensor([t["return"] for t in transitions[batch_start:batch_end]], device="cuda")
            batch_obs = [t["obs"] for t in transitions[batch_start:batch_end]]
            batch_actions = [t["action"] for t in transitions[batch_start:batch_end]]
            
            optimizer.zero_grad(set_to_none=True)
            with th.enable_grad():
                outputs = agent.get_sequence_and_training_info(
                    minerl_obs_list=batch_obs,
                    initial_hidden_state=agent.policy.initial_state(1),
                    stochastic=False,
                    taken_actions_list=batch_actions
                )
                _, _, aux_values, _, _ = outputs
            
            aux_value_loss = ((aux_values - batch_returns) ** 2).mean()
            scaler.scale(aux_value_loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            aux_value_loss_sum += aux_value_loss.item() * len(batch_returns)
            num_transitions += len(batch_returns)
        
        cycle2_data.extend(cpu_transitions)
    
    # Cycle 2: Policy distillation
    policy_distill_loss_sum = 0.0
    for batch_start in range(0, len(cycle2_data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(cycle2_data))
        batch = cycle2_data[batch_start:batch_end]
        
        batch_returns = th.tensor([t["return"] for t in batch], device="cuda")
        batch_obs = [t["obs"] for t in batch]
        batch_actions = [t["action"] for t in batch]
        
        optimizer.zero_grad(set_to_none=True)
        outputs = agent.get_sequence_and_training_info(
            minerl_obs_list=batch_obs,
            initial_hidden_state=agent.policy.initial_state(1),
            stochastic=False,
            taken_actions_list=batch_actions
        )
        curr_pi, _, aux_values, _, _ = outputs
        
        policy_distill_losses = []
        for i, t in enumerate(batch):
            orig_pi = {k: v.to("cuda") for k, v in t["orig_pi"].items()}
            curr_pi_i = {k: v[i] for k, v in curr_pi.items()}
            kl_loss = compute_kl_loss(curr_pi_i, orig_pi)
            policy_distill_losses.append(kl_loss)
        
        policy_distill_loss = th.stack(policy_distill_losses).mean()
        aux_value_loss = ((aux_values - batch_returns) ** 2).mean()
        loss = aux_value_loss + beta_clone * policy_distill_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        policy_distill_loss_sum += policy_distill_loss.item() * len(batch)
    
    print(f"[Sleep Phase] Completed")

def run_policy_update(agent, pretrained_policy, rollouts, optimizer, scaler, 
                      value_loss_coef=0.5, lambda_kl=0.2, max_grad_norm=1.0, temp=2.0):
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl_loss = 0.0
    num_valid_envs = 0
    total_transitions = 0

    optimizer.zero_grad()
    for env_idx, env_rollout in enumerate(rollouts):
        if len(env_rollout["obs"]) == 0:
            continue
        
        env_transitions = train_unroll(agent, pretrained_policy, env_rollout)
        if len(env_transitions) == 0:
            continue
        
        env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in env_transitions])
        env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
        env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
        env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
        
        env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
        
        with autocast():
            policy_loss = -(env_advantages * env_log_probs).mean()
            value_loss = ((env_v_preds - env_returns) ** 2).mean()
            kl_losses = []
            for t in env_transitions:
                kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"], T=temp)
                kl_losses.append(kl_loss)
            kl_loss = th.stack(kl_losses).mean()
            env_loss = policy_loss + (value_loss_coef * value_loss) + (lambda_kl * kl_loss)
        
        scaler.scale(env_loss).backward()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
        num_valid_envs += 1
        total_transitions += len(env_transitions)
    
    if num_valid_envs == 0:
        return 0.0, 0.0, 0.0, 0
    
    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    avg_policy_loss = total_policy_loss / num_valid_envs
    avg_value_loss = total_value_loss / num_valid_envs
    avg_kl_loss = total_kl_loss / num_valid_envs
    return avg_policy_loss, avg_value_loss, avg_kl_loss, total_transitions

def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator, args):
    LEARNING_RATE = 3e-7
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = args.lambda_kl
    VALUE_LOSS_COEF = 0.5
    PPG_N_PI_UPDATES = 8
    PPG_BETA_CLONE = 1.0

    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    pi_update_counter = 0
    stored_rollouts = []
    has_aux_head = hasattr(agent.policy, 'aux_value_head')

    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        do_aux_phase = has_aux_head and pi_update_counter >= PPG_N_PI_UPDATES and len(stored_rollouts) > 0
        
        if do_aux_phase:
            phase_coordinator.start_auxiliary_phase()
            recent_rollouts = get_recent_rollouts(stored_rollouts)
            run_sleep_phase(agent, recent_rollouts, optimizer, scaler, MAX_GRAD_NORM, PPG_BETA_CLONE)
            phase_coordinator.end_auxiliary_phase()
            pi_update_counter = 0
            stored_rollouts = []
            th.cuda.empty_cache()
        else:
            pi_update_counter += 1
            rollouts = rollout_queue.get()
            
            if has_aux_head:
                stored_rollouts.append(rollouts)
                if len(stored_rollouts) > 2:
                    stored_rollouts = stored_rollouts[-2:]
            
            avg_policy_loss, avg_value_loss, avg_kl_loss, num_transitions = run_policy_update(
                agent, pretrained_policy, rollouts, optimizer, scaler,
                VALUE_LOSS_COEF, LAMBDA_KL, MAX_GRAD_NORM, args.temp
            )

def train_rl_mp(in_model, in_weights, out_weights, out_episodes,
                num_iterations=10, rollout_steps=40, num_envs=2, queue_size=3):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    agent = MineRLAgent(dummy_env, device="cuda",
                       policy_kwargs=agent_policy_kwargs,
                       pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    
    pretrained_policy = MineRLAgent(dummy_env, device="cuda",
                                   policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    pretrained_policy.load_weights(in_weights)
    
    phase_coordinator = PhaseCoordinator()
    stop_flag = mp.Value('b', False)
    action_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    workers = []
    for env_id in range(num_envs):
        p = Process(target=env_worker, args=(env_id, action_queues[env_id], result_queue, stop_flag))
        p.start()
        workers.append(p)
        time.sleep(0.4)
    
    thread_stop = [False]
    env_thread = threading.Thread(target=environment_thread,
                                 args=(agent, rollout_steps, action_queues, result_queue, rollout_queue,
                                       out_episodes, thread_stop, num_envs, phase_coordinator))
    train_thread = threading.Thread(target=training_thread,
                                  args=(agent, pretrained_policy, rollout_queue, thread_stop,
                                        num_iterations, phase_coordinator, args))
    
    env_thread.start()
    train_thread.start()
    
    try:
        train_thread.join()
    except KeyboardInterrupt:
        thread_stop[0] = True
        stop_flag.value = True
    finally:
        for q in action_queues:
            q.put(None)
        for p in workers:
            p.join()
        dummy_env.close()
        th.save(agent.policy.state_dict(), out_weights)

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
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--lambda-kl", type=float, default=50.0)
    args = parser.parse_args()
    
    train_rl_mp(
        args.in_model,
        args.in_weights,
        args.out_weights,
        args.out_episodes,
        args.num_iterations,
        args.rollout_steps,
        args.num_envs,
        args.queue_size
    )
