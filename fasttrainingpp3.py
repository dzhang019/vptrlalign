from argparse import ArgumentParser
import pickle
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import os

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from data_loader import DataLoader
from lib.tree_util import tree_map
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from torch.cuda.amp import autocast

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import minerl
    minerl.core.start_process_watcher()
    
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    try:
        env = HumanSurvival(**ENV_KWARGS).make()
        obs = env.reset()
        prev_logs = 0
        prev_sword = 0
        episode_step_count = 0
        
        result_queue.put((env_id, None, obs, False, 0, None))
        
        while not stop_flag.value:
            try:
                action = action_queue.get(timeout=0.1)
                if action is None: break
                
                next_obs, _, done, info = env.step(action)
                
                # Reward calculation
                current_logs = info.get('stat/logged', 0)
                current_sword = info.get('stat/crafted_iron_sword', 0)
                custom_reward = (current_logs - prev_logs) * 1.0 + (current_sword - prev_sword) * 10.0
                prev_logs = current_logs
                prev_sword = current_sword
                
                if done: 
                    custom_reward -= 2000.0
                    result_queue.put((env_id, None, None, True, episode_step_count, None))
                    obs = env.reset()
                    prev_logs = 0
                    prev_sword = 0
                    episode_step_count = 0
                    result_queue.put((env_id, None, obs, False, 0, None))
                else:
                    obs = next_obs
                    episode_step_count += 1
                
                result_queue.put((env_id, action, next_obs, done, custom_reward, info))
                
            except queue.Empty:
                continue
        env.close()
    except Exception as e:
        print(f"[Env {env_id}] Critical error: {str(e)}")

def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
                      out_episodes, stop_flag, num_envs, phase_coordinator):
    obs_list = [None] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    for _ in range(num_envs):
        env_id, _, obs, _, _, _ = result_queue.get()
        obs_list[env_id] = obs

    while not stop_flag[0]:
        if phase_coordinator.in_auxiliary_phase():
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            continue
        
        rollouts = [{"obs": [], "actions": [], "rewards": [], "dones": [], 
                    "hidden_states": [], "next_obs": []} for _ in range(num_envs)]
        
        for env_id in range(num_envs):
            if obs_list[env_id] is None: continue
            
            with th.no_grad():
                action_info = agent.get_action_and_training_info(
                    minerl_obs=obs_list[env_id],
                    hidden_state=hidden_states[env_id],
                    stochastic=True
                )
                action_queues[env_id].put(action_info[0])
                hidden_states[env_id] = tree_map(lambda x: x.detach(), action_info[-1])

        total_transitions = 0
        while total_transitions < rollout_steps * num_envs:
            try:
                env_id, action, next_obs, done, reward, info = result_queue.get(timeout=0.1)
                if action is None: continue
                
                rollouts[env_id]["obs"].append(obs_list[env_id])
                rollouts[env_id]["actions"].append(action)
                rollouts[env_id]["rewards"].append(reward)
                rollouts[env_id]["dones"].append(done)
                rollouts[env_id]["hidden_states"].append(
                    tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id]))
                rollouts[env_id]["next_obs"].append(next_obs)
                
                obs_list[env_id] = next_obs
                total_transitions += 1
                
            except queue.Empty:
                continue
        
        if not phase_coordinator.in_auxiliary_phase():
            rollout_queue.put(rollouts)

def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    if T == 0: return transitions
    
    agent_outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=rollout["obs"],
        initial_hidden_state=rollout["hidden_states"][0],
        taken_actions_list=rollout["actions"]
    )
    
    # KL divergence calculation
    with th.no_grad():
        old_outputs = pretrained_policy.get_sequence_and_training_info(
            minerl_obs_list=rollout["obs"],
            initial_hidden_state=pretrained_policy.policy.initial_state(1),
            taken_actions_list=rollout["actions"]
        )
    
    # Advantage calculation
    advantages = []
    last_value = 0.0
    for t in reversed(range(T)):
        delta = rollout["rewards"][t] + gamma * last_value * (1 - rollout["dones"][t]) - agent_outputs[1][t]
        advantages.append(delta + gamma * lam * (advantages[-1] if advantages else 0.0))
        last_value = agent_outputs[1][t]
    
    return [{
        "obs": rollout["obs"][t],
        "action": rollout["actions"][t],
        "advantage": advantages[::-1][t],
        "old_pd": {k: v[t] for k, v in old_outputs[0].items()},
        "cur_pd": {k: v[t] for k, v in agent_outputs[0].items()},
        "value": agent_outputs[1][t]
    } for t in range(T)]

def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator):
    LEARNING_RATE = 3e-6
    CLIP_EPSILON = 0.2
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 15.0
    
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    scaler = th.amp.GradScaler(init_scale=2.**16)
    
    for iteration in range(num_iterations):
        if stop_flag[0]: break
        
        # Policy Update Phase
        rollouts = rollout_queue.get()
        all_transitions = []
        for env_rollout in rollouts:
            all_transitions.extend(train_unroll(agent, pretrained_policy, env_rollout))
        
        # PPO Optimization
        optimizer.zero_grad()
        policy_loss = 0.0
        value_loss = 0.0
        kl_loss = 0.0
        
        for batch in DataLoader(all_transitions, batch_size=256, shuffle=True):
            with autocast():
                advantages = th.tensor([t["advantage"] for t in batch], device="cuda")
                ratios = th.exp(agent.policy.log_prob(batch) - th.exp(pretrained_policy.policy.log_prob(batch))
                surr1 = ratios * advantages
                surr2 = th.clamp(ratios, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
                
                policy_loss += -th.min(surr1, surr2).mean()
                value_loss += F.mse_loss(agent.policy.value(batch), th.tensor([t["value"] for t in batch], device="cuda"))
                kl_loss += compute_kl_loss(
                    {k: v for k, v in agent.policy.dist_params(batch).items()},
                    {k: v for k, v in pretrained_policy.policy.dist_params(batch).items()}
                ).mean()
        
        total_loss = policy_loss + 0.5 * value_loss + LAMBDA_KL * kl_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Iteration {iteration+1}/{num_iterations} | Loss: {total_loss.item():.2f} | KL: {kl_loss.item():.2f}")

def train_rl_mp(in_model, in_weights, out_weights, **kwargs):
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    agent = MineRLAgent(dummy_env, device="cuda", 
                       policy_kwargs=agent_policy_kwargs,
                       pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights, weights_only=True)
    
    pretrained_policy = MineRLAgent(dummy_env, device="cuda",
                                   policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    pretrained_policy.load_weights(in_weights, weights_only=True)
    
    coordinator = PhaseCoordinator()
    stop_flag = Value('b', False)
    action_queues = [Queue() for _ in range(kwargs['num_envs'])]
    result_queue = Queue()
    rollout_queue = RolloutQueue(kwargs['queue_size'])
    
    workers = []
    for env_id in range(kwargs['num_envs']):
        p = Process(target=env_worker, args=(env_id, action_queues[env_id], result_queue, stop_flag))
        p.start()
        workers.append(p)
        time.sleep(0.5)
    
    train_thread = threading.Thread(target=training_thread,
                                   args=(agent, pretrained_policy, rollout_queue, [False], 
                                        kwargs['num_iterations'], coordinator))
    env_thread = threading.Thread(target=environment_thread,
                                 args=(agent, kwargs['rollout_steps'], action_queues,
                                      result_queue, rollout_queue, kwargs['out_episodes'],
                                      [False], kwargs['num_envs'], coordinator))
    
    env_thread.start()
    train_thread.start()
    
    try:
        train_thread.join()
    except KeyboardInterrupt:
        stop_flag.value = True
    finally:
        for p in workers:
            p.terminate()
        th.save(agent.policy.state_dict(), out_weights)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-episodes", default="episodes.txt")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=3)
    
    args = parser.parse_args()
    train_rl_mp(**vars(args))
