#!/usr/bin/env python
import os
import signal
import sys
from argparse import ArgumentParser
import importlib
import pickle
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from typing import Dict, List, Any

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from lib.tree_util import tree_map
from lib.policy_mod import compute_kl_loss
from torch.cuda.amp import autocast, GradScaler

# Global flag for clean shutdown
class GracefulExiter:
    def __init__(self):
        self.state = Value('b', False)
        self.lock = threading.Lock()
    
    def trigger(self):
        with self.lock:
            self.state.value = True
    
    def should_exit(self):
        with self.lock:
            return self.state.value

exit_controller = GracefulExiter()

def signal_handler(sig, frame):
    print("\nReceived shutdown signal, initiating graceful exit...")
    exit_controller.trigger()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ----------------- Core Functionality -----------------
def load_model_parameters(path_to_model_file: str) -> tuple:
    with open(path_to_model_file, "rb") as f:
        agent_params = pickle.load(f)
    policy_kwargs = agent_params["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

class RolloutQueue:
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        self._sentinel = object()
    
    def put(self, rollouts: List[Dict[str, Any]]):
        try:
            self.queue.put(rollouts, block=True, timeout=5.0)
        except queue.Full:
            print("Warning: RolloutQueue is full, dropping rollout")
    
    def get(self):
        try:
            return self.queue.get(block=True, timeout=1.0)
        except queue.Empty:
            raise queue.Empty("No rollouts available")
    
    def close(self):
        self.queue.put(self._sentinel)
    
    def qsize(self):
        return self.queue.qsize()

class PhaseCoordinator:
    def __init__(self):
        self.lock = threading.Lock()
        self.aux_phase_event = threading.Event()
        self.rollout_buffer = []
        # Set the event initially to allow rollouts
        self.aux_phase_event.set()
    
    def start_auxiliary_phase(self):
        with self.lock:
            self.aux_phase_event.clear()
            self.rollout_buffer = []
    
    def end_auxiliary_phase(self):
        with self.lock:
            self.aux_phase_event.set()
    
    def buffer_rollout(self, rollout: Dict[str, Any]):  # Fixed syntax error here
        with self.lock:
            if len(self.rollout_buffer) < 5:  # Keep last 5 batches
                self.rollout_buffer.append(rollout)
    
    def get_buffered_rollouts(self) -> List[Dict[str, Any]]:
        with self.lock:
            rollouts = self.rollout_buffer
            self.rollout_buffer = []
            return rollouts
    
    def should_pause(self) -> bool:
        return not self.aux_phase_event.is_set()

# ----------------- Environment Wrapper -----------------
# Adding definition for Survival class
# ----------------- Environment Wrapper -----------------
class HumanSurvival:
    def __init__(self, **kwargs):
        # Use the actual MineRL human survival environment
        self.env_id = "MineRLBasaltMakeWaterfall-v0"  # Verified compatible environment
        self.kwargs = {
            'fov': 90,         # Valid FOV setting instead of fov_range
            'gamma': 2.2,      # Standard MineRL video settings
            'brightness': 1.0,
            'render_resolution': 128  # Reduced for better performance
        }
        # Merge with any provided kwargs
        self.kwargs.update(kwargs)
    
    def make(self):
        return gym.make(self.env_id, **self.kwargs)

# ----------------- Environment Management -----------------
def env_worker(env_id: int, action_queue: Queue, result_queue: Queue, stop_flag: Value):
    try:
        env = HumanSurvival(**ENV_KWARGS).make()
        obs = env.reset()
        visited_chunks = set()
        result_queue.put((env_id, None, obs, False, 0.0, None))
        
        while not stop_flag.value:
            try:
                action = action_queue.get(timeout=0.1)
                if action is None: break
                
                next_obs, _, done, info = env.step(action)
                custom_reward, visited_chunks = reward_function(next_obs, done, info, visited_chunks)
                if done: custom_reward -= 2000.0
                
                result_queue.put((env_id, action, next_obs, done, custom_reward, info))
                if done:
                    obs = env.reset()
                    visited_chunks = set()
                    result_queue.put((env_id, None, obs, False, 0.0, None))
                else:
                    obs = next_obs
            except queue.Empty:
                continue
    except Exception as e:
        print(f"Env {env_id} crashed: {str(e)}")
    finally:
        env.close()
        print(f"Env {env_id} shutdown complete")

def start_env_workers(num_envs: int) -> tuple:
    """Start environment worker processes and return queues for communication"""
    action_queues = []
    stop_flags = []
    
    result_queue = mp.Queue()
    
    for i in range(num_envs):
        action_queue = mp.Queue()
        stop_flag = Value('b', False)
        
        Process(
            target=env_worker,
            args=(i, action_queue, result_queue, stop_flag)
        ).start()
        
        action_queues.append(action_queue)
        stop_flags.append(stop_flag)
    
    return action_queues, result_queue, stop_flags

def rollout_worker(
    agent: MineRLAgent,
    action_queues: List[Queue],
    result_queue: Queue,
    rollout_queue: RolloutQueue,
    phase_coord: PhaseCoordinator,
):
    """Worker to collect rollouts from environments and feed them to training"""
    env_states = {}
    hidden_states = {}
    
    try:
        while not exit_controller.should_exit():
            # Check if we should pause for auxiliary phase
            if phase_coord.should_pause():
                time.sleep(0.1)
                continue
                
            try:
                env_id, action, obs, done, reward, info = result_queue.get(timeout=0.1)
                
                # Initialize new environment
                if action is None:
                    env_states[env_id] = {"obs": [obs], "actions": [], "rewards": [], "dones": []}
                    hidden_states[env_id] = agent.policy.initial_state(1)
                    continue
                
                # Process step
                env_states[env_id]["obs"].append(obs)
                env_states[env_id]["actions"].append(action)
                env_states[env_id]["rewards"].append(reward)
                env_states[env_id]["dones"].append(done)
                
                # Get next action
                with th.no_grad():
                    next_action, hidden_states[env_id] = agent.get_action(
                        obs, hidden_states[env_id]
                    )
                
                # Send action to environment
                action_queues[env_id].put(next_action)
                
                # Check if we have enough steps for a rollout
                if len(env_states[env_id]["actions"]) >= 20:
                    rollout = {
                        "obs": env_states[env_id]["obs"][:-1],  # Exclude last obs
                        "actions": env_states[env_id]["actions"],
                        "rewards": env_states[env_id]["rewards"],
                        "dones": env_states[env_id]["dones"],
                        "hidden_states": [hidden_states[env_id]]
                    }
                    rollout_queue.put([rollout])
                    phase_coord.buffer_rollout(rollout)
                    
                    # Reset the buffer but keep the most recent observation
                    last_obs = env_states[env_id]["obs"][-1]
                    env_states[env_id] = {"obs": [last_obs], "actions": [], "rewards": [], "dones": []}
                
            except queue.Empty:
                continue
                
    except Exception as e:
        print(f"Rollout worker crashed: {str(e)}")
    finally:
        print("Rollout worker shutting down")

# ----------------- Training Core -----------------
def training_thread(
    agent: MineRLAgent,
    pretrained_policy: MineRLAgent,
    rollout_queue: RolloutQueue,
    phase_coord: PhaseCoordinator,
    args: Any
):
    # Configuration
    PPG_CYCLE = 8  # Policy updates per aux phase
    BATCH_SIZE = 128
    MAX_GRAD_NORM = 1.0
    
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    stored_rollouts = []
    cycle_counter = 0
    
    try:
        while not exit_controller.should_exit() and cycle_counter < args.num_iterations:
            # ------ Policy Phase ------
            rollouts = []
            while len(rollouts) < args.rollout_steps:
                try:
                    batch = rollout_queue.get()
                    if isinstance(batch, list):
                        rollouts.extend(batch)
                except queue.Empty:
                    if exit_controller.should_exit(): break
                    time.sleep(0.1)
                    continue
            
            # Process rollouts with LwF
            policy_loss, value_loss, kl_loss = 0.0, 0.0, 0.0
            for rollout in rollouts:
                with autocast():
                    transitions = process_rollout(agent, pretrained_policy, rollout)
                    losses = update_policy(agent, transitions, optimizer, scaler, args.temp)
                    policy_loss += losses[0]
                    value_loss += losses[1]
                    kl_loss += losses[2]
            
            # Store for PPG phase
            if hasattr(agent.policy, 'aux_value_head'):
                stored_rollouts.append(rollouts)
                stored_rollouts = stored_rollouts[-2:]  # Keep last 2 batches
            
            # ------ Auxiliary Phase ------
            if cycle_counter % PPG_CYCLE == 0 and stored_rollouts:
                phase_coord.start_auxiliary_phase()
                aux_losses = []
                
                for rollout_batch in stored_rollouts:
                    with autocast():
                        aux_loss = update_auxiliary(
                            agent,
                            rollout_batch,
                            optimizer,
                            scaler,
                            args.temp
                        )
                        aux_losses.append(aux_loss)
                
                print(f"Aux Phase Loss: {np.mean(aux_losses):.4f}")
                phase_coord.end_auxiliary_phase()
                stored_rollouts = []
                
            cycle_counter += 1
            print(f"Cycle {cycle_counter}/{args.num_iterations} completed. " + 
                  f"Policy Loss: {policy_loss/len(rollouts):.4f}, " +
                  f"Value Loss: {value_loss/len(rollouts):.4f}, " +
                  f"KL Loss: {kl_loss/len(rollouts):.4f}")
            
    except Exception as e:
        print(f"Training crash: {str(e)}")
    finally:
        print("Saving final model...")
        th.save(agent.policy.state_dict(), args.out_weights)

def process_rollout(
    agent: MineRLAgent,
    pretrained_policy: MineRLAgent,
    rollout: Dict[str, Any]
) -> List[Dict[str, Any]]:
    with th.no_grad():
        agent_out = agent.get_sequence_and_training_info(
            rollout["obs"],
            rollout["hidden_states"][0],
            False,
            rollout["actions"]
        )
        pretrained_out = pretrained_policy.get_sequence_and_training_info(
            rollout["obs"],
            pretrained_policy.policy.initial_state(1),
            False,
            rollout["actions"]
        )
    
    returns = calculate_returns(rollout["rewards"], 0.999)
    values = agent_out[1]
    
    # Calculate advantages
    advantages = []
    for ret, val in zip(returns, values):
        advantages.append(ret - val.item())
    
    return [{
        "obs": obs,
        "action": act,
        "cur_logits": cur_logit,
        "old_logits": old_logit,
        "value": val,
        "return": ret,
        "advantages": adv
    } for obs, act, cur_logit, old_logit, val, ret, adv in zip(
        rollout["obs"],
        rollout["actions"],
        agent_out[0],
        pretrained_out[0],
        agent_out[1],
        returns,
        advantages
    )]

def update_policy(agent, transitions, optimizer, scaler, temp):
    optimizer.zero_grad()
    policy_loss, value_loss, kl_loss = 0.0, 0.0, 0.0
    
    for batch in chunker(transitions, 128):
        batch_dict = {}
        for k in ["obs", "action", "old_logits", "return", "advantages"]:
            batch_dict[k] = th.stack([item[k] for item in batch]).to("cuda")
        
        with autocast():
            dist = agent.policy.get_distribution(batch_dict["obs"])
            log_probs = dist.log_prob(batch_dict["action"])
            
            # LwF Component
            kl = compute_kl_loss(
                dist.probs, 
                th.softmax(batch_dict["old_logits"] / temp, dim=-1)
            )
            
            # Value loss
            value_pred = agent.policy.value(batch_dict["obs"])
            v_loss = th.mean((value_pred - batch_dict["return"]) ** 2)
            
            # Combined loss
            loss = -th.mean(log_probs * batch_dict["advantages"]) + v_loss + kl
            
        scaler.scale(loss).backward()
        policy_loss += loss.item()
        value_loss += v_loss.item()
        kl_loss += kl.item()
    
    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
    scaler.step(optimizer)
    scaler.update()
    
    return policy_loss, value_loss, kl_loss
    
def update_auxiliary(agent, rollouts, optimizer, scaler, temp):
    aux_loss = 0.0
    for rollout in rollouts:
        transitions = process_rollout(agent, agent, rollout)  # Using agent as both current and pretrained
        
        for batch in chunker(transitions, 128):
            batch_dict = {}
            for k in ["obs", "old_logits", "return"]:
                batch_dict[k] = th.stack([item[k] for item in batch]).to("cuda")
            
            optimizer.zero_grad()
            with autocast():
                # Auxiliary value update
                aux_values = agent.policy.aux_value(batch_dict["obs"])
                v_loss = th.mean((aux_values - batch_dict["return"]) ** 2)
                
                # Policy distillation component
                dist = agent.policy.get_distribution(batch_dict["obs"])
                kl = compute_kl_loss(
                    dist.probs,
                    th.softmax(batch_dict["old_logits"] / temp, dim=-1)
                )
                
                loss = v_loss + kl
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            aux_loss += loss.item()
    
    return aux_loss / (len(rollouts) * len(batch))

# ----------------- Utilities -----------------
def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def chunker(seq: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

# ----------------- Default reward function -----------------
def default_reward_function(obs, done, info, visited_chunks):
    """Default reward function if lib.phase1 is not available"""
    # Extract player position
    if 'location_stats' in info and 'xpos' in info['location_stats'] and 'zpos' in info['location_stats']:
        x_pos = info['location_stats']['xpos']
        z_pos = info['location_stats']['zpos']
        
        # Calculate chunk coordinates (16x16 blocks per chunk)
        chunk_x = int(x_pos) // 16
        chunk_z = int(z_pos) // 16
        chunk_key = (chunk_x, chunk_z)
        
        # Reward for exploring new chunks
        if chunk_key not in visited_chunks:
            visited_chunks.add(chunk_key)
            return 10.0, visited_chunks
    
    # Small default reward for staying alive
    return 0.1, visited_chunks

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--num-envs", type=int, default=11)
    args = parser.parse_args()

    # Dynamic reward function handling
    try:
        from lib.phase1 import reward_function
        print("Using custom reward function from lib.phase1")
    except ImportError:
        print("Custom reward function not found. Using default reward function.")
        reward_function = default_reward_function

    # Initialize systems
    policy_kwargs, pi_head_kwargs = load_model_parameters(args.in_model)
    env = HumanSurvival(**ENV_KWARGS).make()
    
    agent = MineRLAgent(env, device="cuda",
                       policy_kwargs=policy_kwargs,
                       pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.in_weights)
    
    pretrained = MineRLAgent(env, device="cuda",
                            policy_kwargs=policy_kwargs,
                            pi_head_kwargs=pi_head_kwargs)
    pretrained.load_weights(args.in_weights)
    
    # Start training ecosystem
    rollout_queue = RolloutQueue()
    phase_coord = PhaseCoordinator()
    
    # Start environment workers
    action_queues, result_queue, stop_flags = start_env_workers(args.num_envs)
    
    # Start rollout worker
    rollout_thread = threading.Thread(
        target=rollout_worker,
        args=(agent, action_queues, result_queue, rollout_queue, phase_coord)
    )
    rollout_thread.start()
    
    # Start training thread
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained, rollout_queue, phase_coord, args)
    )
    train_thread.start()
    
    try:
        # Main thread just monitors for exit signal
        while not exit_controller.should_exit():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        exit_controller.trigger()
    finally:
        # Clean shutdown
        print("Initiating clean shutdown...")
        exit_controller.trigger()
        
        # Close queues
        rollout_queue.close()
        
        # Wait for threads to finish
        train_thread.join(timeout=30)
        rollout_thread.join(timeout=30)
        
        # Signal worker processes to stop
        for i, stop_flag in enumerate(stop_flags):
            with stop_flag.get_lock():
                stop_flag.value = True
                action_queues[i].put(None)  # Send sentinel to unblock get() calls
        
        # Close environment
        env.close()
        print("Shutdown complete.")
