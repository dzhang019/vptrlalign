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
        self.queue.put(rollouts, block=True, timeout=5.0)
    
    def get(self):
        return self.queue.get(block=True, timeout=1.0)
    
    def close(self):
        self.queue.put(self._sentinel)
    
    def qsize(self):
        return self.queue.qsize()

class PhaseCoordinator:
    def __init__(self):
        self.lock = threading.Lock()
        self.aux_phase_event = threading.Event()
        self.rollout_buffer = []
    
    def start_auxiliary_phase(self):
        with self.lock:
            self.aux_phase_event.clear()
            self.rollout_buffer = []
    
    def end_auxiliary_phase(self):
        with self.lock:
            self.aux_phase_event.set()
    
    def buffer_rollout(self, rollout: Dict[str, Any]):
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
    
    return [{
        "obs": obs,
        "action": act,
        "cur_logits": cur_logit,
        "old_logits": old_logit,
        "value": val,
        "return": ret
    } for obs, act, cur_logit, old_logit, val, ret in zip(
        rollout["obs"],
        rollout["actions"],
        agent_out[0],
        pretrained_out[0],
        agent_out[1],
        calculate_returns(rollout["rewards"], 0.999)
    )]

def update_policy(agent, transitions, optimizer, scaler, temp):
    optimizer.zero_grad()
    policy_loss, value_loss, kl_loss = 0.0, 0.0, 0.0
    
    for batch in chunker(transitions, 128):
        batch = {k: th.stack(v).to("cuda") for k, v in batch.items()}
        
        with autocast():
            dist = agent.policy.get_distribution(batch["obs"])
            log_probs = dist.log_prob(batch["action"])
            
            # LwF Component
            kl = compute_kl_loss(
                dist.probs, 
                th.softmax(batch["old_logits"] / temp, dim=-1)
            )
            
            # Value loss
            value_pred = agent.policy.value(batch["obs"])
            v_loss = th.mean((value_pred - batch["return"]) ** 2)
            
            # Combined loss
            loss = -th.mean(log_probs * batch["advantages"]) + v_loss + kl
            
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
        with th.no_grad():
            returns = calculate_returns(rollout["rewards"], 0.999)
            values = agent.policy.value(rollout["obs"])
        
        # Auxiliary value update
        optimizer.zero_grad()
        with autocast():
            current_values = agent.policy.aux_value(rollout["obs"])
            v_loss = th.mean((current_values - returns) ** 2)
            
            # Policy consistency
            dist = agent.policy.get_distribution(rollout["obs"])
            kl = compute_kl_loss(
                dist.probs,
                th.softmax(rollout["old_logits"] / temp, dim=-1)
            )
            
            loss = v_loss + kl
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        aux_loss += loss.item()
    
    return aux_loss / len(rollouts)

# ----------------- Utilities -----------------
def calculate_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def chunker(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    args = parser.parse_args()

    # Dynamic reward function handling
    try:
        from lib.phase1 import reward_function
    except ImportError:
        print("Using default reward function")
        def reward_function(obs, done, info, visited):
            return 1.0 if not done else 0.0, visited

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
    
    train_proc = threading.Thread(
        target=training_thread,
        args=(agent, pretrained, rollout_queue, phase_coord, args)
    )
    train_proc.start()
    
    try:
        while not exit_controller.should_exit():
            time.sleep(1)
    finally:
        rollout_queue.close()
        train_proc.join()
        env.close()
