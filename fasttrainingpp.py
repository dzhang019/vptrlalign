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

#from lib.phase1 import reward_function
from lib.reward_structure_mod import custom_reward_function
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from torch.cuda.amp import autocast, GradScaler


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


# Simple thread-safe queue for passing rollouts between threads
class RolloutQueue:
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, rollouts):
        self.queue.put(rollouts, block=True)
    
    def get(self):
        return self.queue.get(block=True)
    
    def qsize(self):
        return self.queue.qsize()


# Phase coordinator for synchronizing policy and auxiliary phases
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
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize
    obs = env.reset()
    visited_chunks = {}
    episode_step_count = 0
    
    print(f"[Env {env_id}] Started")
    
    # Send initial observation to main process
    result_queue.put((env_id, None, obs, False, 0, None))
    
    action_timeout = 0.01
    step_count = 0
    while not stop_flag.value:
        try:
            # Get action from queue
            action = action_queue.get(timeout=action_timeout)
            
            if action is None:  # Signal to terminate
                break
                
            # Step environment
            step_start = time.time()
            next_obs, env_reward, done, info = env.step(action)
            step_time = time.time() - step_start
            #print(f"[ENV WORKER {env_id}] Environment step took {step_time:.3f}s (raw)")
            step_count += 1
            
            # Calculate custom reward
            custom_reward, visited_chunks = custom_reward_function(
                next_obs, done, info, visited_chunks
            )
            
            # Apply death penalty if done
            if done:
                custom_reward -= 2000.0
                
            # Increment step count
            episode_step_count += 1
            
            # Send results back
            result_queue.put((env_id, action, next_obs, done, custom_reward, info))
            if env_id == 0:
                env.render()
            # Reset if episode is done
            if done:
                result_queue.put((env_id, None, None, True, episode_step_count, None))  # Send episode complete signal
                obs = env.reset()
                visited_chunks = {}
                episode_step_count = 0
                result_queue.put((env_id, None, obs, False, 0, None))  # Send new observation
            else:
                obs = next_obs
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Env {env_id}] Error: {e}")
            
    # Clean up
    print(f"[Env {env_id}] Processed {step_count} steps")
    env.close()
    print(f"[Env {env_id}] Stopped")


# Thread for coordinating environments and collecting rollouts
def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
                       out_episodes, stop_flag, num_envs, phase_coordinator):
    # Initialize tracking variables
    obs_list = [None] * num_envs
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    # Wait for initial observations from all environments
    for _ in range(num_envs):
        env_id, _, obs, _, _, _ = result_queue.get()
        obs_list[env_id] = obs
        print(f"[Environment Thread] Got initial observation from env {env_id}")
    
    iteration = 0
    while not stop_flag[0]:
        # Check if we're in auxiliary phase - if so, wait
        if phase_coordinator.in_auxiliary_phase():
            print("[Environment Thread] Pausing collection during auxiliary phase")
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            if phase_coordinator.in_auxiliary_phase():
                continue
        
        iteration += 1
        start_time = time.time()
        
        # Initialize rollouts for each environment
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
        
        # Track which environments are waiting for results
        env_waiting_for_result = [False] * num_envs
        env_step_counts = [0] * num_envs
        
        # Start all environments processing at once
        for env_id in range(num_envs):
            if obs_list[env_id] is not None:
                # Generate action using agent
                with th.no_grad():
                    action_info = agent.get_action_and_training_info(
                        minerl_obs=obs_list[env_id],
                        hidden_state=hidden_states[env_id],
                        stochastic=True,
                        taken_action=None
                    )
                    
                    # Extract new hidden state (last element of return tuple)
                    minerl_action = action_info[0]
                    new_hid = action_info[-1]
                
                # Update hidden state
                hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                
                # Send action to environment
                action_queues[env_id].put(minerl_action)
                env_waiting_for_result[env_id] = True
        
        # Process environment results until all steps are complete
        total_transitions = 0
        result_timeout = 0.01
        
        while total_transitions < rollout_steps * num_envs:
            # Check if auxiliary phase started during collection
            if phase_coordinator.in_auxiliary_phase():
                print(f"[Environment Thread] Auxiliary phase started during collection, step {total_transitions}/{rollout_steps * num_envs}")
                break
            
            try:
                # Get result from any environment
                env_id, action, next_obs, done, reward, info = result_queue.get(timeout=result_timeout)
                
                # Check if this is an episode completion signal
                if action is None and done and next_obs is None:
                    # This is an episode completion notification
                    episode_length = reward  # Using reward field to pass episode length
                    with open(out_episodes, "a") as f:
                        f.write(f"{episode_length}\n")
                    continue  # Don't count this as a transition
                
                # Check if this is an observation update without stepping
                if action is None and not done:
                    obs_list[env_id] = next_obs
                    continue  # Don't count this as a transition
                
                # Normal step result - store in rollout if we were waiting for this environment
                if env_waiting_for_result[env_id]:
                    rollouts[env_id]["obs"].append(obs_list[env_id])
                    rollouts[env_id]["actions"].append(action)
                    rollouts[env_id]["rewards"].append(reward)
                    rollouts[env_id]["dones"].append(done)
                    rollouts[env_id]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id])
                    )
                    rollouts[env_id]["next_obs"].append(next_obs)
                    
                    # Update state
                    obs_list[env_id] = next_obs
                    
                    # Reset hidden state if done
                    if done:
                        hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                    
                    # Mark environment as processed
                    env_waiting_for_result[env_id] = False
                    env_step_counts[env_id] += 1
                    total_transitions += 1
                    
                    # If this environment needs more steps, immediately send next action
                    if env_step_counts[env_id] < rollout_steps:
                        # Generate next action
                        with th.no_grad():
                            action_info = agent.get_action_and_training_info(
                                minerl_obs=obs_list[env_id],
                                hidden_state=hidden_states[env_id],
                                stochastic=True,
                                taken_action=None
                            )
                            
                            # Extract new hidden state (last element of return tuple)
                            minerl_action = action_info[0]
                            new_hid = action_info[-1]
                        
                        # Update hidden state
                        hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                        
                        # Send action to environment
                        action_queues[env_id].put(minerl_action)
                        env_waiting_for_result[env_id] = True
            except queue.Empty:
                continue
        
        # Check if we're in auxiliary phase again before putting rollouts in queue
        if not phase_coordinator.in_auxiliary_phase():
            # Send collected rollouts to training thread
            end_time = time.time()
            duration = end_time - start_time
            
            # Count total transitions
            actual_transitions = sum(len(r["obs"]) for r in rollouts)
            
            rollout_queue.put(rollouts)
            
            print(f"[Environment Thread] Iteration {iteration} collected {actual_transitions} transitions "
                f"across {num_envs} envs in {duration:.3f}s")
        else:
            # Buffer the rollouts for later use
            phase_coordinator.buffer_rollout(rollouts)
            print(f"[Environment Thread] Iteration {iteration} - buffering {sum(len(r['obs']) for r in rollouts)} transitions")


# Process rollouts from a single environment
def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """
    Process a rollout from a single environment into a series of transitions
    with policy distributions, values, advantages, and returns.
    
    Handles both cases where auxiliary value head is present or not.
    """
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    hidden_states_seq = rollout["hidden_states"]

    # Get sequence data from current agent policy
    agent_outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=hidden_states_seq[0],
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    # Handle outputs flexibly based on what's returned
    if len(agent_outputs) == 5:  # With auxiliary value head
        pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hid = agent_outputs
        has_aux_head = True
    else:  # Without auxiliary value head
        pi_dist_seq, vpred_seq, log_prob_seq, final_hid = agent_outputs
        aux_vpred_seq = None
        has_aux_head = False
    
    # Get sequence data from pretrained policy (for KL divergence)
    old_outputs = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    # Handle outputs from pretrained policy
    if len(old_outputs) == 5:  # With auxiliary value head
        old_pi_dist_seq, old_vpred_seq, _, old_log_prob_seq, _ = old_outputs
    else:  # Without auxiliary value head
        old_pi_dist_seq, old_vpred_seq, old_log_prob_seq, _ = old_outputs

    # Create transition for each timestep
    for t in range(T):
        # Extract policy distributions for this timestep
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
        # Create transition data
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
        
        # Add auxiliary value prediction if available
        if has_aux_head and aux_vpred_seq is not None:
            transition["aux_v_pred"] = aux_vpred_seq[t]
            
        transitions.append(transition)

    # Bootstrap value calculation for GAE
    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
            # Get action and training info for bootstrap value
            action_outputs = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            
            # Value is at index 2 regardless of aux head (they return different length tuples)
            vpred_index = 2
            bootstrap_value = action_outputs[vpred_index].item()
    
    # GAE calculation
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
        transitions[i]["return"] = gae + v_i  # Store returns for both value heads

    return transitions


# Get recent rollouts for sleep phase
def get_recent_rollouts(stored_rollouts, max_rollouts=5):
    """
    Extract recent rollouts for the sleep phase, limiting the number to save memory.
    
    Args:
        stored_rollouts: List of rollout batches (each batch contains rollouts from multiple envs)
        max_rollouts: Maximum number of individual rollouts to return
        
    Returns:
        List of individual rollouts (not batched by environment)
    """
    recent_rollouts = []
    
    # Process most recent rollout batches first
    for rollout_batch in reversed(stored_rollouts):
        # Each batch contains rollouts from multiple environments
        for env_rollout in rollout_batch:
            # Skip empty rollouts
            if len(env_rollout["obs"]) > 0:
                recent_rollouts.append(env_rollout)
                if len(recent_rollouts) >= max_rollouts:
                    break
        
        if len(recent_rollouts) >= max_rollouts:
            break
    
    # Reverse to maintain chronological order
    recent_rollouts.reverse()
    
    print(f"[Training Thread] Selected {len(recent_rollouts)} rollouts for sleep phase")
    return recent_rollouts


# Run a PPG sleep phase
# Run a PPG sleep phase
def run_sleep_phase(agent, recent_rollouts, optimizer, scaler, max_grad_norm=1.0, beta_clone=1.0):
    """Run the PPG auxiliary phase with proper memory cleanup between rollouts."""
    has_aux_head = hasattr(agent.policy, 'aux_value_head')
    if not has_aux_head:
        print("[Sleep Phase] Warning: Agent does not have auxiliary value head, skipping sleep phase")
        return
    
    print(f"[Sleep Phase] Running with {len(recent_rollouts)} rollouts")
    
    # Track memory usage
    print(f"Initial CUDA memory: {th.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Maximum sequence length to process at once
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 16
    
    # First cycle: Process rollouts, compute auxiliary value loss, and store original distributions
    print("[Sleep Phase] Running cycle 1/2")
    aux_value_loss_sum = 0.0
    num_transitions = 0
    
    # Create a list to store CPU-only transition data for cycle 2
    cycle2_data = []
    
    # Process each rollout
    for rollout_idx, rollout in enumerate(recent_rollouts):
        if len(rollout["obs"]) == 0:
            continue
            
        print(f"[Sleep Phase] Processing rollout {rollout_idx+1}/{len(recent_rollouts)} in cycle 1/2")
        print(f"Before rollout {rollout_idx+1} processing: {th.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Process rollout to get transitions with returns calculation
        transitions = train_unroll(agent, agent, rollout)
        if len(transitions) == 0:
            continue
        
        # Create a CPU-only copy of essential data for cycle 2
        cpu_transitions = []
        for t in transitions:
            # Create a minimal CPU-only copy with only what we need
            cpu_trans = {
                "obs": t["obs"],  # Keep observation references
                "action": t["action"],  # Keep action references
                "return": t["return"],  # Scalar, so no GPU memory
            }
            cpu_transitions.append(cpu_trans)
        
        # Process in smaller chunks for original distributions
        orig_dist_count = 0
        
        # Get the initial hidden state for this rollout
        current_hidden = tree_map(lambda x: x.to("cuda").contiguous(), rollout["hidden_states"][0])
        
        for chunk_start in range(0, len(transitions), MAX_SEQ_LEN):
            chunk_end = min(chunk_start + MAX_SEQ_LEN, len(transitions))
            chunk = transitions[chunk_start:chunk_end]
            
            # Get obs and actions for this chunk
            chunk_obs = [t["obs"] for t in chunk]
            chunk_actions = [t["action"] for t in chunk]
            
            # Store original policy distributions on CPU
            with th.no_grad():
                outputs = agent.get_sequence_and_training_info(
                    minerl_obs_list=chunk_obs,
                    initial_hidden_state=current_hidden,  # Use current hidden state
                    stochastic=False,
                    taken_actions_list=chunk_actions
                )
                
                # Update hidden state for next chunk (assuming last element contains hidden state)
                final_hidden = outputs[-1]
                current_hidden = tree_map(lambda x: x.detach(), final_hidden)
                
                # Store distributions directly in CPU transitions
                if len(outputs) >= 5:
                    pi_dist_seq = outputs[0]
                    for i, t in enumerate(chunk):
                        cpu_idx = chunk_start + i
                        if cpu_idx < len(cpu_transitions):
                            cpu_transitions[cpu_idx]["orig_pi"] = {
                                k: v[i].clone().detach().cpu() for k, v in pi_dist_seq.items()
                            }
                            orig_dist_count += 1
                
                # Force release GPU memory
                del outputs, pi_dist_seq, chunk_obs, chunk_actions
            
            # Force clear cache
            th.cuda.empty_cache()
        
        # Auxiliary value head optimization for cycle 1
        for batch_start in range(0, len(transitions), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(transitions))
            batch = transitions[batch_start:batch_end]
            
            # Get returns and observations for this batch
            batch_returns = th.tensor([t["return"] for t in batch], device="cuda")
            batch_obs = [t["obs"] for t in batch]
            batch_actions = [t["action"] for t in batch]
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Forward pass with gradients
            with th.enable_grad():
                # Note: We use initial_state here as we're not processing sequentially
                # This is fine for optimizing the auxiliary value head
                outputs = agent.get_sequence_and_training_info(
                    minerl_obs_list=batch_obs,
                    initial_hidden_state=agent.policy.initial_state(1),
                    stochastic=False,
                    taken_actions_list=batch_actions
                )
                
                # Get auxiliary values
                if len(outputs) >= 5:
                    _, _, aux_values, _, _ = outputs
                else:
                    print("[Sleep Phase] Error: No auxiliary value predictions")
                    continue
            
            try:
                # Compute auxiliary value loss
                with th.autocast(device_type='cuda'):
                    aux_value_loss = ((aux_values - batch_returns) ** 2).mean()
                
                # Backward and optimize
                scaler.scale(aux_value_loss).backward()
                scaler.unscale_(optimizer)
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # Update statistics
                aux_value_loss_sum += aux_value_loss.item() * len(batch)
                num_transitions += len(batch)
                
            except Exception as e:
                print(f"[Sleep Phase] Error during cycle 1 optimization: {e}")
                import traceback
                traceback.print_exc()
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
            
            # Force release GPU memory
            del batch_returns, batch_obs, batch_actions, outputs, aux_values
            if 'aux_value_loss' in locals() and isinstance(aux_value_loss, th.Tensor):
                del aux_value_loss
            
            # Force clear cache
            th.cuda.empty_cache()
        
        # Store CPU-only transitions for cycle 2
        cycle2_data.extend(cpu_transitions)
        
        # !!! CRITICAL MEMORY CLEANUP !!!
        # Force complete GPU memory cleanup for this rollout
        del transitions, cpu_transitions
        
        # Force empty CUDA cache
        th.cuda.empty_cache()
        
        print(f"After rollout {rollout_idx+1} processing: {th.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Stored {orig_dist_count} original distributions")
    
    # Rest of the function remains unchanged...
    # Report statistics for cycle 1
    if num_transitions > 0:
        avg_aux_value_loss = aux_value_loss_sum / num_transitions
        print(f"[Sleep Phase] Cycle 1/2 completed - "
              f"Transitions: {num_transitions}, "
              f"AvgAuxValueLoss={avg_aux_value_loss:.6f}, "
              f"AvgPolicyDistillLoss: 0.000000")
    else:
        print(f"[Sleep Phase] Cycle 1/2 - No transitions processed")
    
    # Second cycle remains the same as it doesn't rely on sequential processing
    # for the KL divergence calculation...
    
    print(f"Before cycle 2: {th.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"[Sleep Phase] Running cycle 2/2")
    
    # Track metrics for cycle 2
    aux_value_loss_sum = 0.0
    policy_distill_loss_sum = 0.0
    num_transitions = 0
    
    # Process CPU-only transitions in batches
    for batch_start in range(0, len(cycle2_data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(cycle2_data))
        batch = cycle2_data[batch_start:batch_end]
        
        # Skip empty batches or ones without original distributions
        if not batch or not all("orig_pi" in t for t in batch):
            continue
        
        # Get returns and observations for this batch
        batch_returns = th.tensor([t["return"] for t in batch], device="cuda")
        batch_obs = [t["obs"] for t in batch]
        batch_actions = [t["action"] for t in batch]
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with gradients
        with th.enable_grad():
            outputs = agent.get_sequence_and_training_info(
                minerl_obs_list=batch_obs,
                initial_hidden_state=agent.policy.initial_state(1),
                stochastic=False,
                taken_actions_list=batch_actions
            )
            
            # Get auxiliary values and current policy distributions
            if len(outputs) >= 5:
                curr_pi, _, aux_values, _, _ = outputs
            else:
                print("[Sleep Phase] Error: No auxiliary value predictions in cycle 2")
                continue
        
        try:
            # Compute losses
            with th.autocast(device_type='cuda'):
                # Auxiliary value loss
                aux_value_loss = ((aux_values - batch_returns) ** 2).mean()
                
                # Policy distillation loss
                policy_distill_losses = []
                for i, t in enumerate(batch):
                    # Get original distribution from CPU storage
                    orig_pi = {k: v.to("cuda") for k, v in t["orig_pi"].items()}
                    
                    # Get current distribution
                    curr_pi_i = {k: v[i] for k, v in curr_pi.items()}
                    
                    # Compute KL divergence
                    kl_loss = compute_kl_loss(curr_pi_i, orig_pi)
                    policy_distill_losses.append(kl_loss)
                    
                    # Force release GPU memory for original distribution
                    for k, v in orig_pi.items():
                        del v
                    del orig_pi
                
                if policy_distill_losses:
                    policy_distill_loss = th.stack(policy_distill_losses).mean()
                    loss = aux_value_loss + beta_clone * policy_distill_loss
                    policy_distill_loss_val = policy_distill_loss.item()
                else:
                    loss = aux_value_loss
                    policy_distill_loss_val = 0.0
            
            # Backward and optimize
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            aux_value_loss_sum += aux_value_loss.item() * len(batch)
            policy_distill_loss_sum += policy_distill_loss_val * len(batch)
            num_transitions += len(batch)
            
        except Exception as e:
            print(f"[Sleep Phase] Error during cycle 2 optimization: {e}")
            import traceback
            traceback.print_exc()
            # Reset gradients
            optimizer.zero_grad(set_to_none=True)
        
        # Force release GPU memory
        del batch_returns, batch_obs, batch_actions, outputs, curr_pi, aux_values
        if 'aux_value_loss' in locals() and isinstance(aux_value_loss, th.Tensor):
            del aux_value_loss
        if 'policy_distill_losses' in locals():
            del policy_distill_losses
        if 'policy_distill_loss' in locals() and isinstance(policy_distill_loss, th.Tensor):
            del policy_distill_loss
        if 'loss' in locals() and isinstance(loss, th.Tensor):
            del loss
        
        # Force clear cache
        th.cuda.empty_cache()
    
    # Report statistics for cycle 2
    if num_transitions > 0:
        avg_aux_value_loss = aux_value_loss_sum / num_transitions
        avg_policy_distill_loss = policy_distill_loss_sum / num_transitions
        print(f"[Sleep Phase] Cycle 2/2 completed - "
              f"Transitions: {num_transitions}, "
              f"AvgAuxValueLoss={avg_aux_value_loss:.6f}, "
              f"AvgPolicyDistillLoss={avg_policy_distill_loss:.6f}")
    else:
        print(f"[Sleep Phase] Cycle 2/2 - No transitions processed")
    
    # Final cleanup and memory usage reporting
    del cycle2_data
    th.cuda.empty_cache()
    print("[Sleep Phase] Completed")

# Run policy optimization (wake phase)
def run_policy_update(agent, pretrained_policy, rollouts, optimizer, scaler, 
                      value_loss_coef=0.5, lambda_kl=0.2, max_grad_norm=1.0):
    """
    Run a PPO policy update (wake phase) on the provided rollouts.
    
    Args:
        agent: The agent being trained
        pretrained_policy: Reference policy for KL divergence
        rollouts: List of rollouts to use for optimization
        optimizer: The optimizer to use
        scaler: Gradient scaler for mixed precision training
        value_loss_coef: Coefficient for value function loss
        lambda_kl: Coefficient for KL divergence loss
        max_grad_norm: Maximum gradient norm for clipping
    """
    # Track statistics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl_loss = 0.0
    num_valid_envs = 0
    total_transitions = 0
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Process each environment's rollouts
    for env_idx, env_rollout in enumerate(rollouts):
        # Skip empty rollouts
        if len(env_rollout["obs"]) == 0:
            print(f"[Policy Update] Environment {env_idx} has no transitions, skipping")
            continue
        
        # Process rollout into transitions
        env_transitions = train_unroll(
            agent,
            pretrained_policy,
            env_rollout,
            gamma=0.9999,
            lam=0.95
        )
        
        if len(env_transitions) == 0:
            continue
        
        # Extract data for this environment
        env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) 
                                for t in env_transitions])
        env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
        env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
        env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
        
        # Normalize advantages
        env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
        
        # Compute losses
        with autocast():
            # Policy loss (Actor)
            policy_loss = -(env_advantages * env_log_probs).mean()
            
            # Value function loss (Critic)
            value_loss = ((env_v_preds - env_returns) ** 2).mean()
            
            # KL divergence loss
            kl_losses = []
            for t in env_transitions:
                kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
                kl_losses.append(kl_loss)
            kl_loss = th.stack(kl_losses).mean()
            
            # Total loss
            env_loss = policy_loss + (value_loss_coef * value_loss) + (lambda_kl * kl_loss)
        
        # Backward pass
        scaler.scale(env_loss).backward()
        
        # Update statistics
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
        num_valid_envs += 1
        total_transitions += len(env_transitions)
    
    # Skip update if no valid transitions
    if num_valid_envs == 0:
        print("[Policy Update] No valid transitions, skipping update")
        return 0.0, 0.0, 0.0, 0
    
    # Apply gradients
    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    # Compute averages
    avg_policy_loss = total_policy_loss / num_valid_envs
    avg_value_loss = total_value_loss / num_valid_envs
    avg_kl_loss = total_kl_loss / num_valid_envs
    
    return avg_policy_loss, avg_value_loss, avg_kl_loss, total_transitions


def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator):
    """
    Training thread that handles both policy gradient and auxiliary phases of PPG.
    
    Args:
        agent: The agent being trained
        pretrained_policy: Reference policy for KL divergence
        rollout_queue: Queue for receiving rollouts from environment thread
        stop_flag: Flag for signaling termination
        num_iterations: Number of iterations to train for
        phase_coordinator: Coordinator for synchronizing phases between threads
    """
    # Hyperparameters
    LEARNING_RATE = 1e-5
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 0.5
    GAMMA = 0.9999
    LAM = 0.95
    VALUE_LOSS_COEF = 1
    KL_DECAY = 0.99
    
    # PPG specific hyperparameters
    PPG_ENABLED = False  # Enable/disable PPG
    PPG_N_PI_UPDATES = 8  # Number of policy updates before auxiliary phase
    PPG_BETA_CLONE = 1.0  # Weight for the policy distillation loss
    
    # Setup optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0
    iteration = 0
    scaler = GradScaler()
    
    # PPG tracking variables
    pi_update_counter = 0
    stored_rollouts = []
    
    # Check if agent has auxiliary value head
    has_aux_head = hasattr(agent.policy, 'aux_value_head')
    if has_aux_head:
        print("[Training Thread] Detected auxiliary value head, enabling PPG")
    else:
        print("[Training Thread] No auxiliary value head detected, PPG will be disabled")
        PPG_ENABLED = False
    
    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        
        # Determine if we should run sleep phase
        do_aux_phase = (PPG_ENABLED and 
                         has_aux_head and 
                         pi_update_counter >= PPG_N_PI_UPDATES and 
                         len(stored_rollouts) > 0)
        
        if do_aux_phase:
            # Signal start of auxiliary phase
            phase_coordinator.start_auxiliary_phase()
            print(f"[Training Thread] Starting PPG auxiliary phase (iteration {iteration})")
            
            # Get recent rollouts for sleep phase
            recent_rollouts = get_recent_rollouts(stored_rollouts, max_rollouts=5)
            
            # Run sleep phase
            run_sleep_phase(
                agent=agent,
                recent_rollouts=recent_rollouts,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=MAX_GRAD_NORM,
                beta_clone=PPG_BETA_CLONE
            )
            
            # End auxiliary phase
            phase_coordinator.end_auxiliary_phase()
            print("[Training Thread] Auxiliary phase complete")
            
            # Process buffered rollouts
            buffered_rollouts = phase_coordinator.get_buffered_rollouts()
            if buffered_rollouts:
                print(f"[Training Thread] Processing {len(buffered_rollouts)} buffered rollouts")
                for rollout in buffered_rollouts:
                    rollout_queue.put(rollout)
            
            # Reset tracking variables
            pi_update_counter = 0
            stored_rollouts = []
            
            # Clear CUDA cache
            th.cuda.empty_cache()
            
        else:
            # ===== POLICY PHASE =====
            pi_update_counter += 1
            
            print(f"[Training Thread] Policy phase {pi_update_counter}/{PPG_N_PI_UPDATES} - "
                 f"Waiting for rollouts...")
            
            wait_start = time.time()
            rollouts = rollout_queue.get()
            wait_duration = time.time() - wait_start
            print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts.")
            
            # Store rollouts for PPG auxiliary phase if enabled
            if PPG_ENABLED and has_aux_head:
                # Store rollouts for later use
                stored_rollouts.append(rollouts)
                # Limit stored rollouts to save memory
                if len(stored_rollouts) > 2:
                    stored_rollouts = stored_rollouts[-2:]
            
            train_start = time.time()
            print(f"[Training Thread] Processing rollouts for iteration {iteration}")
            
            # Run policy update
            avg_policy_loss, avg_value_loss, avg_kl_loss, num_transitions = run_policy_update(
                agent=agent,
                pretrained_policy=pretrained_policy,
                rollouts=rollouts,
                optimizer=optimizer,
                scaler=scaler,
                value_loss_coef=VALUE_LOSS_COEF,
                lambda_kl=LAMBDA_KL,
                max_grad_norm=MAX_GRAD_NORM
            )
            
            # Report statistics
            train_duration = time.time() - train_start
            
            print(f"[Training Thread] Policy Phase {pi_update_counter}/{PPG_N_PI_UPDATES} - "
                  f"Time: {train_duration:.3f}s, Transitions: {num_transitions}, "
                  f"PolicyLoss: {avg_policy_loss:.4f}, ValueLoss: {avg_value_loss:.4f}, "
                  f"KLLoss: {avg_kl_loss:.4f}")
            
            # Update running stats
            running_loss += (avg_policy_loss + avg_value_loss + avg_kl_loss) * num_transitions
            total_steps += num_transitions
            avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
            LAMBDA_KL *= KL_DECAY
def train_rl_mp(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2,
    queue_size=3
):
    """
    Multiprocessing version with separate processes for environment stepping
    """
    # Set spawn method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set")
    
    # Create dummy environment for agent initialization
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent for main thread
    agent = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    # Create pretrained policy for KL divergence
    pretrained_policy = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    # Create phase coordinator
    phase_coordinator = PhaseCoordinator()
    
    # Create multiprocessing shared objects
    stop_flag = mp.Value('b', False)
    action_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    # Start environment worker processes
    workers = []
    for env_id in range(num_envs):
        p = Process(
            target=env_worker,
            args=(env_id, action_queues[env_id], result_queue, stop_flag)
        )
        p.daemon = True
        p.start()
        workers.append(p)
        time.sleep(0.4)
    
    # Thread stop flag (for clean shutdown)
    thread_stop = [False]
    
    # Create and start threads
    env_thread = threading.Thread(
        target=environment_thread,
        args=(
            agent, 
            rollout_steps, 
            action_queues, 
            result_queue, 
            rollout_queue, 
            out_episodes, 
            thread_stop,
            num_envs,
            phase_coordinator  # Add phase coordinator
        )
    )
    
    train_thread = threading.Thread(
        target=training_thread,
        args=(
            agent, 
            pretrained_policy, 
            rollout_queue, 
            thread_stop, 
            num_iterations,
            phase_coordinator  # Add phase coordinator
        )
    )
    
    print("Starting threads...")
    env_thread.start()
    train_thread.start()
    
    try:
        # Wait for training thread to complete
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping threads and processes...")
    finally:
        # Signal threads and processes to stop
        print("Setting stop flag...")
        thread_stop[0] = True
        stop_flag.value = True
        
        # Signal all workers to exit
        for q in action_queues:
            try:
                q.put(None)  # Signal to exit
            except:
                pass
        
        # Wait for threads to finish
        print("Waiting for threads to finish...")
        env_thread.join(timeout=10) #Might need to change timeout values
        train_thread.join(timeout=5)
        
        # Wait for workers to finish
        print("Waiting for worker processes to finish...")
        for i, p in enumerate(workers):
            p.join(timeout=5) # Might need to change this
            if p.is_alive():
                print(f"Worker {i} did not terminate, force killing...")
                p.terminate()
        
        # Close dummy environment
        dummy_env.close()
        
        # Save weights
        print(f"Saving weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--num-iterations", required=False, type=int, default=10)
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=4)
    parser.add_argument("--queue-size", required=False, type=int, default=3,
                       help="Size of the queue between environment and training threads")

    args = parser.parse_args()
    
    # Check for auxiliary value head in weights
    weights = th.load(args.in_weights, map_location="cpu")
    has_aux_head = any('aux' in key for key in weights.keys())
    print(f"Model weights {'have' if has_aux_head else 'do not have'} auxiliary value head keys")

    train_rl_mp(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        queue_size=args.queue_size
    )
