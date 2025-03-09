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

from lib.height import reward_function
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


# Function run in each environment process
def env_worker(env_id, action_queue, result_queue, stop_flag):
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize
    obs = env.reset()
    visited_chunks = set()
    episode_step_count = 0
    
    print(f"[Env {env_id}] Started")
    
    # Send initial observation to main process
    result_queue.put((env_id, None, obs, False, 0, None))
    
    while not stop_flag.value:
        try:
            # Get action from queue
            action = action_queue.get(timeout=1.0)
            
            if action is None:  # Signal to terminate
                break
                
            # Step environment
            next_obs, env_reward, done, info = env.step(action)
            
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
            
            # Render (only first environment)
            if env_id == 0:
                env.render()
            
            # Reset if episode is done
            if done:
                result_queue.put((env_id, None, None, True, episode_step_count, None))  # Send episode complete signal
                obs = env.reset()
                visited_chunks = set()
                episode_step_count = 0
                result_queue.put((env_id, None, obs, False, 0, None))  # Send new observation
            else:
                obs = next_obs
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Env {env_id}] Error: {e}")
            
    # Clean up
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
        
        # Collect rollouts
        for step in range(rollout_steps):
            # Check if auxiliary phase started during collection
            if phase_coordinator.in_auxiliary_phase():
                print(f"[Environment Thread] Auxiliary phase started during collection, step {step}/{rollout_steps}")
                break
                
            # For each environment, generate action and send it
            for env_id in range(num_envs):
                if obs_list[env_id] is not None:
                    # Generate action using agent
                    with th.no_grad():
                        # Handle different return signatures
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
            
            # Collect results from all environments
            pending_results = num_envs
            while pending_results > 0:
                try:
                    env_id, action, next_obs, done, reward, info = result_queue.get(timeout=5.0)
                    
                    # Check if this is an episode completion signal
                    if action is None and done and next_obs is None:
                        # This is an episode completion notification
                        episode_length = reward  # Using reward field to pass episode length
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_length}\n")
                        continue  # Don't decrement pending_results, we'll get a new obs
                    
                    # Check if this is an observation update without stepping
                    if action is None and not done:
                        obs_list[env_id] = next_obs
                        continue  # Don't decrement pending_results
                    
                    # Normal step result - store in rollout
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
                    
                    pending_results -= 1
                    
                except queue.Empty:
                    print(f"[Environment Thread] Timeout waiting for results in step {step}")
                    break
        
        # Check if we're in auxiliary phase again before putting rollouts in queue
        # This handles the case where auxiliary phase begins during collection
        if not phase_coordinator.in_auxiliary_phase():
            # Send collected rollouts to training thread
            end_time = time.time()
            duration = end_time - start_time
            
            # Count total transitions
            total_transitions = sum(len(r["obs"]) for r in rollouts)
            
            print(f"[Environment Thread] Iteration {iteration} collected {total_transitions} transitions "
                f"across {num_envs} envs in {duration:.3f}s")
            
            rollout_queue.put(rollouts)
        else:
            # Buffer the rollouts for later use
            print(f"[Environment Thread] Iteration {iteration} - buffering {sum(len(r['obs']) for r in rollouts)} transitions")
            phase_coordinator.buffer_rollout(rollouts)


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
            
            # Value is at index 2 or 3 depending on auxiliary head
            if len(action_outputs) >= 6:  # With aux value head
                vpred_index = 2
            else:
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


# Training thread that handles both policy and auxiliary phases
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator):
    """
    Training thread that handles both the policy gradient phase and the auxiliary phase (PPG).
    
    Args:
        agent: The agent being trained
        pretrained_policy: Reference policy for KL divergence
        rollout_queue: Queue for receiving rollouts from environment thread
        stop_flag: Flag for signaling termination
        num_iterations: Number of iterations to train for
        phase_coordinator: Coordinator for synchronizing phases between threads
    """
    # Hyperparameters
    LEARNING_RATE = 2e-5
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 10.0
    GAMMA = 0.9999
    LAM = 0.95
    VALUE_LOSS_COEF = 0.5
    AUX_VALUE_LOSS_COEF = 0.5  # Coefficient for auxiliary value loss
    KL_DECAY = 0.9995
    
    # PPG specific hyperparameters
    PPG_ENABLED = True  # Enable/disable PPG
    PPG_N_PI_UPDATES = 8  # Number of policy updates before auxiliary phase
    PPG_N_AUX_UPDATES = 6  # Number of auxiliary updates in the auxiliary phase
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
        
        # Determine if we're in auxiliary phase
        do_aux_phase = (PPG_ENABLED and 
                         has_aux_head and 
                         pi_update_counter >= PPG_N_PI_UPDATES and 
                         len(stored_rollouts) > 0)
        
        if do_aux_phase:
            # ===== AUXILIARY PHASE =====
            # Signal start of auxiliary phase
            phase_coordinator.start_auxiliary_phase()
            print(f"[Training Thread] Starting PPG auxiliary phase (iteration {iteration})")
            
            # Flatten the stored rollouts for the auxiliary phase
            flat_rollouts = []
            for rollout_set in stored_rollouts:
                for env_rollout in rollout_set:
                    if len(env_rollout["obs"]) > 0:
                        flat_rollouts.append(env_rollout)
            
            # Skip if no valid rollouts
            if len(flat_rollouts) == 0:
                print("[Training Thread] No valid rollouts for auxiliary phase, skipping")
                phase_coordinator.end_auxiliary_phase()
                pi_update_counter = 0
                stored_rollouts = []
                continue
            
            # Do multiple auxiliary updates
            for aux_iter in range(PPG_N_AUX_UPDATES):
                print(f"[Training Thread] Auxiliary update {aux_iter+1}/{PPG_N_AUX_UPDATES}")
                
                # Track statistics
                total_aux_value_loss = 0.0
                total_policy_distill_loss = 0.0
                total_rollouts = 0
                
                # Process each rollout
                optimizer.zero_grad()
                
                for rollout_idx, rollout in enumerate(flat_rollouts):
                    # Process this rollout into transitions
                    transitions = train_unroll(
                        agent,
                        pretrained_policy, 
                        rollout,
                        gamma=GAMMA,
                        lam=LAM
                    )
                    
                    if len(transitions) == 0:
                        continue
                    
                    # Get policy distributions before update
                    original_policy_dists = []
                    for t in transitions:
                        # Store the current policy distribution
                        original_policy_dists.append({k: v.clone().detach() for k, v in t["cur_pd"].items()})
                    
                    # Extract value targets
                    returns = th.tensor([t["return"] for t in transitions], device="cuda")
                    
                    # Only proceed if we have the auxiliary value head
                    if "aux_v_pred" in transitions[0]:
                        aux_vpreds = th.cat([t["aux_v_pred"].unsqueeze(0) for t in transitions])
                        
                        with autocast():
                            # Auxiliary value loss
                            aux_value_loss = ((aux_vpreds - returns) ** 2).mean()
                            
                            # Policy distillation loss (joint policy update in auxiliary phase)
                            policy_distill_losses = []
                            for i, t in enumerate(transitions):
                                cur_pd = t["cur_pd"]
                                orig_pd = original_policy_dists[i]
                                pd_loss = compute_kl_loss(cur_pd, orig_pd)
                                policy_distill_losses.append(pd_loss)
                            
                            policy_distill_loss = th.stack(policy_distill_losses).mean()
                            
                            # Total auxiliary phase loss
                            loss = (AUX_VALUE_LOSS_COEF * aux_value_loss + 
                                   PPG_BETA_CLONE * policy_distill_loss)
                        
                        # Backward pass
                        scaler.scale(loss).backward()
                        
                        # Update statistics
                        total_aux_value_loss += aux_value_loss.item()
                        total_policy_distill_loss += policy_distill_loss.item()
                    
                    total_rollouts += 1
                
                # Skip update if no valid rollouts
                if total_rollouts == 0:
                    print("[Training Thread] No valid rollouts for auxiliary update, skipping")
                    continue
                
                # Apply gradients
                scaler.unscale_(optimizer)
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Report statistics
                if total_rollouts > 0:
                    avg_aux_value_loss = total_aux_value_loss / total_rollouts
                    avg_policy_distill_loss = total_policy_distill_loss / total_rollouts
                    print(f"[Training Thread] Aux update {aux_iter+1} - "
                          f"AuxValueLoss={avg_aux_value_loss:.4f}, "
                          f"PolicyDistillLoss={avg_policy_distill_loss:.4f}")
            
            # End auxiliary phase and signal environment thread
            phase_coordinator.end_auxiliary_phase()
            print("[Training Thread] Auxiliary phase complete, resuming experience collection")
            
            # Check if environment thread buffered any rollouts during aux phase
            buffered_rollouts = phase_coordinator.get_buffered_rollouts()
            if buffered_rollouts:
                print(f"[Training Thread] Processing {len(buffered_rollouts)} buffered rollouts from auxiliary phase")
                for rollout in buffered_rollouts:
                    rollout_queue.put(rollout)
            
            # Reset tracking variables after auxiliary phase
            pi_update_counter = 0
            stored_rollouts = []
            
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
                # Make a copy to avoid modifications affecting stored rollouts
                stored_rollouts.append(rollouts)
                # Limit stored rollouts to save memory
                if len(stored_rollouts) > 2:
                    stored_rollouts = stored_rollouts[-2:]
            
            train_start = time.time()
            print(f"[Training Thread] Processing rollouts for iteration {iteration}")
            
            # Reset optimizer gradients
            optimizer.zero_grad()
            
            # Track statistics for reporting
            total_loss_val = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_kl_loss = 0.0
            valid_envs = 0
            total_transitions = 0
            
            # Process each environment separately
            for env_i, env_rollout in enumerate(rollouts):
                # Skip empty rollouts
                if len(env_rollout["obs"]) == 0:
                    print(f"[Training Thread] Environment {env_i} has no transitions, skipping")
                    continue
                    
                # Process this environment's rollout
                env_transitions = train_unroll(
                    agent,
                    pretrained_policy,
                    env_rollout,
                    gamma=GAMMA,
                    lam=LAM
                )
                
                if len(env_transitions) == 0:
                    continue
                
                # Process this environment's transitions
                env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) 
                                         for t in env_transitions])
                env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
                env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
                env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
                
                # Normalize advantages for this environment
                env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
                
                # Compute losses for this environment
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
                    
                    # Total loss for this environment
                    env_loss = policy_loss + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
                
                # Backward pass
                scaler.scale(env_loss).backward()
                
                # Accumulate statistics for reporting
                total_loss_val += env_loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_loss += kl_loss.item()
                valid_envs += 1
                total_transitions += len(env_transitions)
            
            # Skip update if no valid transitions found
            if valid_envs == 0:
                print(f"[Training Thread] No valid transitions collected, skipping update.")
                continue
            
            # Apply gradients
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            # Report statistics
            train_duration = time.time() - train_start
            avg_policy_loss = total_policy_loss / valid_envs
            avg_value_loss = total_value_loss / valid_envs
            avg_kl_loss = total_kl_loss / valid_envs
            
            print(f"[Training Thread] Policy Phase {pi_update_counter}/{PPG_N_PI_UPDATES} - "
                  f"Time: {train_duration:.3f}s, Transitions: {total_transitions}, "
                  f"PolicyLoss: {avg_policy_loss:.4f}, ValueLoss: {avg_value_loss:.4f}, "
                  f"KLLoss: {avg_kl_loss:.4f}")
            
            # Update running stats
            running_loss += total_loss_val
            total_steps += total_transitions
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
        env_thread.join(timeout=10)
        train_thread.join(timeout=5)
        
        # Wait for workers to finish
        print("Waiting for worker processes to finish...")
        for i, p in enumerate(workers):
            p.join(timeout=5)
            if p.is_alive():
                print(f"Worker {i} did not terminate, force killing...")
                p.terminate()
        
        # Close dummy environment
        dummy_env.close()
        
        # Save weights
        print(f"Saving weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)