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
def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, out_episodes, stop_flag, num_envs):
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
            # For each environment, generate action and send it
            for env_id in range(num_envs):
                if obs_list[env_id] is not None:
                    # Generate action using agent
                    with th.no_grad():
                        minerl_action, _, _, _, _, new_hid = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_id],
                            hidden_state=hidden_states[env_id],
                            stochastic=True,
                            taken_action=None
                        )
                    
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
        
        # Send collected rollouts to training thread
        end_time = time.time()
        duration = end_time - start_time
        
        # Count total transitions
        total_transitions = sum(len(r["obs"]) for r in rollouts)
        
        print(f"[Environment Thread] Iteration {iteration} collected {total_transitions} transitions "
              f"across {num_envs} envs in {duration:.3f}s")
        
        rollout_queue.put(rollouts)

# Modified training thread to maintain sequence integrity in both forward and backward passes
# def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
#     # Hyperparameters
#     LEARNING_RATE = 2e-5
#     MAX_GRAD_NORM = 1.0
#     LAMBDA_KL = 10.0
#     GAMMA = 0.9999
#     LAM = 0.95
#     VALUE_LOSS_COEF = 0.5
#     KL_DECAY = 0.9995
    
#     # Setup optimizer
#     optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
#     running_loss = 0.0
#     total_steps = 0
#     iteration = 0
#     scaler = GradScaler()
    
#     while iteration < num_iterations and not stop_flag[0]:
#         iteration += 1
        
#         print(f"[Training Thread] Waiting for rollouts...")
#         wait_start = time.time()
#         rollouts = rollout_queue.get()
#         wait_duration = time.time() - wait_start
#         print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts.")
        
#         train_start = time.time()
#         print(f"[Training Thread] Processing rollouts for iteration {iteration}")
        
#         # Reset optimizer gradients once before processing all environments
#         optimizer.zero_grad()
        
#         # Track statistics for reporting
#         total_loss_val = 0
#         total_policy_loss = 0
#         total_value_loss = 0
#         total_kl_loss = 0
#         valid_envs = 0
#         total_transitions = 0
        
#         # Process each environment separately
#         for env_i, env_rollout in enumerate(rollouts):
#             # Skip empty rollouts
#             if len(env_rollout["obs"]) == 0:
#                 print(f"[Training Thread] Environment {env_i} has no transitions, skipping")
#                 continue
                
#             # Process this environment's rollout
#             env_transitions = train_unroll(
#                 agent,
#                 pretrained_policy,
#                 env_rollout,
#                 gamma=GAMMA,
#                 lam=LAM
#             )
            
#             if len(env_transitions) == 0:
#                 continue
            
#             # Process this environment's transitions
#             env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in env_transitions])
#             env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
#             env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
#             env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
            
#             # Normalize advantages for this environment
#             env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
            
#             # Compute losses for this environment
#             with autocast():
#                 # Policy loss (Actor)
#                 policy_loss = -(env_advantages * env_log_probs).mean()
                
#                 # Value function loss (Critic)
#                 value_loss = ((env_v_preds - env_returns) ** 2).mean()
                
#                 # KL divergence loss
#                 kl_losses = []
#                 for t in env_transitions:
#                     kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
#                     kl_losses.append(kl_loss)
#                 kl_loss = th.stack(kl_losses).mean()
                
#                 # Total loss for this environment
#                 env_loss = policy_loss + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
            
#             # Separate backward pass for each environment to maintain sequence integrity
#             scaler.scale(env_loss).backward()
            
#             # Accumulate statistics for reporting
#             total_loss_val += env_loss.item()
#             total_policy_loss += policy_loss.item()
#             total_value_loss += value_loss.item()
#             total_kl_loss += kl_loss.item()
#             valid_envs += 1
#             total_transitions += len(env_transitions)
        
#         # Skip update if no valid transitions found
#         if valid_envs == 0:
#             print(f"[Training Thread] No valid transitions collected, skipping update.")
#             continue
        
#         # Average losses for reporting
#         avg_policy_loss = total_policy_loss / valid_envs
#         avg_value_loss = total_value_loss / valid_envs
#         avg_kl_loss = total_kl_loss / valid_envs
        
#         # Apply gradients that were already accumulated from each environment's backward pass
#         scaler.unscale_(optimizer)
#         th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
#         scaler.step(optimizer)
#         scaler.update()
        
#         train_duration = time.time() - train_start
        
#         print(f"[Training Thread] Iteration {iteration}/{num_iterations} took {train_duration:.3f}s "
#               f"to process and train on {total_transitions} transitions from {valid_envs} environments.")
        
#         # Update stats
#         running_loss += total_loss_val
#         total_steps += total_transitions
#         avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
#         LAMBDA_KL *= KL_DECAY
        
#         print(f"[Training Thread] Iteration {iteration}/{num_iterations} "
#               f"Loss={total_loss_val:.4f}, PolicyLoss={avg_policy_loss:.4f}, "
#               f"ValueLoss={avg_value_loss:.4f}, KLLoss={avg_kl_loss:.4f}, "
#               f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Queue={rollout_queue.qsize()}")


# # Keep train_unroll function exactly the same as the original
# def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
#     transitions = []
#     T = len(rollout["obs"])
#     if T == 0:
#         return transitions
    
#     obs_seq = rollout["obs"]
#     act_seq = rollout["actions"]
#     hidden_states_seq = rollout["hidden_states"]

#     pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hid = agent.get_sequence_and_training_info(
#         minerl_obs_list=obs_seq,
#         initial_hidden_state=hidden_states_seq[0],
#         stochastic=False,
#         taken_actions_list=act_seq
#     )
    
#     old_pi_dist_seq, old_vpred_seq, aux_vpred_seq, old_logprob_seq, _ = pretrained_policy.get_sequence_and_training_info(
#         minerl_obs_list=obs_seq,
#         initial_hidden_state=pretrained_policy.policy.initial_state(1),
#         stochastic=False,
#         taken_actions_list=act_seq
#     )

#     for t in range(T):
#         # Create a timestep-specific policy distribution dictionary
#         cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
#         old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
#         transitions.append({
#             "obs": rollout["obs"][t],
#             "action": rollout["actions"][t],
#             "reward": rollout["rewards"][t],
#             "done": rollout["dones"][t],
#             "v_pred": vpred_seq[t],
#             "aux_v_pred": aux_vpred_seq[t],
#             "log_prob": log_prob_seq[t],
#             "cur_pd": cur_pd_t,
#             "old_pd": old_pd_t,
#             "next_obs": rollout["next_obs"][t]
#         })

#     # Bootstrap value calculation
#     bootstrap_value = 0.0
#     if not transitions[-1]["done"]:
#         with th.no_grad():
#             hid_t_cpu = rollout["hidden_states"][-1]
#             hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
#             _, _, v_next, _, _, _ = agent.get_action_and_training_info(
#                 minerl_obs=transitions[-1]["next_obs"],
#                 hidden_state=hid_t,
#                 stochastic=False,
#                 taken_action=None
#             )
#             bootstrap_value = v_next.item()
    
#     # GAE calculation
#     gae = 0.0
#     for i in reversed(range(T)):
#         r_i = transitions[i]["reward"]
#         v_i = transitions[i]["v_pred"].item()
#         done_i = transitions[i]["done"]
#         mask = 1.0 - float(done_i)
#         next_val = bootstrap_value if i == T - 1 else transitions[i+1]["v_pred"].item()
#         delta = r_i + gamma * next_val * mask - v_i
#         gae = delta + gamma * lam * mask * gae
#         transitions[i]["advantage"] = gae
#         transitions[i]["return"] = v_i + gae

#     return transitions
def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    hidden_states_seq = rollout["hidden_states"]

    # Get sequence data from agent policy 
    outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=hidden_states_seq[0],
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    # Handle both 4 and 5 return values
    if len(outputs) == 5:
        pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hid = outputs
    else:
        pi_dist_seq, vpred_seq, log_prob_seq, final_hid = outputs
        aux_vpred_seq = None
    
    # Get sequence data from pretrained policy
    old_outputs = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    # Handle both 4 and 5 return values
    if len(old_outputs) == 5:
        old_pi_dist_seq, old_vpred_seq, _, old_log_prob_seq, _ = old_outputs
    else:
        old_pi_dist_seq, old_vpred_seq, old_log_prob_seq, _ = old_outputs

    for t in range(T):
        # Create a timestep-specific policy distribution dictionary
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
        # Create transition
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
        if aux_vpred_seq is not None:
            transition["aux_v_pred"] = aux_vpred_seq[t]
            
        transitions.append(transition)

    # Bootstrap value calculation
    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
            # Get action and training info
            outputs = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            
            # Extract value prediction (handles different output sizes)
            if len(outputs) >= 3:
                vpred_index = 2  # Index of vpred in returned tuple
                bootstrap_value = outputs[vpred_index].item()
    
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
        transitions[i]["return"] = gae + v_i  # Store return for both value heads

    return transitions


def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
    # Hyperparameters
    LEARNING_RATE = 2e-5
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 10.0
    GAMMA = 0.9999
    LAM = 0.95
    VALUE_LOSS_COEF = 0.5
    AUX_VALUE_LOSS_COEF = 0.5  # For auxiliary value head
    KL_DECAY = 0.9995
    
    # PPG specific hyperparameters
    PPG_ENABLED = True  # Set to False to disable PPG
    PPG_N_MAIN = 6      # Number of main iterations before auxiliary phase
    
    # Setup optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0
    iteration = 0
    scaler = GradScaler()
    
    # Track PPG phases
    main_phase_counter = 0
    stored_rollouts = []  # Store rollouts for auxiliary phase
    
    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        
        # Determine if we should do auxiliary phase
        is_aux_phase = PPG_ENABLED and main_phase_counter >= PPG_N_MAIN and len(stored_rollouts) > 0
        
        if is_aux_phase:
            # ===== AUXILIARY PHASE =====
            print(f"[Training Thread] Starting PPG auxiliary phase at iteration {iteration}")
            
            # Use stored rollouts for auxiliary phase
            aux_rollouts = stored_rollouts[-1]  # Use most recent rollout
            
            train_start = time.time()
            
            # Reset optimizer gradients
            optimizer.zero_grad()
            
            # Track statistics
            total_aux_value_loss = 0
            total_kl_loss = 0
            valid_envs = 0
            
            # Process each environment separately (same as main phase)
            for env_i, env_rollout in enumerate(aux_rollouts):
                # Skip empty rollouts
                if len(env_rollout["obs"]) == 0:
                    continue
                    
                # Get fresh transitions from stored rollout
                env_transitions = train_unroll(
                    agent,
                    pretrained_policy,
                    env_rollout,
                    gamma=GAMMA,
                    lam=LAM
                )
                
                if len(env_transitions) == 0:
                    continue
                
                # Get observations and actions for fresh forward pass
                obs_seq = [t["obs"] for t in env_transitions]
                act_seq = [t["action"] for t in env_transitions]
                returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
                print("Auxiliary Value Head Parameters:")
                for name, param in agent.policy.aux_value_head.named_parameters():
                    print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")

                # After getting the aux_vpred_seq:
                print(f"Raw aux_vpred_seq: {aux_vpred_seq}")
                print("Model attributes:", dir(agent.policy))

                # Check if the attribute is properly initialized
                if hasattr(agent.policy, 'aux_value_head'):
                    print("aux_value_head exists in model")
                else:
                    print("aux_value_head missing from model")
                # Fresh forward pass with gradients
                with th.amp.autocast(device_type='cuda'):
                    # Get policy distributions and auxiliary values
                    pi_dist_seq, _, aux_vpred_seq, _, _ = agent.get_sequence_and_training_info(
                        minerl_obs_list=obs_seq,
                        initial_hidden_state=agent.policy.initial_state(1),
                        stochastic=False,
                        taken_actions_list=act_seq
                    )
                    print(f"Aux value range: {aux_vpred_seq.min().item()} to {aux_vpred_seq.max().item()}")
                    print(f"Returns range: {returns.min().item()} to {returns.max().item()}")

                    # Check for NaNs in the inputs
                    if th.isnan(aux_vpred_seq).any():
                        print("NaN detected in auxiliary value predictions")
                    if th.isnan(returns).any():
                        print("NaN detected in returns")
                    
                    # Auxiliary value loss
                    aux_value_loss = (th.clamp((aux_vpred_seq - returns), min=-100, max=100) ** 2).mean()
                    
                    # KL divergence loss (reuse from main phase)
                    kl_losses = []
                    for t_idx, t in enumerate(env_transitions):
                        # Get current policy distribution for this step
                        cur_pd_t = {k: v[t_idx] for k, v in pi_dist_seq.items()}
                        
                        try:
                            # Try computing KL loss with error handling
                            kl_loss = compute_kl_loss(cur_pd_t, t["old_pd"])
                            
                            # Check for NaN and replace with a reasonable default if needed
                            if th.isnan(kl_loss).any():
                                print(f"Warning: NaN detected in KL loss, using default value")
                                kl_loss = th.tensor(0.1, device="cuda")
                                
                            kl_losses.append(kl_loss)
                        except Exception as e:
                            print(f"Error in KL calculation: {e}")
                            # Use a small default KL value
                            kl_losses.append(th.tensor(0.1, device="cuda"))

                    # Average losses with stability check
                    if len(kl_losses) > 0:
                        kl_loss = th.stack(kl_losses).mean()
                    else:
                        kl_loss = th.tensor(0.0, device="cuda")
                                        
                    # Total auxiliary phase loss
                    env_loss = AUX_VALUE_LOSS_COEF * aux_value_loss + LAMBDA_KL * kl_loss
                
                # Backward pass (same as main phase)
                scaler.scale(env_loss).backward()
                
                # Update statistics
                total_aux_value_loss += aux_value_loss.item()
                total_kl_loss += kl_loss.item()
                valid_envs += 1
            
            # Skip update if no valid environments
            if valid_envs == 0:
                print(f"[Training Thread] No valid environments for auxiliary phase, skipping update.")
                main_phase_counter = 0  # Reset counter anyway
                continue
            
            # Apply gradients (same as main phase)
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            # Report auxiliary phase statistics
            avg_aux_value_loss = total_aux_value_loss / valid_envs
            avg_kl_loss = total_kl_loss / valid_envs
            
            train_duration = time.time() - train_start
            print(f"[Training Thread] Auxiliary Phase completed in {train_duration:.3f}s, "
                  f"AuxValueLoss={avg_aux_value_loss:.4f}, KLLoss={avg_kl_loss:.4f}")
            
            # Reset main phase counter
            main_phase_counter = 0
            
        else:
            # ===== MAIN PHASE =====
            main_phase_counter += 1
            
            print(f"[Training Thread] Main Phase {main_phase_counter}/{PPG_N_MAIN} - Waiting for rollouts...")
            wait_start = time.time()
            rollouts = rollout_queue.get()
            wait_duration = time.time() - wait_start
            print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts.")
            
            # Store rollout for auxiliary phase if PPG is enabled
            if PPG_ENABLED:
                stored_rollouts.append(rollouts)
                if len(stored_rollouts) > 2:  # Keep only recent rollouts
                    stored_rollouts = stored_rollouts[-2:]
            
            train_start = time.time()
            print(f"[Training Thread] Processing rollouts for iteration {iteration}")
            
            # Reset optimizer gradients
            optimizer.zero_grad()
            
            # Track statistics for reporting
            total_loss_val = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_kl_loss = 0
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
                env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in env_transitions])
                env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
                env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
                env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
                
                # Normalize advantages for this environment
                env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
                
                # Compute losses for this environment
                with th.amp.autocast(device_type='cuda'):
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
                
                # Separate backward pass for each environment to maintain sequence integrity
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
            
            # Average losses for reporting
            avg_policy_loss = total_policy_loss / valid_envs
            avg_value_loss = total_value_loss / valid_envs
            avg_kl_loss = total_kl_loss / valid_envs
            
            # Apply gradients
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            train_duration = time.time() - train_start
            
            print(f"[Training Thread] Main Phase {main_phase_counter}/{PPG_N_MAIN} took {train_duration:.3f}s "
                  f"to process and train on {total_transitions} transitions from {valid_envs} environments.")
            
            # Update stats
            running_loss += total_loss_val
            total_steps += total_transitions
            avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
            LAMBDA_KL *= KL_DECAY
            
            print(f"[Training Thread] Main Phase {main_phase_counter}/{PPG_N_MAIN} "
                  f"Loss={total_loss_val:.4f}, PolicyLoss={avg_policy_loss:.4f}, "
                  f"ValueLoss={avg_value_loss:.4f}, KLLoss={avg_kl_loss:.4f}, "
                  f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Queue={rollout_queue.qsize()}")


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
            num_envs
        )
    )
    
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained_policy, rollout_queue, thread_stop, num_iterations)
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
    weights = th.load(args.in_weights, map_location="cpu")
    print("Weight keys:", weights.keys())

    # Look for any keys that might suggest an auxiliary value head
    for key in weights.keys():
        if "auxiliary" in key or "aux" in key:
            print(f"Possible auxiliary head key: {key}")
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