from argparse import ArgumentParser
import pickle
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import ctypes
import queue

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


th.autograd.set_detect_anomaly(True)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


# Process for handling a single environment without the model
def environment_worker(
    env_id,
    action_queue,
    obs_queue,
    rollout_queue,
    rollout_steps,
    out_episodes,
    stop_flag
):
    # Set different seed for each process
    np.random.seed(env_id + int(time.time()) % 10000)
    
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize state
    obs = env.reset()
    done = False
    episode_step_count = 0
    visited_chunks = set()
    iteration = 0
    
    print(f"[Env Worker {env_id}] Started")
    
    while not stop_flag.value:
        iteration += 1
        
        # Initialize rollout
        rollout = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_obs": []
        }
        
        # Collect rollout
        env_start_time = time.time()
        for step in range(rollout_steps):
            if env_id == 0 and step % 10 == 0:  # Only render occasionally for first env to save resources
                env.render()
            
            if not done:
                episode_step_count += 1
                
                # Send observation to main process and wait for action
                try:
                    obs_queue.put((env_id, obs, done), block=True, timeout=5)
                    worker_id, minerl_action = action_queue.get(block=True, timeout=10)
                    
                    if worker_id != env_id:
                        print(f"[Env Worker {env_id}] Received action for wrong worker {worker_id}")
                        continue
                except queue.Empty:
                    print(f"[Env Worker {env_id}] Timeout waiting for action")
                    continue
                except Exception as e:
                    print(f"[Env Worker {env_id}] Error in action exchange: {e}")
                    continue
                
                # Step environment
                next_obs, env_reward, done_flag, info = env.step(minerl_action)
                if "error" in info:
                    print(f"[Env Worker {env_id}] Error in info: {info['error']}")
                    done_flag = True
                
                # Calculate custom reward
                custom_reward, visited_chunks = custom_reward_function(
                    next_obs, done_flag, info, visited_chunks
                )
                
                # Apply death penalty
                if done_flag:
                    custom_reward += -1000.0  # DEATH_PENALTY
                
                # Store rollout data
                rollout["obs"].append(obs)
                rollout["actions"].append(minerl_action)
                rollout["rewards"].append(custom_reward)
                rollout["dones"].append(done_flag)
                rollout["next_obs"].append(next_obs)
                
                # Update state
                obs = next_obs
                done = done_flag
                
                if done_flag:
                    # Log episode length
                    try:
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_count}\n")
                    except Exception as e:
                        print(f"[Env Worker {env_id}] Failed to write to episodes file: {e}")
                    
                    # Reset environment
                    episode_step_count = 0
                    obs = env.reset()
                    done = False
                    visited_chunks = set()
        
        # Send the rollout to the training process
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        
        try:
            rollout_queue.put((env_id, rollout), block=True, timeout=5)
            print(f"[Env Worker {env_id}] Iteration {iteration} collected {rollout_steps} steps "
                  f"in {env_duration:.3f}s")
        except queue.Full:
            print(f"[Env Worker {env_id}] Queue full, skipping rollout")
    
    # Clean up
    print(f"[Env Worker {env_id}] Shutting down")
    env.close()


# Thread to handle action generation for environment workers
def action_generator_thread(agent, obs_queue, action_queue, worker_hidden_states, stop_flag):
    """Thread that handles processing observations and generating actions for all workers"""
    print("[Action Generator] Thread started")
    
    while not stop_flag.value:
        try:
            # Get observation from a worker
            worker_id, obs, done = obs_queue.get(block=True, timeout=1)
            
            # Reset hidden state if needed
            if done:
                worker_hidden_states[worker_id] = agent.policy.initial_state(batch_size=1)
            
            # Generate action
            with th.no_grad():
                minerl_action, _, _, _, new_hid = agent.get_action_and_training_info(
                    minerl_obs=obs,
                    hidden_state=worker_hidden_states[worker_id],
                    stochastic=True,
                    taken_action=None
                )
            
            # Update hidden state
            worker_hidden_states[worker_id] = tree_map(lambda x: x.detach(), new_hid)
            
            # Send action back to worker
            action_queue.put((worker_id, minerl_action))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Action Generator] Error: {e}")
    
    print("[Action Generator] Thread stopping")

# Main training process
def train_rl_multiprocessed(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=4,
    queue_size=10
):
    """
    Training with multiprocessing for environment stepping
    """
    mp.set_start_method('spawn', force=True)  # Required for CUDA support
    
    # Load model parameters
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create dummy environment and agent for the main process
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent = MineRLAgent(
        dummy_env, 
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    # Create pretrained policy for KL divergence calculation
    pretrained_policy = MineRLAgent(
        dummy_env, 
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    # Create shared queues for communication between processes
    rollout_queue = mp.Queue(maxsize=queue_size)
    obs_queue = mp.Queue(maxsize=num_envs * 2)  # For observations from workers
    action_queue = mp.Queue(maxsize=num_envs * 2)  # For actions to workers
    
    # Initialize hidden states for each worker
    worker_hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    # Shared flag to signal processes to stop
    stop_flag = Value(ctypes.c_bool, False)
    
    # Start action generator thread
    action_thread = threading.Thread(
        target=action_generator_thread,
        args=(agent, obs_queue, action_queue, worker_hidden_states, stop_flag)
    )
    action_thread.daemon = True
    action_thread.start()
    
    # Start environment worker processes
    env_processes = []
    for env_id in range(num_envs):
        p = Process(
            target=environment_worker,
            args=(
                env_id,
                action_queue,
                obs_queue,
                rollout_queue,
                rollout_steps,
                out_episodes,
                stop_flag
            )
        )
        p.daemon = True  # Process will be terminated when main process exits
        p.start()
        env_processes.append(p)
    
    # Wait for processes to initialize
    time.sleep(2)
    
    # Training hyperparameters
    LEARNING_RATE = 3e-7
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 10.0
    GAMMA = 0.9999
    LAM = 0.95
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995
    
    # Setup optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0
    scaler = GradScaler()
    
    print(f"[Main Process] Starting training for {num_iterations} iterations")
    
    # Main training loop
    for iteration in range(1, num_iterations + 1):
        print(f"[Main Process] Iteration {iteration}/{num_iterations} - Collecting rollouts...")
        wait_start_time = time.time()
        
        # Collect rollouts from all environments
        all_rollouts = []
        num_collected = 0
        
        # We need to collect rollouts from each environment
        while num_collected < num_envs:
            try:
                env_id, rollout = rollout_queue.get(block=True, timeout=30)
                all_rollouts.append(rollout)
                num_collected += 1
                print(f"[Main Process] Received rollout from env {env_id}")
            except queue.Empty:
                print(f"[Main Process] Timeout waiting for rollouts, got {num_collected}/{num_envs}")
                # Check if processes are still alive
                alive_processes = sum(1 for p in env_processes if p.is_alive())
                print(f"[Main Process] Alive processes: {alive_processes}/{num_envs}")
                if alive_processes == 0:
                    print("[Main Process] All environment processes died, ending training")
                    break
        
        wait_end_time = time.time()
        wait_duration = wait_end_time - wait_start_time
        print(f"[Main Process] Waited {wait_duration:.3f}s for rollouts")
        
        # If we didn't get any rollouts, skip this iteration
        if not all_rollouts:
            print(f"[Main Process] No rollouts collected, skipping iteration {iteration}")
            continue
        
        # Process rollouts
        train_start_time = time.time()
        transitions_all = []
        
        for rollout in all_rollouts:
            env_transitions = train_unroll(
                agent,
                pretrained_policy,
                rollout,
                gamma=GAMMA,
                lam=LAM
            )
            transitions_all.extend(env_transitions)
        
        if len(transitions_all) == 0:
            print(f"[Main Process] No transitions collected, skipping update")
            continue
        
        # Batch processing
        batch_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in transitions_all])
        batch_returns = th.tensor([t["return"] for t in transitions_all], device="cuda")
        batch_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in transitions_all])
        batch_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in transitions_all])
        
        # Compute losses in batch
        optimizer.zero_grad()
        
        with autocast():
            # Policy loss
            policy_loss = -(batch_advantages * batch_log_probs).mean()
            
            # Value function loss
            value_loss = ((batch_v_preds - batch_returns) ** 2).mean()
            
            # KL divergence loss
            kl_losses = []
            for t in transitions_all:
                kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
                kl_losses.append(kl_loss)
            kl_loss = th.stack(kl_losses).mean()
            
            # Total loss
            total_loss = policy_loss + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
        
        # Backpropagate
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        
        # Update stats
        total_loss_val = total_loss.item()
        running_loss += total_loss_val * len(transitions_all)
        total_steps += len(transitions_all)
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        LAMBDA_KL *= KL_DECAY
        
        print(f"[Main Process] Iteration {iteration}/{num_iterations} "
            f"Loss={total_loss_val:.4f}, PolicyLoss={policy_loss.item():.4f}, "
            f"ValueLoss={value_loss.item():.4f}, KLLoss={kl_loss.item():.4f}, "
            f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Train Duration={train_duration:.3f}s")
        
        # Periodically save weights (every 5 iterations)
        if iteration % 5 == 0:
            temp_weights = f"{out_weights}.iter{iteration}"
            print(f"[Main Process] Saving checkpoint to {temp_weights}")
            th.save(agent.policy.state_dict(), temp_weights)
    
    # Signal processes to stop
    print("[Main Process] Training complete, stopping environment processes")
    stop_flag.value = True
    
    # Wait for processes to finish (with timeout)
    for i, p in enumerate(env_processes):
        p.join(timeout=10)
        if p.is_alive():
            print(f"[Main Process] Environment process {i} did not terminate properly")
    
    # Save final weights
    print(f"[Main Process] Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)
    
    # Clean up
    dummy_env.close()


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """Process a rollout to create transitions for training (modified for the new approach)"""
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    
    # No hidden states in rollout, use fresh initial state
    initial_hidden_state = agent.policy.initial_state(batch_size=1)

    pi_dist_seq, vpred_seq, log_prob_seq, final_hid = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=initial_hidden_state,
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    old_pi_dist_seq, old_vpred_seq, old_logprob_seq, _ = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=act_seq
    )

    for t in range(T):
        # Create a timestep-specific policy distribution dictionary
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
        transitions.append({
            "obs": rollout["obs"][t],
            "action": rollout["actions"][t],
            "reward": rollout["rewards"][t],
            "done": rollout["dones"][t],
            "v_pred": vpred_seq[t],
            "log_prob": log_prob_seq[t],
            "cur_pd": cur_pd_t,
            "old_pd": old_pd_t,
            "next_obs": rollout["next_obs"][t]
        })

    # Bootstrap value for advantage calculation
    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            bootstrap_value = v_next.item()
    
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
        transitions[i]["return"] = v_i + gae

    return transitions


def post_process_rollout(rollout, agent, worker_id, gamma=0.9999, lam=0.95):
    """Add hidden states and compute advantages for a rollout directly after collection"""
    T = len(rollout["obs"])
    if T == 0:
        return rollout
    
    # Get fresh initial state
    initial_hidden_state = agent.policy.initial_state(batch_size=1)
    
    # Get policy distribution, value predictions, and log probabilities
    with th.no_grad():
        pi_dist_seq, vpred_seq, log_prob_seq, final_hid = agent.get_sequence_and_training_info(
            minerl_obs_list=rollout["obs"],
            initial_hidden_state=initial_hidden_state,
            stochastic=False,
            taken_actions_list=rollout["actions"]
        )
    
    # Add value predictions to rollout
    rollout["values"] = vpred_seq
    
    # Compute advantages using GAE
    advantages = []
    returns = []
    gae = 0.0
    
    # Calculate bootstrap value if needed
    bootstrap_value = 0.0
    if not rollout["dones"][-1]:
        with th.no_grad():
            last_obs = rollout["next_obs"][-1]
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=last_obs,
                hidden_state=initial_hidden_state,  # Using fresh state is fine for just value estimate
                stochastic=False,
                taken_action=None
            )
            bootstrap_value = v_next.item()
    
    # GAE calculation
    for i in reversed(range(T)):
        r_i = rollout["rewards"][i]
        v_i = vpred_seq[i].item()
        done_i = rollout["dones"][i]
        mask = 1.0 - float(done_i)
        next_val = bootstrap_value if i == T - 1 else vpred_seq[i+1].item()
        delta = r_i + gamma * next_val * mask - v_i
        gae = delta + gamma * lam * mask * gae
        advantages.append(gae)
        returns.append(v_i + gae)
    
    # Reverse lists to match original order
    advantages.reverse()
    returns.reverse()
    
    # Add to rollout
    rollout["advantages"] = advantages
    rollout["returns"] = returns
    rollout["log_probs"] = log_prob_seq
    rollout["pi_dist"] = pi_dist_seq
    
    return rollout

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--num-iterations", required=False, type=int, default=10)
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=8)
    parser.add_argument("--queue-size", required=False, type=int, default=10,
                        help="Size of the queue between environment and training processes")

    args = parser.parse_args()

    train_rl_multiprocessed(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        queue_size=args.queue_size
    )