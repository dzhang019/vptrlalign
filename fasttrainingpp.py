from argparse import ArgumentParser
import pickle
import time
import threading
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


# Function to collect data from a single environment
def env_worker_process(
    worker_id,
    env_seed,
    obs_input_queue,
    action_output_queue,
    rollout_output_queue,
    rollout_steps,
    out_episodes,
    stop_flag
):
    """
    Worker process for running a single environment
    
    Args:
        worker_id: ID of this worker
        env_seed: Seed for environment randomness
        obs_input_queue: Queue to receive observations
        action_output_queue: Queue to send actions
        rollout_output_queue: Queue to send completed rollouts
        rollout_steps: Number of steps per rollout
        out_episodes: Path to episode length logging file
        stop_flag: Shared flag to signal processes to stop
    """
    # Set random seed for this process
    np.random.seed(env_seed)
    
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize environment state
    obs = env.reset()
    done = False
    episode_step_count = 0
    visited_chunks = set()
    total_steps = 0
    
    print(f"[Worker {worker_id}] Started environment with seed {env_seed}")
    
    # Main loop
    while not stop_flag.value:
        # Initialize new rollout
        rollout = {
            "worker_id": worker_id,
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_obs": [],
            "infos": []
        }
        
        # Collect single rollout
        steps_collected = 0
        rollout_start_time = time.time()
        
        for step in range(rollout_steps):
            if worker_id == 0 and steps_collected % 10 == 0:  # Render only first environment occasionally
                env.render()
            
            # Send current observation to main process and get action
            obs_input_queue.put((worker_id, obs, done), block=True)
            
            try:
                # Wait for action from main process
                recv_worker_id, action = action_output_queue.get(block=True, timeout=10)
                
                # Make sure action is for this worker
                if recv_worker_id != worker_id:
                    print(f"[Worker {worker_id}] Received action for worker {recv_worker_id}. Skipping step.")
                    continue
                
                # Take action in environment
                next_obs, env_reward, done_flag, info = env.step(action)
                
                # Check for environment errors
                if "error" in info:
                    print(f"[Worker {worker_id}] Error in environment: {info['error']}")
                    done_flag = True
                
                # Calculate custom reward
                custom_reward, visited_chunks = custom_reward_function(
                    next_obs, done_flag, info, visited_chunks
                )
                
                # Apply death penalty if episode ended
                if done_flag:
                    custom_reward -= 1000.0  # Death penalty
                
                # Store step data in rollout
                rollout["obs"].append(obs)
                rollout["actions"].append(action)
                rollout["rewards"].append(custom_reward)
                rollout["dones"].append(done_flag)
                rollout["next_obs"].append(next_obs)
                rollout["infos"].append(info)
                
                # Update state
                obs = next_obs
                done = done_flag
                steps_collected += 1
                total_steps += 1
                
                # Handle episode termination
                if done:
                    # Log episode length
                    try:
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_count}\n")
                    except Exception as e:
                        print(f"[Worker {worker_id}] Failed to write episode length: {e}")
                    
                    # Reset environment
                    obs = env.reset()
                    done = False
                    episode_step_count = 0
                    visited_chunks = set()
                    
                    # Notify main process of reset
                    obs_input_queue.put((worker_id, obs, True), block=True)  # True for reset
                else:
                    episode_step_count += 1
                
            except queue.Empty:
                print(f"[Worker {worker_id}] Timeout waiting for action")
                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Exception during step: {e}")
                continue
        
        # Send completed rollout to main process
        if steps_collected > 0:
            rollout_time = time.time() - rollout_start_time
            print(f"[Worker {worker_id}] Completed rollout with {steps_collected} steps in {rollout_time:.2f}s")
            
            try:
                rollout_output_queue.put(rollout, block=True, timeout=5)
            except queue.Full:
                print(f"[Worker {worker_id}] Rollout queue full, discarding rollout")
    
    # Clean up when stopping
    print(f"[Worker {worker_id}] Stopping, collected {total_steps} total steps")
    env.close()


# Thread for handling action generation
def action_server_thread(agent, obs_queue, action_queue, worker_hidden_states, stop_flag):
    """
    Thread that generates actions for environment workers using the shared agent
    
    Args:
        agent: The MineRLAgent instance
        obs_queue: Queue to receive observations from workers
        action_queue: Queue to send actions back to workers
        worker_hidden_states: Dict mapping worker IDs to their hidden states
        stop_flag: Flag to signal thread to stop
    """
    print("[Action Server] Started")
    
    while not stop_flag.value:
        try:
            # Get next observation from queue
            worker_id, obs, reset_flag = obs_queue.get(block=True, timeout=0.1)
            
            # Reset hidden state if needed (new episode or first observation)
            if reset_flag or worker_id not in worker_hidden_states:
                worker_hidden_states[worker_id] = agent.policy.initial_state(batch_size=1)
                print(f"[Action Server] Reset hidden state for worker {worker_id}")
            
            # Get action using agent
            try:
                start_time = time.time()
                with th.no_grad():
                    minerl_action, _, _, _, new_hidden_state = agent.get_action_and_training_info(
                        minerl_obs=obs,
                        hidden_state=worker_hidden_states[worker_id],
                        stochastic=True,
                        taken_action=None
                    )
                
                # Update hidden state
                worker_hidden_states[worker_id] = tree_map(lambda x: x.detach(), new_hidden_state)
                
                # Send action back to worker
                action_queue.put((worker_id, minerl_action))
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000
                if inference_time > 50:  # Log only slow inferences
                    print(f"[Action Server] Generated action for worker {worker_id} in {inference_time:.2f}ms")
                
            except Exception as e:
                print(f"[Action Server] Error generating action: {e}")
                # Send a default action
                action_queue.put((worker_id, {}))
        
        except queue.Empty:
            # No observations to process, which is fine
            continue
        except Exception as e:
            print(f"[Action Server] Unexpected error: {e}")
    
    print("[Action Server] Stopping")


# Process rollout data for training
def prepare_rollout_for_training(rollout, agent, pretrained_policy, precomputed_values=None):
    """
    Process a rollout into transitions suitable for training
    
    Args:
        rollout: Rollout data from an environment
        agent: Current training agent
        pretrained_policy: Original pretrained policy for KL divergence
        precomputed_values: Optional precomputed value predictions
        
    Returns:
        list: List of transition dictionaries with all required training data
    """
    # Extract data from rollout
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    reward_seq = rollout["rewards"]
    done_seq = rollout["dones"]
    next_obs_seq = rollout["next_obs"]
    
    T = len(obs_seq)  # Sequence length
    if T == 0:
        return []
    
    # Get initial hidden states
    initial_hidden_state = agent.policy.initial_state(batch_size=1)
    pretrained_hidden_state = pretrained_policy.policy.initial_state(batch_size=1)
    
    # Get agent's policy distribution, value predictions, and log probabilities
    with th.no_grad():
        pi_dist_seq, vpred_seq, log_prob_seq, _ = agent.get_sequence_and_training_info(
            minerl_obs_list=obs_seq,
            initial_hidden_state=initial_hidden_state,
            stochastic=False,
            taken_actions_list=act_seq
        )
        
        # Get pretrained policy's distributions for KL calculation
        old_pi_dist_seq, old_vpred_seq, old_logprob_seq, _ = pretrained_policy.get_sequence_and_training_info(
            minerl_obs_list=obs_seq,
            initial_hidden_state=pretrained_hidden_state,
            stochastic=False,
            taken_actions_list=act_seq
        )
    
    # Create transition dictionaries for each timestep
    transitions = []
    for t in range(T):
        # Create timestep-specific policy distribution dictionaries
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
        transitions.append({
            "obs": obs_seq[t],
            "action": act_seq[t],
            "reward": reward_seq[t],
            "done": done_seq[t],
            "v_pred": vpred_seq[t],
            "log_prob": log_prob_seq[t],
            "cur_pd": cur_pd_t,
            "old_pd": old_pd_t,
            "next_obs": next_obs_seq[t]
        })
    
    # Calculate bootstrap value for Generalized Advantage Estimation (GAE)
    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            # Get value of last next_obs
            last_obs = transitions[-1]["next_obs"]
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=last_obs,
                hidden_state=initial_hidden_state,  # Fresh state is fine for value function
                stochastic=False,
                taken_action=None
            )
            bootstrap_value = v_next.item()
    
    # Calculate advantages and returns using GAE
    gamma = 0.9999  # Discount factor
    lam = 0.95      # GAE lambda parameter
    gae = 0.0
    
    for i in reversed(range(T)):
        r_i = transitions[i]["reward"]
        v_i = transitions[i]["v_pred"].item()
        done_i = transitions[i]["done"]
        mask = 1.0 - float(done_i)  # 0 if done, 1 otherwise
        
        # Get next value (either bootstrap or from next transition)
        next_val = bootstrap_value if i == T - 1 else transitions[i+1]["v_pred"].item()
        
        # Calculate TD error and GAE
        delta = r_i + gamma * next_val * mask - v_i
        gae = delta + gamma * lam * mask * gae
        
        # Add advantage and return to transition
        transitions[i]["advantage"] = gae
        transitions[i]["return"] = v_i + gae
    
    return transitions


# Main training function
def train_rl_multiprocessed(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=4,
    queue_size=10,
    batch_size=128
):
    """
    Main training function using multiprocessing for environments
    
    Args:
        in_model: Path to input model pickle file
        in_weights: Path to input weights file
        out_weights: Path to output weights file
        out_episodes: Path to episode length logging file
        num_iterations: Number of training iterations
        rollout_steps: Number of steps per rollout
        num_envs: Number of parallel environments
        queue_size: Size of communication queues
        batch_size: Maximum batch size for training updates
    """
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    print(f"Starting training with {num_envs} environments, {rollout_steps} steps per rollout")
    
    # Load model parameters
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create dummy environment for initialization
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    
    # Create agent for training
    agent = MineRLAgent(
        dummy_env, 
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    # Create pretrained policy for KL divergence
    pretrained_policy = MineRLAgent(
        dummy_env, 
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    print("Agents created successfully")
    
    # Verify weights are not shared between agent and pretrained policy
    for p1, p2 in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert p1.data_ptr() != p2.data_ptr(), "Error: Weights are shared between agent and pretrained policy"
    
    # Create communication queues
    rollout_queue = mp.Queue(maxsize=queue_size)
    obs_queue = mp.Queue(maxsize=num_envs * 2)
    action_queue = mp.Queue(maxsize=num_envs * 2)
    
    # Create shared stop flag
    stop_flag = Value(ctypes.c_bool, False)
    
    # Dictionary to store hidden states for each worker
    worker_hidden_states = {}
    
    # Start action server thread
    action_server = threading.Thread(
        target=action_server_thread,
        args=(agent, obs_queue, action_queue, worker_hidden_states, stop_flag)
    )
    action_server.daemon = True
    action_server.start()
    
    # Start environment worker processes
    env_processes = []
    for i in range(num_envs):
        env_seed = i + int(time.time()) % 10000  # Different seed for each worker
        p = Process(
            target=env_worker_process,
            args=(
                i,  # Worker ID
                env_seed,
                obs_queue,
                action_queue,
                rollout_queue,
                rollout_steps,
                out_episodes,
                stop_flag
            )
        )
        p.daemon = True
        p.start()
        env_processes.append(p)
    
    # Wait for processes to initialize
    print("Waiting for environment processes to initialize...")
    time.sleep(3)
    
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
    scaler = GradScaler()  # For mixed precision training
    
    # Training statistics
    running_loss = 0.0
    total_steps = 0
    total_rollouts = 0
    
    # Training loop
    print(f"Starting training for {num_iterations} iterations")
    for iteration in range(1, num_iterations + 1):
        iter_start_time = time.time()
        
        # Collect rollouts from all workers
        print(f"[Iteration {iteration}/{num_iterations}] Collecting rollouts...")
        rollouts = []
        num_rollouts_collected = 0
        collection_timeout = 60  # Maximum time to wait for rollouts (seconds)
        collection_start = time.time()
        
        while num_rollouts_collected < num_envs:
            # Check if we've waited too long
            if time.time() - collection_start > collection_timeout:
                print(f"[Iteration {iteration}] Timeout collecting rollouts, proceeding with {len(rollouts)}")
                break
            
            try:
                # Try to get a rollout from the queue
                rollout = rollout_queue.get(block=True, timeout=1.0)
                rollouts.append(rollout)
                num_rollouts_collected += 1
            except queue.Empty:
                # Check if processes are still alive
                alive_processes = sum(1 for p in env_processes if p.is_alive())
                print(f"[Iteration {iteration}] Waiting for rollouts... ({num_rollouts_collected}/{num_envs}), "
                      f"{alive_processes} workers alive")
                if alive_processes == 0:
                    print("All worker processes have died, stopping training")
                    stop_flag.value = True
                    break
        
        # Skip iteration if no rollouts collected
        if len(rollouts) == 0:
            print(f"[Iteration {iteration}] No rollouts collected, skipping update")
            continue
        
        # Process rollouts into transitions
        print(f"[Iteration {iteration}] Processing {len(rollouts)} rollouts")
        all_transitions = []
        
        for rollout in rollouts:
            transitions = prepare_rollout_for_training(rollout, agent, pretrained_policy)
            all_transitions.extend(transitions)
            total_rollouts += 1
        
        # Skip if no valid transitions
        if len(all_transitions) == 0:
            print(f"[Iteration {iteration}] No valid transitions, skipping update")
            continue
        
        # Training update
        print(f"[Iteration {iteration}] Training on {len(all_transitions)} transitions")
        train_start_time = time.time()
        
        # Process transitions in batches
        np.random.shuffle(all_transitions)  # Shuffle transitions
        num_transitions = len(all_transitions)
        
        # Max batch size for GPU memory
        batches = [all_transitions[i:i+batch_size] for i in range(0, num_transitions, batch_size)]
        print(f"[Iteration {iteration}] Split into {len(batches)} batches of max size {batch_size}")
        
        # Batch training loop
        batch_losses = []
        for batch_idx, batch in enumerate(batches):
            # Prepare batch data
            batch_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in batch])
            batch_returns = th.tensor([t["return"] for t in batch], device="cuda")
            batch_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in batch])
            batch_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in batch])
            
            # Normalize advantages (important for training stability)
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            
            # Compute losses
            optimizer.zero_grad()
            
            with autocast():  # Mixed precision for faster training
                # Policy loss (Actor)
                policy_loss = -(batch_advantages * batch_log_probs).mean()
                
                # Value function loss (Critic)
                value_loss = ((batch_v_preds - batch_returns) ** 2).mean()
                
                # KL divergence loss (to prevent large policy updates)
                kl_losses = []
                for t in batch:
                    kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
                    kl_losses.append(kl_loss)
                
                kl_loss = th.stack(kl_losses).mean()
                
                # Total loss
                total_loss = policy_loss + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
            
            # Backpropagate and update
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            # Store batch loss
            batch_losses.append(total_loss.item())
            
            print(f"[Iteration {iteration}] Batch {batch_idx+1}/{len(batches)}: "
                  f"Loss={total_loss.item():.4f}, PolicyLoss={policy_loss.item():.4f}, "
                  f"ValueLoss={value_loss.item():.4f}, KLLoss={kl_loss.item():.4f}")
        
        # Update stats
        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        running_loss += avg_loss * len(all_transitions)
        total_steps += len(all_transitions)
        
        # Decay KL weight (as in original code)
        LAMBDA_KL *= KL_DECAY
        
        # Iteration summary
        iter_duration = time.time() - iter_start_time
        print(f"[Iteration {iteration}/{num_iterations}] "
              f"Complete: {len(all_transitions)} transitions, "
              f"AvgLoss={avg_loss:.4f}, RunningAvgLoss={running_loss/total_steps if total_steps > 0 else 0:.4f}, "
              f"Duration={iter_duration:.2f}s")
        
        # Save checkpoint (every 5 iterations or last iteration)
        if iteration % 5 == 0 or iteration == num_iterations:
            checkpoint_path = f"{out_weights}.iter{iteration}" if iteration != num_iterations else out_weights
            print(f"[Iteration {iteration}] Saving weights to {checkpoint_path}")
            th.save(agent.policy.state_dict(), checkpoint_path)
    
    # End of training
    print(f"Training complete: {total_rollouts} rollouts, {total_steps} total steps")
    
    # Save final weights if not already saved
    if num_iterations % 5 != 0:
        print(f"Saving final weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)
    
    # Signal processes to stop
    stop_flag.value = True
    
    # Wait for processes to terminate
    print("Waiting for environment processes to terminate...")
    for p in env_processes:
        p.join(timeout=5)
    
    # Close dummy environment
    dummy_env.close()
    
    print("Training successfully completed")


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
    parser.add_argument("--batch-size", required=False, type=int, default=128,
                        help="Maximum batch size for training updates")

    args = parser.parse_args()

    train_rl_multiprocessed(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        queue_size=args.queue_size,
        batch_size=args.batch_size
    )