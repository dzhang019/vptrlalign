from argparse import ArgumentParser
import pickle
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import ctypes

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from lib.tree_util import tree_map
from lib.reward_structure_mod import custom_reward_function
from lib.policy_mod import compute_kl_loss
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


# Environment worker process function
def env_worker(env_id, action_queue, result_queue, stop_flag):
    """
    Worker process for running a single environment
    
    Args:
        env_id: ID for this environment
        action_queue: Queue to receive actions from main process
        result_queue: Queue to send results back to main process
        stop_flag: Shared flag to signal process to stop
    """
    try:
        # Create environment
        env = HumanSurvival(**ENV_KWARGS).make()
        
        # Initialize state
        obs = env.reset()
        visited_chunks = set()
        episode_step_count = 0
        
        print(f"[Env {env_id}] Started")
        
        # Send initial observation
        result_queue.put((env_id, None, obs, False, 0, None))
        
        while not stop_flag.value:
            try:
                # Get action from main process
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
                    custom_reward -= 1000.0
                    
                # Update counter
                episode_step_count += 1
                
                # Send results back to main process
                result_queue.put((env_id, action, next_obs, done, custom_reward, info))
                
                # Render occasionally for visualization (only first environment)
                if env_id == 0 and episode_step_count % 10 == 0:
                    env.render()
                
                # Reset if episode done
                if done:
                    # Send episode completion signal
                    result_queue.put((env_id, None, None, True, episode_step_count, None))
                    
                    # Reset environment
                    obs = env.reset()
                    visited_chunks = set()
                    episode_step_count = 0
                    
                    # Send new observation
                    result_queue.put((env_id, None, obs, False, 0, None))
                else:
                    # Update observation
                    obs = next_obs
                    
            except queue.Empty:
                # Timeout waiting for action, just continue
                continue
                
            except Exception as e:
                import traceback
                print(f"[Env {env_id}] Error: {e}")
                print(traceback.format_exc())
                
                # Try to recover by resetting
                try:
                    obs = env.reset()
                    visited_chunks = set()
                    episode_step_count = 0
                    result_queue.put((env_id, None, obs, False, 0, None))
                except:
                    pass
                    
    except Exception as e:
        import traceback
        print(f"[Env {env_id}] Critical error: {e}")
        print(traceback.format_exc())
    
    finally:
        # Clean up
        try:
            env.close()
        except:
            pass
            
        print(f"[Env {env_id}] Stopped")


# Thread for coordinating environments and collecting rollouts
def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, out_episodes, stop_flag, num_envs):
    """
    Thread that coordinates multiple environment processes and collects rollouts
    
    Args:
        agent: The agent model for action generation
        rollout_steps: Number of steps per rollout
        action_queues: List of queues to send actions to each environment
        result_queue: Queue to receive results from environments
        rollout_queue: Queue to send completed rollouts to training thread
        out_episodes: Path to file for episode length logging
        stop_flag: Flag to signal thread to stop
        num_envs: Number of environments
    """
    # Initialize state tracking for each environment
    obs_list = [None] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    # Wait for initial observations from all environments
    print(f"[Environment Thread] Waiting for initial observations from {num_envs} environments")
    envs_ready = 0
    while envs_ready < num_envs:
        try:
            env_id, _, obs, _, _, _ = result_queue.get(timeout=10.0)
            obs_list[env_id] = obs
            envs_ready += 1
            print(f"[Environment Thread] Got initial observation from env {env_id} ({envs_ready}/{num_envs})")
        except queue.Empty:
            print(f"[Environment Thread] Timeout waiting for initial observations ({envs_ready}/{num_envs})")
            break
    
    # Check if all environments provided observations
    missing_obs = [i for i, obs in enumerate(obs_list) if obs is None]
    if missing_obs:
        print(f"[Environment Thread] Warning: Missing initial observations from environments: {missing_obs}")
    
    # Main loop
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
        
        print(f"[Environment Thread] Starting iteration {iteration}, collecting {rollout_steps} steps per env")
        
        # Collect rollouts
        for step in range(rollout_steps):
            # Generate and send actions for all environments
            for env_id in range(num_envs):
                if obs_list[env_id] is not None:
                    try:
                        # Generate action using agent
                        with th.no_grad():
                            minerl_action, _, _, _, new_hid = agent.get_action_and_training_info(
                                minerl_obs=obs_list[env_id],
                                hidden_state=hidden_states[env_id],
                                stochastic=True,
                                taken_action=None
                            )
                        
                        # Update hidden state
                        hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                        
                        # Send action to environment
                        action_queues[env_id].put(minerl_action)
                    except Exception as e:
                        print(f"[Environment Thread] Error generating action for env {env_id}: {e}")
            
            # Collect results from all environments
            results_received = 0
            expected_results = sum(1 for obs in obs_list if obs is not None)
            
            if expected_results == 0:
                print("[Environment Thread] No valid environments to step")
                time.sleep(0.1)
                continue
                
            while results_received < expected_results:
                try:
                    env_id, action, next_obs, done, reward, info = result_queue.get(timeout=5.0)
                    
                    # Handle different types of messages
                    if action is None and done and next_obs is None:
                        # Episode completion message
                        episode_length = reward  # reward field contains episode length
                        try:
                            with open(out_episodes, "a") as f:
                                f.write(f"{episode_length}\n")
                        except Exception as e:
                            print(f"[Environment Thread] Error writing episode length: {e}")
                        continue
                    
                    elif action is None and not done:
                        # Initial observation or reset message
                        obs_list[env_id] = next_obs
                        continue
                    
                    else:
                        # Normal step result
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
                        
                        # Reset hidden state if episode ended
                        if done:
                            hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                        
                        # Count as processed result
                        results_received += 1
                
                except queue.Empty:
                    print(f"[Environment Thread] Timeout waiting for results in step {step}")
                    break
            
            # Log progress occasionally
            if step % 10 == 0 or step == rollout_steps - 1:
                print(f"[Environment Thread] Completed step {step+1}/{rollout_steps}")
        
        # End of rollout collection
        end_time = time.time()
        duration = end_time - start_time
        
        # Count total transitions
        total_transitions = sum(len(r["obs"]) for r in rollouts)
        
        print(f"[Environment Thread] Iteration {iteration} collected {total_transitions} transitions "
              f"across {num_envs} environments in {duration:.3f}s")
        
        # Skip empty rollouts
        if total_transitions == 0:
            print("[Environment Thread] No transitions collected, skipping update")
            continue
        
        # Send rollouts to training thread
        rollout_queue.put(rollouts)


# Thread for training the agent
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
    """
    Thread for processing rollouts and training the agent
    
    This implementation maintains sequential integrity of each environment
    by processing them separately during training.
    
    Args:
        agent: The agent model to train
        pretrained_policy: Reference policy for KL divergence
        rollout_queue: Queue to receive rollouts from environment thread
        stop_flag: Flag to signal thread to stop
        num_iterations: Maximum number of training iterations
    """
    # Hyperparameters
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
    iteration = 0
    scaler = GradScaler()
    
    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        
        print(f"[Training Thread] Waiting for rollouts...")
        wait_start = time.time()
        rollouts = rollout_queue.get()
        wait_duration = time.time() - wait_start
        print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts")
        
        train_start = time.time()
        print(f"[Training Thread] Processing rollouts for iteration {iteration}")
        
        # Process each environment's rollout separately, but accumulate gradients
        optimizer.zero_grad()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_loss = 0
        env_count = 0
        transitions_count = 0
        
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
                
            # Process transitions for this environment
            env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in env_transitions])
            env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
            env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
            env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
            
            # Normalize advantages for this environment
            env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
            
            # Compute losses for this environment
            with autocast():
                # Policy loss
                policy_loss = -(env_advantages * env_log_probs).mean()
                
                # Value function loss
                value_loss = ((env_v_preds - env_returns) ** 2).mean()
                
                # KL divergence loss
                kl_losses = []
                for t in env_transitions:
                    kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
                    kl_losses.append(kl_loss)
                kl_loss = th.stack(kl_losses).mean()
                
                # Total loss for this environment
                env_loss = policy_loss + (VALUE_LOSS_COEF * value_loss) + (LAMBDA_KL * kl_loss)
            
            # Accumulate losses
            total_loss += env_loss
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl_loss += kl_loss.item()
            env_count += 1
            transitions_count += len(env_transitions)
        
        # Skip update if no valid transitions
        if env_count == 0:
            print(f"[Training Thread] No valid transitions, skipping update")
            continue
        
        # Average losses
        avg_policy_loss = total_policy_loss / env_count
        avg_value_loss = total_value_loss / env_count
        avg_kl_loss = total_kl_loss / env_count
        
        # Backpropagate combined loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        train_end = time.time()
        train_duration = train_end - train_start
        
        # Update stats
        total_loss_val = total_loss.item()
        running_loss += total_loss_val
        total_steps += transitions_count
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        LAMBDA_KL *= KL_DECAY
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} took {train_duration:.3f}s "
              f"to process and train on {transitions_count} transitions from {env_count} environments")
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} "
              f"Loss={total_loss_val:.4f}, PolicyLoss={avg_policy_loss:.4f}, "
              f"ValueLoss={avg_value_loss:.4f}, KLLoss={avg_kl_loss:.4f}, "
              f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Queue={rollout_queue.qsize()}")


# Keep train_unroll function exactly the same as the original
def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """Process a rollout into transitions for training - unchanged from original"""
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    hidden_states_seq = rollout["hidden_states"]

    pi_dist_seq, vpred_seq, log_prob_seq, final_hid = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=hidden_states_seq[0],
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

    # Bootstrap value calculation
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
    Train an RL agent using multiprocessing for environment parallelism
    
    Args:
        in_model: Path to input model parameters file
        in_weights: Path to input weights file
        out_weights: Path to output weights file
        out_episodes: Path to episode length logging file
        num_iterations: Number of training iterations
        rollout_steps: Number of steps per rollout
        num_envs: Number of parallel environments
        queue_size: Size of rollout queue between environment and training threads
    """
    # Set spawn method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set")
    
    print(f"Starting training with {num_envs} environments, {rollout_steps} steps per rollout")
    
    # Create dummy environment for initialization
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent
    print("Creating agent...")
    agent = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    # Create pretrained policy for KL divergence
    print("Creating pretrained policy...")
    pretrained_policy = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    # Create communication channels
    print("Setting up communication channels...")
    stop_flag = Value(ctypes.c_bool, False)
    action_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    # Thread stop flag (Python list for thread access)
    thread_stop = [False]
    
    # Start environment worker processes
    print("Starting environment processes...")
    workers = []
    for env_id in range(num_envs):
        p = Process(
            target=env_worker,
            args=(env_id, action_queues[env_id], result_queue, stop_flag)
        )
        p.daemon = True
        p.start()
        workers.append(p)
        print(f"Started environment process {env_id}, pid: {p.pid}")
    
    # Wait for processes to initialize
    time.sleep(2)
    
    # Create and start threads
    print("Starting coordinator threads...")
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
    
    env_thread.start()
    train_thread.start()
    
    try:
        # Wait for training thread to complete
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping threads and processes...")
    finally:
        # Signal threads and processes to stop
        print("Setting stop flags...")
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
        if env_thread.is_alive():
            print("Warning: Environment thread did not terminate properly")
        
        train_thread.join(timeout=5)
        if train_thread.is_alive():
            print("Warning: Training thread did not terminate properly")
        
        # Wait for workers to finish
        print("Waiting for worker processes to finish...")
        for i, p in enumerate(workers):
            p.join(timeout=5)
            if p.is_alive():
                print(f"Worker {i} did not terminate, force terminating...")
                p.terminate()
        
        # Close dummy environment
        dummy_env.close()
        
        # Save weights
        print(f"Saving weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)
        print("Done!")


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