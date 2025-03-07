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


# Thread for stepping through environments and collecting rollouts
def environment_thread(agent, envs, rollout_steps, rollout_queue, out_episodes, stop_flag):
    num_envs = len(envs)
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    visited_chunks_list = [set() for _ in range(num_envs)]
    iteration = 0
    while not stop_flag[0]:
        iteration += 1
        env_start_time = time.time()
        
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
            for env_i in range(num_envs):
                if env_i == 0:
                    envs[env_i].render()
                
                if not done_list[env_i]:
                    episode_step_counts[env_i] += 1
                    
                    # Prevent gradient tracking during rollout
                    with th.no_grad():
                        minerl_action_i, _, _, _, new_hid_i = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_i],
                            hidden_state=hidden_states[env_i],
                            stochastic=True,
                            taken_action=None
                        )
                    
                    next_obs_i, env_reward_i, done_flag_i, info_i = envs[env_i].step(minerl_action_i)
                    if "error" in info_i:
                        print(f"[Env {env_i}] Error in info: {info_i['error']}")
                        done_flag_i = True
                    custom_reward, visited_chunks_list[env_i] = custom_reward_function(
                            next_obs_i, done_flag_i, info_i, visited_chunks_list[env_i]
                    )
                    
                    # Apply death penalty
                    if done_flag_i:
                        custom_reward += -1000.0  # DEATH_PENALTY
                    
                    # Store rollout data
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(custom_reward)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    rollouts[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_i])
                    )
                    rollouts[env_i]["next_obs"].append(next_obs_i)
                    
                    # Update state
                    obs_list[env_i] = next_obs_i
                    hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid_i)
                    done_list[env_i] = done_flag_i
                    
                    if done_flag_i:
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_step_counts[env_i]}\n")
                        episode_step_counts[env_i] = 0
                        obs_list[env_i] = envs[env_i].reset()
                        done_list[env_i] = False
                        hidden_states[env_i] = agent.policy.initial_state(batch_size=1)

                        visited_chunks_list[env_i] = set()
                    else:
                        episode_step_counts[env_i] += 1
        
        # Send the collected rollouts to the training thread
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        print(f"[Environment Thread] Iteration {iteration} collected {rollout_steps} steps "
              f"across {num_envs} envs in {env_duration:.3f}s")
        rollout_queue.put(rollouts)


# Thread for training the agent - updated to process each environment separately
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
    """Training thread that processes each environment's rollout separately to maintain sequential integrity"""
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
        wait_start_time = time.time()
        rollouts = rollout_queue.get()
        wait_end_time = time.time()
        wait_duration = wait_end_time - wait_start_time
        print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts.")
        train_start_time = time.time()
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
        
        if env_count == 0:
            print(f"[Training Thread] No transitions collected, skipping update.")
            continue
        
        # Average losses for reporting
        avg_policy_loss = total_policy_loss / env_count
        avg_value_loss = total_value_loss / env_count
        avg_kl_loss = total_kl_loss / env_count
        
        # Backpropagate combined loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} took {train_duration:.3f}s "
            f"to process and train on {transitions_count} transitions from {env_count} environments.")
        
        # Update stats
        total_loss_val = total_loss.item()
        running_loss += total_loss_val
        total_steps += transitions_count
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        LAMBDA_KL *= KL_DECAY
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} "
            f"Loss={total_loss_val:.4f}, PolicyLoss={avg_policy_loss:.4f}, "
            f"ValueLoss={avg_value_loss:.4f}, KLLoss={avg_kl_loss:.4f}, "
            f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Queue={rollout_queue.qsize()}")


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    # print("sequence length (T): ", T)
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
            "cur_pd": cur_pd_t,     # Now a dictionary for this timestep
            "old_pd": old_pd_t,     # Now a dictionary for this timestep
            "next_obs": rollout["next_obs"][t]
        })

    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
            # No need to unpack - just pass the entire structure
            _, _, v_next, _, _ = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
        bootstrap_value = v_next.item()
    else:
        bootstrap_value = 0.0
        
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


# Process for running environment in a separate process
def env_process(env_id, task_queue, result_queue, stop_flag):
    """
    Process for running a Minecraft environment in a separate process
    
    Args:
        env_id: ID for this environment
        task_queue: Queue to receive tasks from main process
        result_queue: Queue to send results back to main process
        stop_flag: Shared flag to signal when to stop
    """
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize
    obs = env.reset()
    visited_chunks = set()
    episode_step_count = 0
    
    print(f"[Env Process {env_id}] Started")
    
    # Signal ready and send initial observation
    result_queue.put({"type": "ready", "env_id": env_id, "obs": obs})
    
    while not stop_flag.value:
        try:
            # Get task from queue
            task = task_queue.get(timeout=1.0)
            task_type = task.get("type")
            
            # Process task
            if task_type == "step":
                action = task["action"]
                
                # Step environment
                next_obs, env_reward, done, info = env.step(action)
                
                # Calculate custom reward
                custom_reward, visited_chunks = custom_reward_function(
                    next_obs, done, info, visited_chunks
                )
                
                # Apply death penalty if done
                if done:
                    custom_reward -= 1000.0
                    
                # Update step count
                episode_step_count += 1
                
                # Render (only for env 0)
                if env_id == 0 and episode_step_count % 10 == 0:
                    env.render()
                
                # Send result back
                result_queue.put({
                    "type": "step_result",
                    "env_id": env_id,
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": action,
                    "reward": custom_reward,
                    "done": done,
                    "info": info
                })
                
                # Handle done
                if done:
                    result_queue.put({
                        "type": "episode_done",
                        "env_id": env_id,
                        "length": episode_step_count
                    })
                    
                    # Reset environment
                    obs = env.reset()
                    visited_chunks = set()
                    episode_step_count = 0
                else:
                    # Update observation
                    obs = next_obs
                    
            elif task_type == "reset":
                # Reset environment
                obs = env.reset()
                visited_chunks = set()
                episode_step_count = 0
                
                # Send result
                result_queue.put({
                    "type": "reset_done",
                    "env_id": env_id,
                    "obs": obs
                })
                
            elif task_type == "exit":
                # Exit process
                break
                
        except queue.Empty:
            # No tasks, just continue
            continue
            
        except Exception as e:
            import traceback
            print(f"[Env Process {env_id}] Error: {e}")
            print(traceback.format_exc())
            
            # Try to recover
            try:
                obs = env.reset()
                visited_chunks = set()
                episode_step_count = 0
            except:
                pass
    
    # Clean up
    env.close()
    print(f"[Env Process {env_id}] Stopped")


# Environment coordination thread for multiprocessing version
def environment_thread_mp(agent, rollout_steps, task_queues, result_queue, rollout_queue, out_episodes, stop_flag, num_envs):
    """
    Thread that coordinates multiple environment processes
    """
    # Wait for environments to initialize
    print(f"[Environment Thread] Waiting for {num_envs} environments to be ready")
    obs_list = [None] * num_envs
    ready_envs = 0
    
    # Wait for initial observations
    timeout_start = time.time()
    while ready_envs < num_envs and time.time() - timeout_start < 30.0:
        try:
            result = result_queue.get(timeout=1.0)
            
            if result["type"] == "ready":
                env_id = result["env_id"]
                obs_list[env_id] = result["obs"]
                ready_envs += 1
                print(f"[Environment Thread] Environment {env_id} is ready ({ready_envs}/{num_envs})")
                
        except queue.Empty:
            continue
    
    if ready_envs < num_envs:
        print(f"[Environment Thread] Warning: Only {ready_envs}/{num_envs} environments ready. Continuing anyway.")
    
    # Initialize state tracking
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    # Main loop
    iteration = 0
    while not stop_flag[0]:
        iteration += 1
        env_start_time = time.time()
        
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
            # Send actions to all environments
            for env_i in range(num_envs):
                if obs_list[env_i] is not None:
                    try:
                        # Get action using agent
                        with th.no_grad():
                            minerl_action, _, _, _, new_hid = agent.get_action_and_training_info(
                                minerl_obs=obs_list[env_i],
                                hidden_state=hidden_states[env_i],
                                stochastic=True,
                                taken_action=None
                            )
                        
                        # Store updated hidden state
                        hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid)
                        
                        # Send step task to environment
                        task_queues[env_i].put({
                            "type": "step",
                            "action": minerl_action
                        })
                        
                    except Exception as e:
                        print(f"[Environment Thread] Error generating action for env {env_i}: {e}")
            
            # Collect results from all environments
            for env_i in range(num_envs):
                if obs_list[env_i] is not None:
                    try:
                        # Wait for result from environment
                        result = result_queue.get(timeout=5.0)
                        
                        # Process different result types
                        if result["type"] == "step_result":
                            env_id = result["env_id"]
                            
                            # Store in rollout
                            rollouts[env_id]["obs"].append(result["obs"])
                            rollouts[env_id]["actions"].append(result["action"])
                            rollouts[env_id]["rewards"].append(result["reward"])
                            rollouts[env_id]["dones"].append(result["done"])
                            rollouts[env_id]["hidden_states"].append(
                                tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id])
                            )
                            rollouts[env_id]["next_obs"].append(result["next_obs"])
                            
                            # Update state
                            obs_list[env_id] = result["next_obs"]
                            done_list[env_id] = result["done"]
                            
                        elif result["type"] == "episode_done":
                            # Log episode completion
                            env_id = result["env_id"]
                            episode_length = result["length"]
                            
                            with open(out_episodes, "a") as f:
                                f.write(f"{episode_length}\n")
                                
                            # Reset hidden state
                            hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                            
                        elif result["type"] == "reset_done":
                            # Handle reset result
                            env_id = result["env_id"]
                            obs_list[env_id] = result["obs"]
                            
                    except queue.Empty:
                        print(f"[Environment Thread] Timeout waiting for result from env {env_i}")
                        break
            
            # Log progress
            if step % 10 == 0:
                print(f"[Environment Thread] Completed step {step+1}/{rollout_steps}")
        
        # Send completed rollouts to training thread
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        
        # Calculate total transitions
        total_transitions = sum(len(rollout["obs"]) for rollout in rollouts)
        
        print(f"[Environment Thread] Iteration {iteration} collected {total_transitions} transitions "
              f"across {num_envs} envs in {env_duration:.3f}s")
        
        rollout_queue.put(rollouts)


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
    Multiprocessing version with separate processes for environment stepping and sequential training
    """
    # Set spawn method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set")
    
    # Create environments and agents
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent for main thread
    agent = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    # Create pretrained policy for KL divergence calculation
    pretrained_policy = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    # Create communication channels
    stop_flag = Value(ctypes.c_bool, False)
    task_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    # Start environment processes
    env_processes = []
    for env_i in range(num_envs):
        p = Process(
            target=env_process,
            args=(env_i, task_queues[env_i], result_queue, stop_flag)
        )
        p.daemon = True
        p.start()
        env_processes.append(p)
    
    # Give processes time to start
    time.sleep(2)
    
    # Thread stop flag
    thread_stop = [False]
    
    # Create and start threads
    env_thread = threading.Thread(
        target=environment_thread_mp,
        args=(
            agent,
            rollout_steps,
            task_queues,
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
        print("Interrupted by user. Stopping threads...")
    finally:
        # Signal threads and processes to stop
        print("Setting stop flags...")
        thread_stop[0] = True
        stop_flag.value = True
        
        # Send exit signal to all environment processes
        for q in task_queues:
            try:
                q.put({"type": "exit"})
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
        
        # Wait for processes to finish
        print("Waiting for processes to finish...")
        for i, p in enumerate(env_processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"Warning: Environment process {i} did not terminate properly")
                p.terminate()
        
        # Close environments
        for env in envs:
            env.close()
        
        # Save weights
        print(f"Saving fine-tuned weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--num-iterations", required=False, type=int, default=10)
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=2)
    parser.add_argument("--queue-size", required=False, type=int, default=3,
                        help="Size of the queue between environment and training threads")
    parser.add_argument("--use-mp", required=False, action="store_true", default=False,
                        help="Use multiprocessing instead of threading")

    args = parser.parse_args()

    if args.use_mp:
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
    else:
        train_rl_threaded(
            in_model=args.in_model,
            in_weights=args.in_weights,
            out_weights=args.out_weights,
            out_episodes=args.out_episodes,
            num_iterations=args.num_iterations,
            rollout_steps=args.rollout_steps,
            num_envs=args.num_envs,
            queue_size=args.queue_size
        )
        dummy_env.close()
        
        # Save weights
        print(f"Saving fine-tuned weights to {out_weights}")
        th.save(agent.policy.state_dict(), out_weights)


def train_rl_threaded(
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
    Threading version with separate threads for environment stepping and training
    """
    # Create environments and agents
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    agent = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)
    
    pretrained_policy = MineRLAgent(
        dummy_env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    
    # Create environments
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    
    # Create a queue for passing rollouts between threads
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    # Shared flag to signal threads to stop
    stop_flag = [False]
    
    # Create and start threads
    env_thread = threading.Thread(
        target=environment_thread,
        args=(agent, envs, rollout_steps, rollout_queue, out_episodes, stop_flag)
    )
    
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations)
    )
    
    print("Starting threads...")
    env_thread.start()
    train_thread.start()
    
    try:
        # Wait for training thread to complete
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping threads...")
    finally:
        # Signal threads to stop
        stop_flag[0] = True
        
        # Wait for threads to finish
        env_thread.join(timeout=10)
        if env_thread.is_alive():
            print("Warning: Environment thread did not terminate properly")
        
        # Close environments