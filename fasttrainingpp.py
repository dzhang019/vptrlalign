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


# Process for running environment steps and collecting rollouts
def environment_process(env_id, rollouts_queue, ready_queue, data_queue, stop_flag):
    """
    Process function for running a single environment
    
    Args:
        env_id: ID for this environment
        rollouts_queue: Queue to receive rollout data from main process
        ready_queue: Queue to signal readiness to main process
        data_queue: Queue to send collected data back to main process
        stop_flag: Shared flag to signal process to stop
    """
    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()
    
    # Initialize
    obs = env.reset()
    visited_chunks = set()
    episode_step_count = 0
    
    print(f"[Env Process {env_id}] Started")
    
    # Signal ready
    ready_queue.put(env_id)
    
    while not stop_flag.value:
        try:
            # Wait for rollout data from main process
            rollout_data = rollouts_queue.get(timeout=1.0)
            
            # Unpack data
            obs_list = rollout_data["obs"]
            done = rollout_data["done"]
            hidden_state = rollout_data["hidden_state"]
            action = rollout_data["action"]
            
            # If action is None, this is a reset signal
            if action is None:
                obs = env.reset()
                visited_chunks = set()
                episode_step_count = 0
                data_queue.put({
                    "env_id": env_id,
                    "type": "reset_done",
                    "obs": obs
                })
                continue
            
            # Take step in environment
            next_obs, env_reward, done_flag, info = env.step(action)
            
            # Handle errors
            if "error" in info:
                print(f"[Env Process {env_id}] Error in info: {info['error']}")
                done_flag = True
            
            # Calculate custom reward
            custom_reward, visited_chunks = custom_reward_function(
                next_obs, done_flag, info, visited_chunks
            )
            
            # Apply death penalty
            if done_flag:
                custom_reward += -1000.0  # DEATH_PENALTY
                
            # Update episode step count
            episode_step_count += 1
            
            # Send result back to main process
            data_queue.put({
                "env_id": env_id,
                "type": "step_result",
                "obs": obs,
                "next_obs": next_obs,
                "reward": custom_reward,
                "done": done_flag,
                "info": info,
                "episode_step_count": episode_step_count if done_flag else None
            })
            
            # Update state
            obs = next_obs
            
            # If done, reset environment
            if done_flag:
                episode_step_count = 0
                obs = env.reset()
                visited_chunks = set()
            
            # Optionally render
            if env_id == 0:  # Only render first environment
                env.render()
                
        except queue.Empty:
            # Timeout waiting for command - just continue
            continue
    
    # Clean up
    print(f"[Env Process {env_id}] Stopping")
    env.close()


# Thread for stepping through environments and collecting rollouts
def environment_thread(agent, rollout_steps, rollouts_queues, ready_queue, data_queue, rollout_queue, out_episodes, stop_flag, num_envs):
    """
    Coordinates environment processes and collects rollouts
    
    Args:
        agent: MineRLAgent for action generation
        rollout_steps: Number of steps per rollout
        rollouts_queues: List of queues to send rollout data to env processes
        ready_queue: Queue to receive ready signals from env processes
        data_queue: Queue to receive step results from env processes
        rollout_queue: Queue to send completed rollouts to training thread
        out_episodes: Path to file for writing episode lengths
        stop_flag: Flag to signal thread to stop
        num_envs: Number of environments
    """
    # Initialize tracking variables
    obs_list = [None] * num_envs
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    # Wait for all environments to be ready
    ready_envs = 0
    while ready_envs < num_envs:
        try:
            env_id = ready_queue.get(timeout=10.0)
            ready_envs += 1
            print(f"[Environment Thread] Environment {env_id} is ready")
        except queue.Empty:
            print("[Environment Thread] Timeout waiting for environments to be ready")
            if ready_envs < num_envs:
                print(f"[Environment Thread] Only {ready_envs}/{num_envs} environments are ready")
                # Continue anyway
                break
    
    # Get initial observations
    for env_i in range(num_envs):
        # Send reset command to each environment
        rollouts_queues[env_i].put({
            "obs": None,
            "done": True,
            "hidden_state": None,
            "action": None  # None action signals reset
        })
    
    # Wait for initial observations
    for _ in range(num_envs):
        try:
            result = data_queue.get(timeout=10.0)
            if result["type"] == "reset_done":
                env_i = result["env_id"]
                obs_list[env_i] = result["obs"]
                print(f"[Environment Thread] Got initial observation from env {env_i}")
        except queue.Empty:
            print("[Environment Thread] Timeout waiting for initial observations")
    
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
            # Get actions for all environments and send step commands
            step_start_time = time.time()
            action_requests = []
            
            for env_i in range(num_envs):
                if obs_list[env_i] is not None and not done_list[env_i]:
                    # Get action from agent
                    with th.no_grad():
                        minerl_action_i, _, _, _, new_hid_i = agent.get_action_and_training_info(
                            minerl_obs=obs_list[env_i],
                            hidden_state=hidden_states[env_i],
                            stochastic=True,
                            taken_action=None
                        )
                    
                    # Send action to environment process
                    rollouts_queues[env_i].put({
                        "obs": obs_list[env_i],
                        "done": done_list[env_i],
                        "hidden_state": hidden_states[env_i],
                        "action": minerl_action_i
                    })
                    
                    # Store hidden state (will be updated after step results)
                    hidden_states[env_i] = tree_map(lambda x: x.detach(), new_hid_i)
                    action_requests.append(env_i)
            
            # Collect step results
            for _ in range(len(action_requests)):
                try:
                    result = data_queue.get(timeout=10.0)
                    
                    if result["type"] == "step_result":
                        env_i = result["env_id"]
                        
                        # Store rollout data
                        rollouts[env_i]["obs"].append(result["obs"])
                        rollouts[env_i]["actions"].append(result["action"])
                        rollouts[env_i]["rewards"].append(result["reward"])
                        rollouts[env_i]["dones"].append(result["done"])
                        rollouts[env_i]["hidden_states"].append(
                            tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_i])
                        )
                        rollouts[env_i]["next_obs"].append(result["next_obs"])
                        
                        # Update state
                        obs_list[env_i] = result["next_obs"]
                        done_list[env_i] = result["done"]
                        
                        # Handle episode termination
                        if result["done"]:
                            if result["episode_step_count"] is not None:
                                with open(out_episodes, "a") as f:
                                    f.write(f"{result['episode_step_count']}\n")
                            
                            # Reset done flag since environment auto-resets
                            done_list[env_i] = False
                            
                            # Reset hidden state
                            hidden_states[env_i] = agent.policy.initial_state(batch_size=1)
                except queue.Empty:
                    print(f"[Environment Thread] Timeout waiting for step results in step {step}")
                    break
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if step % 10 == 0:  # Log every 10 steps
                print(f"[Environment Thread] Step {step}/{rollout_steps} took {step_duration:.3f}s")
        
        # Send the collected rollouts to the training thread
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        print(f"[Environment Thread] Iteration {iteration} collected {rollout_steps} steps "
              f"across {num_envs} envs in {env_duration:.3f}s")
        
        # Count total valid transitions
        total_transitions = sum(len(r["obs"]) for r in rollouts)
        print(f"[Environment Thread] Collected {total_transitions} total transitions")
        
        rollout_queue.put(rollouts)


# Thread for training the agent
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
    """Training thread with batched processing for better GPU utilization"""
    # Hyperparameters - keeping the same as the original
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
        
        # Process rollouts
        transitions_all = []
        for env_i, env_rollout in enumerate(rollouts):
            if len(env_rollout["obs"]) == 0:
                print(f"[Training Thread] Environment {env_i} has no transitions, skipping")
                continue
                
            env_transitions = train_unroll(
                agent,
                pretrained_policy,
                env_rollout,
                gamma=GAMMA,
                lam=LAM
            )
            transitions_all.extend(env_transitions)
        
        if len(transitions_all) == 0:
            print(f"[Training Thread] No transitions collected, skipping update.")
            continue
        
        # Batch processing - just like the original
        batch_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) for t in transitions_all])
        batch_returns = th.tensor([t["return"] for t in transitions_all], device="cuda")
        batch_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in transitions_all])
        batch_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in transitions_all])
        
        # Compute losses in batch
        optimizer.zero_grad()
        
        # Policy loss (using negative log probability * advantages)
        with autocast():
            policy_loss = -(batch_advantages * batch_log_probs).mean()
            
            # Value function loss
            value_loss = ((batch_v_preds - batch_returns) ** 2).mean()
            
            # KL divergence loss - this needs to be handled separately
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
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} took {train_duration:.3f}s "
            f"to process and train on {len(transitions_all)} transitions.")
        
        # Update stats
        total_loss_val = total_loss.item()
        running_loss += total_loss_val * len(transitions_all)
        total_steps += len(transitions_all)
        avg_loss = (running_loss / total_steps) if total_steps > 0 else 0.0
        LAMBDA_KL *= KL_DECAY
        
        print(f"[Training Thread] Iteration {iteration}/{num_iterations} "
            f"Loss={total_loss_val:.4f}, PolicyLoss={policy_loss.item():.4f}, "
            f"ValueLoss={value_loss.item():.4f}, KLLoss={kl_loss.item():.4f}, "
            f"StepsSoFar={total_steps}, AvgLoss={avg_loss:.4f}, Queue={rollout_queue.qsize()}")


# Keep train_unroll function exactly the same as the original
def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    """Process a rollout into transitions for training - exactly as in original"""
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

    # Bootstrap value calculation - same as original
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
    
    # GAE calculation - same as original
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
    Multiprocessing version with separate processes for environment stepping and training
    """
    # Set spawn method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create environments and agents
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent for main thread (will be used by env thread for action generation)
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
    
    # Create queues for inter-process communication
    rollouts_queues = [Queue() for _ in range(num_envs)]  # Main -> Env processes
    ready_queue = Queue()  # Env processes -> Main (ready signals)
    data_queue = Queue()   # Env processes -> Main (step results)
    rollout_queue = RolloutQueue(maxsize=queue_size)  # Env thread -> Training thread
    
    # Shared flag to signal processes to stop
    stop_flag = Value(ctypes.c_bool, False)
    
    # Create and start environment processes
    env_processes = []
    for env_i in range(num_envs):
        p = Process(
            target=environment_process,
            args=(
                env_i,
                rollouts_queues[env_i],
                ready_queue,
                data_queue,
                stop_flag
            )
        )
        p.daemon = True
        p.start()
        env_processes.append(p)
    
    # Wait for processes to initialize
    print(f"Waiting for {num_envs} environment processes to initialize...")
    time.sleep(2)
    
    # Create and start threads
    env_thread = threading.Thread(
        target=environment_thread,
        args=(
            agent,
            rollout_steps,
            rollouts_queues,
            ready_queue,
            data_queue,
            rollout_queue,
            out_episodes,
            [stop_flag.value],  # Using a list to allow thread to modify
            num_envs
        )
    )
    
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained_policy, rollout_queue, [stop_flag.value], num_iterations)
    )
    
    print("Starting threads...")
    env_thread.start()
    train_thread.start()
    
    try:
        # Wait for training thread to complete
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping threads and processes...")
    finally:
        # Signal threads and processes to stop
        stop_flag.value = True
        
        # Wait for threads to finish
        env_thread.join(timeout=10)
        if env_thread.is_alive():
            print("Warning: Environment thread did not terminate properly")
        
        train_thread.join(timeout=5)
        if train_thread.is_alive():
            print("Warning: Training thread did not terminate properly")
        
        # Wait for processes to finish
        for p in env_processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Warning: Environment process {env_processes.index(p)} did not terminate properly")
                p.terminate()
        
        # Close dummy environment
        dummy_env.close()
        
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