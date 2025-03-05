from argparse import ArgumentParser
import pickle
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from data_loader import DataLoader
from lib.tree_util import tree_map

from lib.height import reward_function
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


def env_worker(env_id, action_queue, result_queue, stop_event, model_file, weights_file):
    """
    Worker process for running a single environment
    
    Args:
        env_id: Unique ID for this environment
        action_queue: Queue to receive actions from main process
        result_queue: Queue to send results back to main process
        stop_event: Event to signal worker to stop
        model_file: Path to model file for agent creation
        weights_file: Path to weights file for agent initialization
    """
    try:
        # Create environment
        env = HumanSurvival(**ENV_KWARGS).make()
        
        # Create a lightweight agent for this process
        dummy_env = HumanSurvival(**ENV_KWARGS).make()
        policy_kwargs, pi_head_kwargs = load_model_parameters(model_file)
        agent = MineRLAgent(
            dummy_env, device="cuda",
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs
        )
        agent.load_weights(weights_file)
        
        # Initialize environment
        obs = env.reset()
        hidden_state = agent.policy.initial_state(batch_size=1)
        done = False
        episode_step_count = 0
        
        while not stop_event.is_set():
            # Get command from main process
            try:
                cmd = action_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Process command
            if cmd["type"] == "step":
                # Take a step in the environment
                episode_step_count += 1
                
                # Get action from agent
                with th.no_grad():
                    minerl_action, _, _, _, new_hidden_state = agent.get_action_and_training_info(
                        minerl_obs=obs,
                        hidden_state=hidden_state,
                        stochastic=True,
                        taken_action=None
                    )
                
                # Step environment
                next_obs, reward, done, info = env.step(minerl_action)
                
                # Check for errors
                if "error" in info:
                    print(f"[Env {env_id}] Error in info: {info['error']}")
                    done = True
                
                # Apply death penalty if done
                if done:
                    reward += -1000.0  # DEATH_PENALTY
                
                # Send result back to main process
                result = {
                    "type": "step_result",
                    "env_id": env_id,
                    "obs": obs,
                    "action": minerl_action,
                    "reward": reward,
                    "done": done,
                    "next_obs": next_obs,
                    "hidden_state": tree_map(lambda x: x.detach().cpu().contiguous(), hidden_state)
                }
                result_queue.put(result)
                
                # Update state
                obs = next_obs
                hidden_state = tree_map(lambda x: x.detach(), new_hidden_state)
                
                # Reset if done
                if done:
                    result_queue.put({
                        "type": "episode_done",
                        "env_id": env_id,
                        "episode_length": episode_step_count
                    })
                    obs = env.reset()
                    hidden_state = agent.policy.initial_state(batch_size=1)
                    done = False
                    episode_step_count = 0
            
            elif cmd["type"] == "reset":
                # Reset environment
                obs = env.reset()
                hidden_state = agent.policy.initial_state(batch_size=1)
                done = False
                episode_step_count = 0
                result_queue.put({
                    "type": "reset_done",
                    "env_id": env_id
                })
                
            elif cmd["type"] == "render":
                # Render environment (only for visualization)
                env.render()
                result_queue.put({
                    "type": "render_done",
                    "env_id": env_id
                })
                
            elif cmd["type"] == "exit":
                # Exit worker
                break
    
    except Exception as e:
        import traceback
        print(f"Error in environment worker {env_id}: {e}")
        print(traceback.format_exc())
    
    finally:
        # Clean up
        try:
            env.close()
            dummy_env.close()
        except:
            pass


def environment_thread_mp(action_queues, result_queue, rollout_steps, rollout_queue, out_episodes, stop_flag, stop_event):
    """
    Thread for coordinating multiple environment processes
    """
    num_envs = len(action_queues)
    episode_step_counts = [0] * num_envs
    
    # Initialize rollouts dictionary to store data from each environment
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
    
    # Make sure all environments are ready
    for env_i in range(num_envs):
        action_queues[env_i].put({"type": "reset"})
    
    for _ in range(num_envs):
        result = result_queue.get()
        if result["type"] != "reset_done":
            print(f"Warning: Unexpected result during initialization: {result['type']}")
    
    iteration = 0
    while not stop_flag[0]:
        iteration += 1
        env_start_time = time.time()
        
        # Clear rollouts
        for env_i in range(num_envs):
            rollouts[env_i] = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "hidden_states": [],
                "next_obs": []
            }
        
        # Collect rollouts
        results_pending = 0
        
        for step in range(rollout_steps):
            # Signal all environments to take a step
            for env_i in range(num_envs):
                action_queues[env_i].put({"type": "step"})
                results_pending += 1
            
            # Request render for first environment only (visualization)
            if step == 0:  # Render first step of each rollout
                action_queues[0].put({"type": "render"})
                results_pending += 1
            
            # Collect all pending results
            while results_pending > 0:
                try:
                    result = result_queue.get(timeout=10.0)
                    results_pending -= 1
                    
                    if result["type"] == "step_result":
                        env_i = result["env_id"]
                        
                        # Store rollout data
                        rollouts[env_i]["obs"].append(result["obs"])
                        rollouts[env_i]["actions"].append(result["action"])
                        rollouts[env_i]["rewards"].append(result["reward"])
                        rollouts[env_i]["dones"].append(result["done"])
                        rollouts[env_i]["hidden_states"].append(result["hidden_state"])
                        rollouts[env_i]["next_obs"].append(result["next_obs"])
                    
                    elif result["type"] == "episode_done":
                        env_i = result["env_id"]
                        episode_length = result["episode_length"]
                        with open(out_episodes, "a") as f:
                            f.write(f"{episode_length}\n")
                        episode_step_counts[env_i] = 0
                    
                    # Ignore render_done results
                    
                except queue.Empty:
                    print(f"Warning: Timeout waiting for environment results (pending: {results_pending})")
                    # Re-send step commands to all environments as a recovery mechanism
                    for env_i in range(num_envs):
                        action_queues[env_i].put({"type": "step"})
        
        # Send the collected rollouts to the training thread
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        print(f"[Environment Thread] Iteration {iteration} collected {rollout_steps} steps "
              f"across {num_envs} envs in {env_duration:.3f}s")
        rollout_queue.put(rollouts)
    
    # Signal all environment processes to exit
    for env_i in range(num_envs):
        action_queues[env_i].put({"type": "exit"})


# Thread for training the agent - updated with batched implementation
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations):
    """Training thread with batched processing for better GPU utilization"""
    # Hyperparameters
    LEARNING_RATE = 3e-7
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 50.0
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
        
        # OPTIMIZATION: Batch processing instead of individual transition processing
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


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
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
            "cur_pd": cur_pd_t,     # Now a dictionary for this timestep
            "old_pd": old_pd_t,     # Now a dictionary for this timestep
            "next_obs": rollout["next_obs"][t]
        })

    # Handle bootstrapping for GAE calculation
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
        
    # Compute GAE (Generalized Advantage Estimation)
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
    Process-based parallelism for environment stepping
    """
    # Set spawn method for multiprocessing (cleaner process creation)
    mp.set_start_method('spawn', force=True)
    
    # Create dummy environment for agent initialization
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent for training thread
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
    
    # Create queues for inter-process communication
    action_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    # Create shared stop event for processes
    stop_event = Event()
    
    # Create and start environment processes
    env_processes = []
    for env_i in range(num_envs):
        p = Process(
            target=env_worker,
            args=(
                env_i, 
                action_queues[env_i], 
                result_queue, 
                stop_event,
                in_model,
                in_weights
            )
        )
        p.daemon = True  # Ensure processes exit when main process exits
        p.start()
        env_processes.append(p)
    
    # Shared flag for threads
    stop_flag = [False]
    
    # Create and start coordinating thread
    env_thread = threading.Thread(
        target=environment_thread_mp,
        args=(
            action_queues, 
            result_queue, 
            rollout_steps, 
            rollout_queue, 
            out_episodes, 
            stop_flag, 
            stop_event
        )
    )
    
    # Create and start training thread
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations)
    )
    
    print("Starting threads and processes...")
    env_thread.start()
    train_thread.start()
    
    try:
        # Wait for training thread to complete
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping threads and processes...")
    finally:
        # Signal threads to stop
        stop_flag[0] = True
        stop_event.set()
        
        # Wait for threads to finish
        env_thread.join(timeout=10)
        if env_thread.is_alive():
            print("Warning: Environment thread did not terminate properly")
        
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
    parser.add_argument("--num-envs", required=False, type=int, default=2)
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