#!/usr/bin/env python
import os
# Disable MineRL process watcher to avoid the warning and potential issues.
os.environ["MINERL_DISABLE_PROCESS_WATCHER"] = "1"

from argparse import ArgumentParser
import importlib
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

# Instead of a fixed import for the reward function, you can later dynamically import one if desired.
#from lib.phase1 import 

# Import our modified compute_kl_loss (which now accepts a temperature T)
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers import RewardForCollectingItems, RewardForCollectingItemsOnce
from minerl.herobraine.hero import handlers
from torch.cuda.amp import autocast, GradScaler

th.autograd.set_detect_anomaly(True)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


# Update environment definition to use proper reward handlers
class CustomHumanSurvival(HumanSurvival):
    def __init__(self):
        super().__init__(**ENV_KWARGS)
        
    def create_reward_handlers(self):
        super_handlers = super().create_reward_handlers()
        # Add +1 reward for each log collected
        log_reward = RewardForCollectingItems([
            dict(type="log", amount=1, reward=1.0)
        ])
        # Add one-time +1000 reward for iron sword
        sword_reward = RewardForCollectingItemsOnce([
            dict(type="iron_sword", amount=1, reward=1000.0)
        ])
        return super_handlers + [log_reward, sword_reward]


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

    def stop(self):
        # Signal queue to stop all operations
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)  # Add poison pill


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


# Environment worker process
def env_worker(env_id, action_queue, result_queue, stop_flag):
    try:
        env = CustomHumanSurvival().make()
        obs = env.reset()
        result_queue.put(("INIT", env_id, obs, False, 0, None))
        print(f"[Env {env_id}] Initialized successfully")
    except Exception as e:
        print(f"[Env {env_id}] INIT ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        result_queue.put(("ERROR", env_id, None, True, 0, str(e)))
        return

    print(f"[Env {env_id}] Started")
    try:
        while not stop_flag.value:
            try:
                action = action_queue.get(timeout=0.1)
                if action is None or stop_flag.value:
                    break
                next_obs, env_reward, done, info = env.step(action)
                # Removed env.render() to prevent hanging in headless environments
                result_queue.put(("STEP", env_id, next_obs, done, env_reward, info),
                                 block=True, timeout=1.0)
                if done:
                    obs = env.reset()
                    result_queue.put(("RESET", env_id, obs, False, 0, None),
                                     block=True, timeout=1.0)
            except queue.Empty:
                if stop_flag.value:
                    break
            except Exception as e:
                print(f"[Env {env_id}] Critical error: {e}")
                result_queue.put(("ERROR", env_id, None, True, 0, str(e)))
                break
    finally:
        try:
            env.close()
        except:
            pass
        print(f"[Env {env_id}] Stopped")


def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue,
                       out_episodes, stop_flag, num_envs, phase_coordinator):
    # Prepare lists for all environments.
    obs_list = [None] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    last_actions = [None] * num_envs
    initialized = [False] * num_envs

    # Drain the result queue for a fixed duration to capture INIT messages.
    drain_duration = 3.0  # seconds
    start_time = time.time()
    print("Draining initialization messages for {:.1f} seconds...".format(drain_duration))
    while time.time() - start_time < drain_duration:
        try:
            # Use a short timeout to try draining repeatedly.
            msg = result_queue.get(timeout=0.5)
            msg_type, env_id, obs, done, reward, info = msg
            if msg_type == "INIT" and obs is not None:
                initialized[env_id] = True
                obs_list[env_id] = obs
                print(f"Collected INIT for env {env_id}")
            else:
                # For now, ignore non-INIT messages during initialization.
                pass
        except queue.Empty:
            pass

    if sum(initialized) < num_envs:
        print(f"Warning: Only {sum(initialized)} out of {num_envs} environments initialized successfully: {initialized}")
        successful_envs = [i for i, init in enumerate(initialized) if init]
        if len(successful_envs) == 0:
            stop_flag[0] = True
            return
        # Filter lists to only include the successful environments.
        obs_list = [obs_list[i] for i in successful_envs]
        episode_step_counts = [episode_step_counts[i] for i in successful_envs]
        hidden_states = [hidden_states[i] for i in successful_envs]
        last_actions = [last_actions[i] for i in successful_envs]
        num_envs = len(successful_envs)
    else:
        print(f"All {num_envs} environments initialized successfully.")

    # Optionally, try to fetch any additional INIT/RESET messages.
    for _ in range(num_envs):
        try:
            msg_type, env_id, obs, done, reward, info = result_queue.get(timeout=1.0)
            if msg_type in ("INIT", "RESET"):
                obs_list[env_id] = obs
                print(f"[Environment Thread] Got initial observation from env {env_id}")
        except queue.Empty:
            if stop_flag[0]:
                return

    # Now continue with rollout collection...
    iteration = 0
    while not stop_flag[0]:
        if phase_coordinator.in_auxiliary_phase():
            print("[Environment Thread] Pausing collection during auxiliary phase")
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            if stop_flag[0]:
                continue
        iteration += 1
        start_iter_time = time.time()
        rollouts = [
            {"obs": [], "actions": [], "rewards": [], "dones": [],
             "hidden_states": [], "next_obs": []}
            for _ in range(num_envs)
        ]
        env_waiting_for_result = [False] * num_envs
        env_step_counts = [0] * num_envs

        # Send actions to each environment.
        for env_id in range(num_envs):
            if stop_flag[0]:
                break
            if obs_list[env_id] is None:
                continue
            try:
                with th.no_grad():
                    action_info = agent.get_action_and_training_info(
                        minerl_obs=obs_list[env_id],
                        hidden_state=hidden_states[env_id],
                        stochastic=True,
                        taken_action=None
                    )
                    minerl_action = action_info[0]
                    new_hid = action_info[-1]
                hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                last_actions[env_id] = minerl_action
                if not action_queues[env_id]._closed:
                    action_queues[env_id].put(minerl_action)
                    env_waiting_for_result[env_id] = True
            except Exception as e:
                print(f"[Env {env_id}] Error generating action: {e}")
                if stop_flag[0]:
                    break

        total_transitions = 0
        result_timeout = 0.1
        while total_transitions < rollout_steps * num_envs and not stop_flag[0]:
            if phase_coordinator.in_auxiliary_phase():
                print("[Environment Thread] Auxiliary phase started during collection")
                break
            try:
                msg_type, env_id, data, done, reward, info = result_queue.get(timeout=result_timeout)
                if msg_type == "STEP":
                    next_obs = data
                    if env_waiting_for_result[env_id]:
                        rollouts[env_id]["obs"].append(obs_list[env_id])
                        rollouts[env_id]["actions"].append(last_actions[env_id])
                        rollouts[env_id]["rewards"].append(reward)
                        rollouts[env_id]["dones"].append(done)
                        rollouts[env_id]["hidden_states"].append(
                            tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id])
                        )
                        rollouts[env_id]["next_obs"].append(next_obs)
                        obs_list[env_id] = next_obs
                        episode_step_counts[env_id] += 1
                        env_waiting_for_result[env_id] = False
                        env_step_counts[env_id] += 1
                        total_transitions += 1
                        if done:
                            hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                            with open(out_episodes, "a") as f:
                                f.write(f"{episode_step_counts[env_id]}\n")
                            episode_step_counts[env_id] = 0
                    if env_step_counts[env_id] < rollout_steps and not done and not stop_flag[0]:
                        try:
                            with th.no_grad():
                                action_info = agent.get_action_and_training_info(
                                    minerl_obs=obs_list[env_id],
                                    hidden_state=hidden_states[env_id],
                                    stochastic=True,
                                    taken_action=None
                                )
                                minerl_action = action_info[0]
                                new_hid = action_info[-1]
                            hidden_states[env_id] = tree_map(lambda x: x.detach(), new_hid)
                            last_actions[env_id] = minerl_action
                            if not action_queues[env_id]._closed:
                                action_queues[env_id].put(minerl_action)
                                env_waiting_for_result[env_id] = True
                        except Exception as e:
                            if stop_flag[0]:
                                break
                elif msg_type == "RESET":
                    obs_list[env_id] = data
                    with open(out_episodes, "a") as f:
                        f.write(f"{episode_step_counts[env_id]}\n")
                    episode_step_counts[env_id] = 0
                    env_waiting_for_result[env_id] = False
                elif msg_type == "ERROR":
                    print(f"[Environment Thread] Error from env {env_id}: {info}")
            except queue.Empty:
                if stop_flag[0]:
                    break
                continue
        if not phase_coordinator.in_auxiliary_phase() and not stop_flag[0]:
            end_iter_time = time.time()
            duration = end_iter_time - start_iter_time
            actual_transitions = sum(len(r["obs"]) for r in rollouts)
            try:
                rollout_queue.put(rollouts, timeout=1.0)
                print(f"[Environment Thread] Iteration {iteration} collected {actual_transitions} transitions in {duration:.2f}s")
            except queue.Full:
                print("[Environment Thread] Dropped rollouts due to full queue")
        elif not stop_flag[0]:
            phase_coordinator.buffer_rollout(rollouts)

    print("[Environment Thread] Shutting down...")
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            break
    for env_id in range(num_envs):
        try:
            if not action_queues[env_id]._closed:
                action_queues[env_id].put(None)
        except:
            pass

# (The rest of the functions train_unroll, run_sleep_phase, run_policy_update,
# training_thread, and train_rl_mp remain essentially as in the previous corrected version.)
# For brevity, please refer to the previous complete implementation for these parts.
# They have not been modified regarding the initialization hang.
def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator, args):
    LEARNING_RATE = 5e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = args.lambda_kl  # from command line
    GAMMA = 0.9999
    LAM = 0.95
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995
    PPG_ENABLED = True
    PPG_N_PI_UPDATES = 8
    PPG_BETA_CLONE = 1.0

    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    total_steps = 0
    iteration = 0
    scaler = GradScaler()
    
    pi_update_counter = 0
    stored_rollouts = []
    has_aux_head = hasattr(agent.policy, 'aux_value_head')
    if has_aux_head:
        print("[Training Thread] Detected auxiliary value head, enabling PPG")
    else:
        print("[Training Thread] No auxiliary value head detected, PPG will be disabled")
        PPG_ENABLED = False

    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        do_aux_phase = (PPG_ENABLED and has_aux_head and 
                        pi_update_counter >= PPG_N_PI_UPDATES and 
                        len(stored_rollouts) > 0)
        if do_aux_phase:
            phase_coordinator.start_auxiliary_phase()
            print(f"[Training Thread] Starting PPG auxiliary phase (iteration {iteration})")
            recent_rollouts = get_recent_rollouts(stored_rollouts, max_rollouts=5)
            run_sleep_phase(
                agent=agent,
                recent_rollouts=recent_rollouts,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=MAX_GRAD_NORM,
                beta_clone=PPG_BETA_CLONE
            )
            phase_coordinator.end_auxiliary_phase()
            print("[Training Thread] Auxiliary phase complete")
            buffered_rollouts = phase_coordinator.get_buffered_rollouts()
            if buffered_rollouts:
                print(f"[Training Thread] Processing {len(buffered_rollouts)} buffered rollouts")
                for rollout in buffered_rollouts:
                    rollout_queue.put(rollout)
            pi_update_counter = 0
            stored_rollouts = []
            th.cuda.empty_cache()
        else:
            pi_update_counter += 1
            print(f"[Training Thread] Policy phase {pi_update_counter}/{PPG_N_PI_UPDATES} - Waiting for rollouts...")
            wait_start = time.time()
            rollouts = rollout_queue.get()
            wait_duration = time.time() - wait_start
            print(f"[Training Thread] Waited {wait_duration:.3f}s for rollouts.")
            if PPG_ENABLED and has_aux_head:
                stored_rollouts.append(rollouts)
                if len(stored_rollouts) > 2:
                    stored_rollouts = stored_rollouts[-2:]
            train_start = time.time()
            print(f"[Training Thread] Processing rollouts for iteration {iteration}")
            avg_policy_loss, avg_value_loss, avg_kl_loss, num_transitions = run_policy_update(
                agent=agent,
                pretrained_policy=pretrained_policy,
                rollouts=rollouts,
                optimizer=optimizer,
                scaler=scaler,
                value_loss_coef=VALUE_LOSS_COEF,
                lambda_kl=LAMBDA_KL,
                max_grad_norm=MAX_GRAD_NORM,
                temp=args.temp  # Pass temperature argument
            )
            train_duration = time.time() - train_start
            print(f"[Training Thread] Policy Phase {pi_update_counter}/{PPG_N_PI_UPDATES} - "
                  f"Time: {train_duration:.3f}s, Transitions: {num_transitions}, "
                  f"PolicyLoss: {avg_policy_loss:.4f}, ValueLoss: {avg_value_loss:.4f}, "
                  f"KLLoss: {avg_kl_loss:.4f}")
            running_loss += (avg_policy_loss + avg_value_loss + avg_kl_loss) * num_transitions
            total_steps += num_transitions
            avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
            LAMBDA_KL *= KL_DECAY
    while iteration < num_iterations and not stop_flag[0]:
        try:
            rollouts = rollout_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_flag[0]:
                break
            continue

# In train_rl_mp, note we remove p.daemon = True.
def train_rl_mp(in_model, in_weights, out_weights, out_episodes,
                num_iterations=10, rollout_steps=40, num_envs=2, queue_size=3):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set")
    
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
    
    phase_coordinator = PhaseCoordinator()
    stop_flag = mp.Value('b', False)
    action_queues = [Queue() for _ in range(num_envs)]
    result_queue = Queue()
    rollout_queue = RolloutQueue(maxsize=queue_size)
    
    workers = []
    for env_id in range(num_envs):
        p = Process(
            target=env_worker,
            args=(env_id, action_queues[env_id], result_queue, stop_flag)
        )
        p.start()
        workers.append(p)
        time.sleep(0.4)
    
    # Add a delay to ensure worker processes have time to send INIT messages.
    print("Waiting for worker processes to initialize...")
    time.sleep(2)
    
    # Then start the threads.
    env_thread = threading.Thread(
        target=environment_thread,
        args=(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
              out_episodes, thread_stop, num_envs, phase_coordinator)
    )
    train_thread = threading.Thread(
        target=training_thread,
        args=(agent, pretrained_policy, rollout_queue, thread_stop, num_iterations, phase_coordinator, args)
    )
    print("Starting threads...")
    env_thread.start()
    train_thread.start()
    try:
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping threads and processes...")
    finally:
        print("\n=== EMERGENCY SHUTDOWN ===")
        stop_flag.value = True
        thread_stop[0] = True
        for p in workers:
            if p.is_alive():
                p.terminate()
        for q in action_queues:
            q.cancel_join_thread()
            q.close()
        result_queue.cancel_join_thread()
        result_queue.close()
        env_thread.join(timeout=1.0)
        train_thread.join(timeout=1.0)
        th.cuda.empty_cache()
        dummy_env.close()
        try:
            th.save(agent.policy.state_dict(), out_weights)
            print(f"Weights saved to {out_weights}")
        except:
            print("Failed to save weights")
        if any(p.is_alive() for p in workers):
            print("Forcing exit due to hanging processes")
            os._exit(1)


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
    parser.add_argument("--temp", type=float, default=2.0, help="Temperature for distillation loss")
    parser.add_argument("--lambda-kl", type=float, default=50.0, help="Weight for KL distillation loss")
    args = parser.parse_args()
    
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
