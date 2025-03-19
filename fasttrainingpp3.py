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


class RolloutQueue:
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, rollouts):
        self.queue.put(rollouts, block=True)
    
    def get(self):
        return self.queue.get(block=True)
    
    def qsize(self):
        return self.queue.qsize()


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
    env = HumanSurvival(**ENV_KWARGS).make()
    
    obs = env.reset()
    prev_logs = 0
    prev_sword = 0
    episode_step_count = 0
    
    print(f"[Env {env_id}] Started")
    
    result_queue.put((env_id, None, obs, False, 0, None))
    
    action_timeout = 0.01
    step_count = 0
    while not stop_flag.value:
        try:
            action = action_queue.get(timeout=action_timeout)
            
            if action is None:
                break
                
            step_start = time.time()
            next_obs, env_reward, done, info = env.step(action)
            step_time = time.time() - step_start
            step_count += 1
            
            # MineRL reward calculation
            current_logs = info.get('stat/logged', 0)
            current_sword = info.get('stat/crafted_iron_sword', 0)
            
            delta_logs = current_logs - prev_logs
            delta_sword = current_sword - prev_sword
            
            custom_reward = delta_logs * 1.0 + delta_sword * 10.0
            
            prev_logs = current_logs
            prev_sword = current_sword
            
            if done:
                custom_reward -= 2000.0
                
            episode_step_count += 1
            
            result_queue.put((env_id, action, next_obs, done, custom_reward, info))
            if env_id == 0:
                env.render()
            
            if done:
                result_queue.put((env_id, None, None, True, episode_step_count, None))
                obs = env.reset()
                prev_logs = 0
                prev_sword = 0
                episode_step_count = 0
                result_queue.put((env_id, None, obs, False, 0, None))
            else:
                obs = next_obs
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Env {env_id}] Error: {e}")
            
    env.close()
    print(f"[Env {env_id}] Stopped")


def environment_thread(agent, rollout_steps, action_queues, result_queue, rollout_queue, 
                       out_episodes, stop_flag, num_envs, phase_coordinator):
    obs_list = [None] * num_envs
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
    for _ in range(num_envs):
        env_id, _, obs, _, _, _ = result_queue.get()
        obs_list[env_id] = obs
    
    iteration = 0
    while not stop_flag[0]:
        if phase_coordinator.in_auxiliary_phase():
            phase_coordinator.auxiliary_phase_complete.wait(timeout=1.0)
            if phase_coordinator.in_auxiliary_phase():
                continue
        
        iteration += 1
        start_time = time.time()
        
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
        
        env_waiting_for_result = [False] * num_envs
        env_step_counts = [0] * num_envs
        
        for env_id in range(num_envs):
            if obs_list[env_id] is not None:
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
                action_queues[env_id].put(minerl_action)
                env_waiting_for_result[env_id] = True
        
        total_transitions = 0
        result_timeout = 0.01
        
        while total_transitions < rollout_steps * num_envs:
            if phase_coordinator.in_auxiliary_phase():
                break
            
            try:
                env_id, action, next_obs, done, reward, info = result_queue.get(timeout=result_timeout)
                
                if action is None and done and next_obs is None:
                    continue
                
                if action is None and not done:
                    obs_list[env_id] = next_obs
                    continue
                
                if env_waiting_for_result[env_id]:
                    rollouts[env_id]["obs"].append(obs_list[env_id])
                    rollouts[env_id]["actions"].append(action)
                    rollouts[env_id]["rewards"].append(reward)
                    rollouts[env_id]["dones"].append(done)
                    rollouts[env_id]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_id])
                    )
                    rollouts[env_id]["next_obs"].append(next_obs)
                    
                    obs_list[env_id] = next_obs
                    
                    if done:
                        hidden_states[env_id] = agent.policy.initial_state(batch_size=1)
                    
                    env_waiting_for_result[env_id] = False
                    env_step_counts[env_id] += 1
                    total_transitions += 1
                    
                    if env_step_counts[env_id] < rollout_steps:
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
                        action_queues[env_id].put(minerl_action)
                        env_waiting_for_result[env_id] = True
            except queue.Empty:
                continue
        
        if not phase_coordinator.in_auxiliary_phase():
            end_time = time.time()
            duration = end_time - start_time
            actual_transitions = sum(len(r["obs"]) for r in rollouts)
            
            rollout_queue.put(rollouts)
            
            print(f"[Environment Thread] Iteration {iteration} collected {actual_transitions} transitions "
                f"across {num_envs} envs in {duration:.3f}s")
        else:
            phase_coordinator.buffer_rollout(rollouts)


def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    if T == 0:
        return transitions
    
    obs_seq = rollout["obs"]
    act_seq = rollout["actions"]
    hidden_states_seq = rollout["hidden_states"]

    agent_outputs = agent.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=hidden_states_seq[0],
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    if len(agent_outputs) == 5:
        pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hid = agent_outputs
        has_aux_head = True
    else:
        pi_dist_seq, vpred_seq, log_prob_seq, final_hid = agent_outputs
        aux_vpred_seq = None
        has_aux_head = False
    
    old_outputs = pretrained_policy.get_sequence_and_training_info(
        minerl_obs_list=obs_seq,
        initial_hidden_state=pretrained_policy.policy.initial_state(1),
        stochastic=False,
        taken_actions_list=act_seq
    )
    
    if len(old_outputs) == 5:
        old_pi_dist_seq, old_vpred_seq, _, old_log_prob_seq, _ = old_outputs
    else:
        old_pi_dist_seq, old_vpred_seq, old_log_prob_seq, _ = old_outputs

    for t in range(T):
        cur_pd_t = {k: v[t] for k, v in pi_dist_seq.items()}
        old_pd_t = {k: v[t] for k, v in old_pi_dist_seq.items()}
        
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
        
        if has_aux_head and aux_vpred_seq is not None:
            transition["aux_v_pred"] = aux_vpred_seq[t]
            
        transitions.append(transition)

    bootstrap_value = 0.0
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            
            action_outputs = agent.get_action_and_training_info(
                minerl_obs=transitions[-1]["next_obs"],
                hidden_state=hid_t,
                stochastic=False,
                taken_action=None
            )
            
            vpred_index = 2
            bootstrap_value = action_outputs[vpred_index].item()
    
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
        transitions[i]["return"] = gae + v_i

    return transitions


def get_recent_rollouts(stored_rollouts, max_rollouts=5):
    recent_rollouts = []
    
    for rollout_batch in reversed(stored_rollouts):
        for env_rollout in rollout_batch:
            if len(env_rollout["obs"]) > 0:
                recent_rollouts.append(env_rollout)
                if len(recent_rollouts) >= max_rollouts:
                    break
        
        if len(recent_rollouts) >= max_rollouts:
            break
    
    recent_rollouts.reverse()
    
    return recent_rollouts


def run_sleep_phase(agent, recent_rollouts, optimizer, scaler, max_grad_norm=1.0, beta_clone=1.0):
    has_aux_head = hasattr(agent.policy, 'aux_value_head')
    if not has_aux_head:
        return
    
    cycle2_data = []
    
    # Cycle 1
    for rollout_idx, rollout in enumerate(recent_rollouts):
        if len(rollout["obs"]) == 0:
            continue
            
        transitions = train_unroll(agent, agent, rollout)
        if len(transitions) == 0:
            continue
        
        cpu_transitions = []
        for t in transitions:
            cpu_trans = {
                "obs": t["obs"],
                "action": t["action"],
                "return": t["return"],
            }
            cpu_transitions.append(cpu_trans)
        
        current_hidden = tree_map(lambda x: x.to("cuda").contiguous(), rollout["hidden_states"][0])
        
        for chunk_start in range(0, len(transitions), 64):
            chunk_end = min(chunk_start + 64, len(transitions))
            chunk = transitions[chunk_start:chunk_end]
            
            chunk_obs = [t["obs"] for t in chunk]
            chunk_actions = [t["action"] for t in chunk]
            
            with th.no_grad():
                outputs = agent.get_sequence_and_training_info(
                    minerl_obs_list=chunk_obs,
                    initial_hidden_state=current_hidden,
                    stochastic=False,
                    taken_actions_list=chunk_actions
                )
                
                final_hidden = outputs[-1]
                current_hidden = tree_map(lambda x: x.detach(), final_hidden)
                
                if len(outputs) >= 5:
                    pi_dist_seq = outputs[0]
                    for i, t in enumerate(chunk):
                        cpu_idx = chunk_start + i
                        if cpu_idx < len(cpu_transitions):
                            cpu_transitions[cpu_idx]["orig_pi"] = {
                                k: v[i].clone().detach().cpu() for k, v in pi_dist_seq.items()
                            }
            
        cycle2_data.extend(cpu_transitions)
    
    # Cycle 2
    for batch_start in range(0, len(cycle2_data), 16):
        batch_end = min(batch_start + 16, len(cycle2_data))
        batch = cycle2_data[batch_start:batch_end]
        
        if not batch or not all("orig_pi" in t for t in batch):
            continue
        
        batch_returns = th.tensor([t["return"] for t in batch], device="cuda")
        batch_obs = [t["obs"] for t in batch]
        batch_actions = [t["action"] for t in batch]
        
        optimizer.zero_grad(set_to_none=True)
        
        with th.enable_grad():
            outputs = agent.get_sequence_and_training_info(
                minerl_obs_list=batch_obs,
                initial_hidden_state=agent.policy.initial_state(1),
                stochastic=False,
                taken_actions_list=batch_actions
            )
            
            if len(outputs) >= 5:
                curr_pi, _, aux_values, _, _ = outputs
        
        try:
            with th.autocast(device_type='cuda'):
                aux_value_loss = ((aux_values - batch_returns) ** 2).mean()
                
                policy_distill_losses = []
                for i, t in enumerate(batch):
                    orig_pi = {k: v.to("cuda") for k, v in t["orig_pi"].items()}
                    curr_pi_i = {k: v[i] for k, v in curr_pi.items()}
                    kl_loss = compute_kl_loss(curr_pi_i, orig_pi)
                    policy_distill_losses.append(kl_loss)
                
                policy_distill_loss = th.stack(policy_distill_losses).mean()
                loss = aux_value_loss + beta_clone * policy_distill_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
        except Exception as e:
            optimizer.zero_grad(set_to_none=True)
        
        th.cuda.empty_cache()


def run_policy_update(agent, pretrained_policy, rollouts, optimizer, scaler, 
                      value_loss_coef=0.5, lambda_kl=0.2, max_grad_norm=1.0):
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl_loss = 0.0
    num_valid_envs = 0
    total_transitions = 0
    
    optimizer.zero_grad()
    
    for env_idx, env_rollout in enumerate(rollouts):
        if len(env_rollout["obs"]) == 0:
            continue
        
        env_transitions = train_unroll(
            agent,
            pretrained_policy,
            env_rollout,
            gamma=0.9999,
            lam=0.95
        )
        
        if len(env_transitions) == 0:
            continue
        
        env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) 
                                for t in env_transitions])
        env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
        env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
        env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
        
        env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
        
        with autocast():
            policy_loss = -(env_advantages * env_log_probs).mean()
            value_loss = ((env_v_preds - env_returns) ** 2).mean()
            
            kl_losses = []
            for t in env_transitions:
                kl_loss = compute_kl_loss(t["cur_pd"], t["old_pd"])
                kl_losses.append(kl_loss)
            kl_loss = th.stack(kl_losses).mean()
            
            env_loss = policy_loss + (value_loss_coef * value_loss) + (lambda_kl * kl_loss)
        
        scaler.scale(env_loss).backward()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
        num_valid_envs += 1
        total_transitions += len(env_transitions)
    
    if num_valid_envs == 0:
        return 0.0, 0.0, 0.0, 0
    
    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    avg_policy_loss = total_policy_loss / num_valid_envs
    avg_value_loss = total_value_loss / num_valid_envs
    avg_kl_loss = total_kl_loss / num_valid_envs
    
    return avg_policy_loss, avg_value_loss, avg_kl_loss, total_transitions


def training_thread(agent, pretrained_policy, rollout_queue, stop_flag, num_iterations, phase_coordinator):
    LEARNING_RATE = 5e-6
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 20.0
    PPG_ENABLED = True
    PPG_N_PI_UPDATES = 8
    PPG_BETA_CLONE = 1.0
    
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    pi_update_counter = 0
    stored_rollouts = []
    
    has_aux_head = hasattr(agent.policy, 'aux_value_head')
    if not has_aux_head:
        PPG_ENABLED = False
    
    while iteration < num_iterations and not stop_flag[0]:
        iteration += 1
        
        do_aux_phase = (PPG_ENABLED and 
                        has_aux_head and 
                        pi_update_counter >= PPG_N_PI_UPDATES and 
                        len(stored_rollouts) > 0)
        
        if do_aux_phase:
            phase_coordinator.start_auxiliary_phase()
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
            buffered_rollouts = phase_coordinator.get_buffered_rollouts()
            if buffered_rollouts:
                for rollout in buffered_rollouts:
                    rollout_queue.put(rollout)
            
            pi_update_counter = 0
            stored_rollouts = []
            th.cuda.empty_cache()
            
        else:
            pi_update_counter += 1
            
            rollouts = rollout_queue.get()
            
            if PPG_ENABLED and has_aux_head:
                stored_rollouts.append(rollouts)
                if len(stored_rollouts) > 2:
                    stored_rollouts = stored_rollouts[-2:]
            
            avg_policy_loss, avg_value_loss, avg_kl_loss, num_transitions = run_policy_update(
                agent=agent,
                pretrained_policy=pretrained_policy,
                rollouts=rollouts,
                optimizer=optimizer,
                scaler=scaler,
                value_loss_coef=0.5,
                lambda_kl=LAMBDA_KL,
                max_grad_norm=MAX_GRAD_NORM
            )
            
            print(f"[Training Thread] Policy Phase {pi_update_counter}/{PPG_N_PI_UPDATES} - "
                  f"Transitions: {num_transitions}, "
                  f"PolicyLoss: {avg_policy_loss:.4f}, ValueLoss: {avg_value_loss:.4f}, "
                  f"KLLoss: {avg_kl_loss:.4f}")
            
            LAMBDA_KL *= 0.9995


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
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
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
        p.daemon = True
        p.start()
        workers.append(p)
        time.sleep(0.4)
    
    thread_stop = [False]
    
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
            phase_coordinator
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
            phase_coordinator
        )
    )
    
    env_thread.start()
    train_thread.start()
    
    try:
        train_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    finally:
        thread_stop[0] = True
        stop_flag.value = True
        
        for q in action_queues:
            try:
                q.put(None)
            except:
                pass
        
        env_thread.join(timeout=10)
        train_thread.join(timeout=5)
        
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        dummy_env.close()
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
    parser.add_argument("--queue-size", required=False, type=int, default=3)

    args = parser.parse_args()
    
    weights = th.load(args.in_weights, map_location="cpu")
    has_aux_head = any('aux' in key for key in weights.keys())

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
