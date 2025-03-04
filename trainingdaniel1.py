from argparse import ArgumentParser
import pickle
import time
import threading
import queue

import torch.profiler
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


# Thread for stepping through environments and collecting rollouts
def environment_thread(agent, envs, rollout_steps, rollout_queue, out_episodes, stop_flag):
    num_envs = len(envs)
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs
    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]
    
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
                    
                    if done_flag_i:
                        env_reward_i += -1000.0  # DEATH_PENALTY
                    
                    # Store rollout data
                    rollouts[env_i]["obs"].append(obs_list[env_i])
                    rollouts[env_i]["actions"].append(minerl_action_i)
                    rollouts[env_i]["rewards"].append(env_reward_i)
                    rollouts[env_i]["dones"].append(done_flag_i)
                    rollouts[env_i]["hidden_states"].append(
                        tree_map(lambda x: x.detach().cpu().contiguous(), hidden_states[env_i])
                    )
                    print(f"Stored hidden state: keys shape = {hidden_states[env_i][0].shape}, values shape = {hidden_states[env_i][1].shape}")
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
        
        # Send the collected rollouts to the training thread
        env_end_time = time.time()
        env_duration = env_end_time - env_start_time
        print(f"[Environment Thread] Iteration {iteration} collected {rollout_steps} steps "
              f"across {num_envs} envs in {env_duration:.3f}s")
        rollout_queue.put(rollouts)


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
    max_profile_iters = 5
    iteration = 0
    scaler = GradScaler()
    max_profile_iters = 5  # how many iterations we record
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
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
            # optimizer.step()
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
            if iteration <= max_profile_iters:
                    prof.step()
            else:
                # after we've done a few profiled iters, no need to keep calling step
                pass
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

def train_unroll(agent, pretrained_policy, rollout, gamma=0.999, lam=0.95):
    transitions = []
    T = len(rollout["obs"])
    print("sequence length (T): ", T)
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
        transitions.append({
            "obs": rollout["obs"][t],
            "action": rollout["actions"][t],
            "reward": rollout["rewards"][t],
            "done": rollout["dones"][t],
            "v_pred": vpred_seq[t],        # Now [T], so index directly
            "log_prob": log_prob_seq[t],    # Now [T]
            "cur_pd": pi_dist_seq[t],       # Now [T, ...]
            "old_pd": old_pi_dist_seq[t],   # Now [T, ...]
            "next_obs": rollout["next_obs"][t]
        })

    # Bootstrap and GAE calculation remain the same
    if not transitions[-1]["done"]:
        with th.no_grad():
            hid_t_cpu = rollout["hidden_states"][-1]
            print(f"Hidden state (CPU): {hid_t_cpu}")

            hid_t = tree_map(lambda x: x.to("cuda").contiguous(), hid_t_cpu)
            print(f"Hidden state (GPU): {hid_t}")
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
    Minimally modified version with separate threads for environment stepping and training
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
    
    # for p1, p2 in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
    #     assert p1.data_ptr() != p2.data_ptr(), "Weights are shared!"
    
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

    args = parser.parse_args()

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