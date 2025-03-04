import os
from argparse import ArgumentParser
import pickle
import torch as th
import numpy as np
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from lib.policy_mod import compute_kl_loss
from lib.tree_util import tree_map
from lib.infinite_build_reward import custom_reward_function  # ✅ Import your custom reward function

th.autograd.set_detect_anomaly(True)

# ==== Hyperparameters ====
LEARNING_RATE = 3e-7
MAX_GRAD_NORM = 1.0
LAMBDA_KL = 50.0
GAMMA = 0.9999
LAM = 0.95
DEATH_PENALTY = -1000.0
VALUE_LOSS_COEF = 0.5
KL_DECAY = 0.9995


def save_checkpoint(agent, out_weights, iteration):
    """Saves a backup checkpoint with iteration number and updates the latest weights."""
    backup_file = f"{out_weights}.bak"
    checkpoint_file = f"{out_weights}.iter{iteration}"

    if os.path.exists(out_weights):
        os.replace(out_weights, backup_file)

    th.save(agent.policy.state_dict(), out_weights)
    print(f"[Checkpoint] Saved latest weights to {out_weights}")

    th.save(agent.policy.state_dict(), checkpoint_file)
    print(f"[Checkpoint] Saved iteration {iteration} weights to {checkpoint_file}")


def train_rl(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2,
    checkpoint_interval=5
):
    """Train the model using the custom reward function, with periodic checkpointing."""
    
    # ==== Load agent and environment ====
    dummy_env = HumanSurvival(**ENV_KWARGS).make()

    agent_policy_kwargs, agent_pi_head_kwargs = pickle.load(open(in_model, "rb")).values()
    agent = MineRLAgent(dummy_env, device="cuda", policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)

    # Load existing weights if available
    if os.path.exists(in_weights):
        print(f"[Loading] Using weights from {in_weights}")
        agent.load_weights(in_weights)
    else:
        print(f"[Warning] No initial weights found at {in_weights}, starting from scratch.")

    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    running_loss, total_steps = 0.0, 0
    visited_squares = set()  # ✅ Tracks visited build locations

    for iteration in range(num_iterations):
        print(f"[Iteration {iteration}] Training...")

        # Collect rollouts (environment interactions) using the reward function
        transitions_all = collect_transitions(agent, rollout_steps, num_envs, visited_squares)
        if not transitions_all:
            print(f"[Iteration {iteration}] No transitions collected, skipping update.")
            continue

        # ==== Perform RL Update ====
        optimizer.zero_grad()
        loss_list = compute_losses(agent, transitions_all)
        total_loss = sum(loss_list) / len(loss_list)
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Log progress
        running_loss += total_loss.item() * len(transitions_all)
        total_steps += len(transitions_all)
        avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
        print(f"[Iteration {iteration}] Loss={total_loss:.4f}, Steps={total_steps}, AvgLoss={avg_loss:.4f}")

        # ==== Save Checkpoint ====
        if iteration % checkpoint_interval == 0 or iteration == num_iterations - 1:
            save_checkpoint(agent, out_weights, iteration)

    print(f"[Final] Training complete. Final weights saved to {out_weights}")


def collect_transitions(agent, rollout_steps, num_envs, visited_squares):
    """Simulates environment interactions and collects transitions using the custom reward function."""
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    transitions_all = []

    for _ in range(rollout_steps):
        for env_i in range(num_envs):
            if not done_list[env_i]:
                with th.no_grad():
                    minerl_action, _, _, _, _ = agent.get_action_and_training_info(
                        minerl_obs=obs_list[env_i], hidden_state=None, stochastic=True, taken_action=None
                    )

                next_obs, _, done, info = envs[env_i].step(minerl_action)

                # ✅ Compute custom reward using the reward function
                reward, visited_squares = custom_reward_function(next_obs, done, info, visited_squares)

                if done:
                    reward += DEATH_PENALTY  # Apply death penalty if episode ends

                transitions_all.append({"obs": obs_list[env_i], "action": minerl_action, "reward": reward, "done": done})

                obs_list[env_i] = next_obs
                done_list[env_i] = done

                if done:
                    obs_list[env_i] = envs[env_i].reset()
                    done_list[env_i] = False

    return transitions_all


def compute_losses(agent, transitions):
    """Computes RL loss for backpropagation."""
    loss_list = []
    for t in transitions:
        reward = t["reward"]
        log_prob, v_pred = None, None  # These should come from the policy outputs

        loss_rl = -(reward * log_prob) if log_prob else 0
        value_loss = (v_pred - th.tensor(reward, device="cuda")) ** 2 if v_pred else 0
        kl_loss = compute_kl_loss(None, None)  # KL Loss placeholder
        total_loss = loss_rl + VALUE_LOSS_COEF * value_loss + LAMBDA_KL * kl_loss
        loss_list.append(total_loss.mean())

    return loss_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)
    parser.add_argument("--out-episodes", required=False, type=str, default="episode_lengths.txt")
    parser.add_argument("--num-iterations", required=False, type=int, default=10)
    parser.add_argument("--rollout-steps", required=False, type=int, default=40)
    parser.add_argument("--num-envs", required=False, type=int, default=2)
    parser.add_argument("--checkpoint-interval", required=False, type=int, default=5)

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_episodes=args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        checkpoint_interval=args.checkpoint_interval
    )
