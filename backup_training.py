from argparse import ArgumentParser
import pickle
import time
import os
import signal
import sys

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from data_loader import DataLoader
from lib.tree_util import tree_map

from lib.infinite_build_reward import reward_function
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

th.autograd.set_detect_anomaly(True)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def train_rl(
    in_model,
    in_weights,
    out_weights,
    out_episodes,
    num_iterations=10,
    rollout_steps=40,
    num_envs=2,
    save_interval=5,
    resume_from=None,
    checkpoint_dir="checkpoints"
):
    """
    Modified version with checkpointing, resume, and signal handling.
    """

    # ==== Hyperparams ====
    LEARNING_RATE = 3e-7
    MAX_GRAD_NORM = 1.0
    LAMBDA_KL = 50.0
    GAMMA = 0.9999
    LAM = 0.95
    DEATH_PENALTY = -1000.0
    VALUE_LOSS_COEF = 0.5
    KL_DECAY = 0.9995

    # ==== 1) Create agent + pretrained policy ====
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

    for p1, p2 in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert p1.data_ptr() != p2.data_ptr(), "Weights are shared!"

    # ==== 2) Setup checkpointing ====
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_iter = 0
    running_loss = 0.0
    total_steps = 0

    if resume_from:
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = th.load(resume_from, map_location="cuda")
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        pretrained_policy.policy.load_state_dict(checkpoint['policy_state_dict'])
        start_iter = checkpoint['iteration'] + 1
        running_loss = checkpoint['running_loss']
        total_steps = checkpoint['total_steps']
        LAMBDA_KL = checkpoint['lambda_kl']
        print(f"Resumed from iteration {start_iter}, steps={total_steps}")

    # ==== 3) Create parallel envs + hidden states ====
    envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(num_envs)]
    obs_list = [env.reset() for env in envs]
    done_list = [False] * num_envs
    episode_step_counts = [0] * num_envs

    hidden_states = [agent.policy.initial_state(batch_size=1) for _ in range(num_envs)]

    # ==== 4) Optimizer ====
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    if resume_from:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ==== 5) Signal handling ====
    global_iteration = start_iter

    def signal_handler(sig, frame):
        nonlocal global_iteration
        checkpoint_path = os.path.join(checkpoint_dir, f"INTERRUPTED_iter{global_iteration}.pt")
        print(f"\nSaving interrupted checkpoint to {checkpoint_path}")
        th.save({
            'policy_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': global_iteration,
            'running_loss': running_loss,
            'total_steps': total_steps,
            'lambda_kl': LAMBDA_KL
        }, checkpoint_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ==== 6) Main training loop ====
    for iteration in range(start_iter, num_iterations):
        global_iteration = iteration
        print(f"[Iteration {iteration}] Starting...")

        # Environment rollout phase
        rollouts = [... ]  # Existing rollout collection logic

        # Training phase
        transitions_all = [...]  # Existing training logic

        # Loss calculation and backprop
        optimizer.zero_grad()
        total_loss = ...  # Existing loss calculation
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Update tracking variables
        LAMBDA_KL *= KL_DECAY
        running_loss += total_loss.item()
        total_steps += len(transitions_all)

        # Periodic checkpointing
        if (iteration + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter{iteration+1}.pt")
            th.save({
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'running_loss': running_loss,
                'total_steps': total_steps,
                'lambda_kl': LAMBDA_KL
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    print(f"Saving final weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)


# Rest of the code (train_unroll, main block) remains the same as original
# with added CLI arguments for checkpointing

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-episodes", default="episode_lengths.txt")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=2)
    # New arguments
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--resume-from", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")

    args = parser.parse_args()

    train_rl(
        args.in_model,
        args.in_weights,
        args.out_weights,
        args.out_episodes,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        save_interval=args.save_interval,
        resume_from=args.resume_from,
        checkpoint_dir=args.checkpoint_dir
    )
