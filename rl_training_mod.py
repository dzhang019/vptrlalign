# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
# NOTE: This is _not_ the original code used for VPT!
#       This is merely to illustrate how to fine-tune the models and includes
#       the processing steps used.

# This will likely be much worse than what original VPT did:
# we are not training on full sequences, but only one step at a time to save VRAM.

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from agent_mod import PI_HEAD_KWARGS, MineRLAgent, ENV_KWARGS
from data_loader import DataLoader
from lib.tree_util import tree_map

from lib.reward_structure_mod import custom_reward_function, curiosity_reward
from lib.policy_mod import compute_kl_loss
from torchvision import transforms
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival


EPOCHS = 2
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

def convert_to_torch(obs):
    """
    Convert all NumPy arrays in the observation tree to PyTorch tensors.
    """
    return tree_map(
        lambda x: th.tensor(x.copy()) if isinstance(x, np.ndarray) and x.strides and any(s < 0 for s in x.strides) else
                  th.tensor(x) if isinstance(x, np.ndarray) else x,
        obs
    )

resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor()
])


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def train_rl(in_model, in_weights, out_weights, num_episodes=10):
    env = HumanSurvival(**ENV_KWARGS).make()

    # Load model parameters
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # Initialize agent with pretrained weights
    agent = MineRLAgent(env, device="cuda", policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    # Create a separate copy of the pretrained policy for KL regularization
    pretrained_policy = MineRLAgent(env, device="cuda", policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    pretrained_policy.load_weights(in_weights)

    # RL Training
    visited_chunks = set()
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=0.0001)
    lambda_kl = 0.1  # KL regularization weight

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        obs = env.reset()  # Use raw observation from the environment
        done = False
        cumulative_reward = 0
        state = agent.policy.initial_state(batch_size=1)

        while not done:
            env.render()

            # Get action using the agent's get_action method
            action = agent.get_action(obs)  # Handles preprocessing internally
            print(f"Action sent to env.step(): {action}")

            try:
                next_obs, env_reward, done, info = env.step(action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            # Compute rewards
            reward, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            curiosity_bonus = curiosity_reward(obs)
            total_reward = reward + curiosity_bonus
            cumulative_reward += total_reward

            # RL and KL Loss
            _, v_pred, _ = agent.policy(obs, state_in=state, first=th.tensor([False]))
            advantage = total_reward - v_pred.item()
            loss_rl = -advantage * agent.policy.get_logprob_of_action(_, action)
            loss_kl = compute_kl_loss(agent.policy, pretrained_policy.policy, obs)
            total_loss = loss_rl + lambda_kl * loss_kl

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            obs = next_obs

        print(f"Episode {episode}: Cumulative reward = {cumulative_reward}")

    # Save fine-tuned weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)




def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()

    policy = agent.policy
    trainable_parameters = policy.parameters()

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                # TODO need to clean up this hidden state after worker is done with the work item.
                #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first
            )

            log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = -log_prob / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be fine-tuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be fine-tuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where fine-tuned weights will be saved")
    parser.add_argument("--num-episodes", required=False, type=int, default=10, help="Number of training episodes")

    args = parser.parse_args()

    train_rl(
        in_model=args.in_model,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        num_episodes=args.num_episodes
    )