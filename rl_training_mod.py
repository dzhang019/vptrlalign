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

from lib.reward_structure_mod import custom_reward_function
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

def preprocess_obs_for_policy(obs, device="cuda"):
    """
    Prepare a separate observation specifically for the model (policy input).
    Converts `pov` to PyTorch tensor and reshapes to (1, 1, C, H, W).
    """
    if "pov" not in obs:
        raise KeyError("'pov' key is missing in observation for policy input.")
    
    img = obs["pov"].copy()  # Extract the POV image
    img = th.tensor(img).permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
    img = img.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and time dimensions
    return {"img": img}


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs
'''
def train_rl(in_model, in_weights, out_weights, num_episodes=10):
    # Example hyperparameters
    LEARNING_RATE = 0
    #1e-5
    MAX_GRAD_NORM = 1.0      # For gradient clipping
    LAMBDA_KL = 1.0          # KL regularization weight

    env = HumanSurvival(**ENV_KWARGS).make()

    # Load parameters for both current agent and pretrained agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)
    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert not (agent_param.data_ptr() == pretrained_param.data_ptr()), "Weights are shared!"
    # Optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    total_steps = 0

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        visited_chunks = set()

        # Optionally reset agent's internal hidden state if needed
        # e.g., agent.reset() if you have such a method, or manually.
        # For example:
        # agent.hidden_state = agent.policy.initial_state(batch_size=1)

        while True:
            env.render()

            # 1) SINGLE FORWARD PASS with new method:
            #    get_action_and_training_info(obs) gives us:
            #      - 'minerl_action' to pass to env.step
            #      - 'pi_dist' (distribution for log-prob or KL)
            #      - 'v_pred' (value estimate)
            #      - 'log_prob' of the chosen action
            #      - 'new_hidden_state' is also returned, but if the agent
            #        tracks hidden_state internally, we might not need it here.
            minerl_action, pi_dist, v_pred, log_prob, new_hidden_state = \
                agent.get_action_and_training_info(obs, stochastic=True)
            
            #print("train_rl: log_prob.requires_grad rights after .get_action_and_training_info", log_prob.requires_grad)
            # 2) Step the environment with 'minerl_action'
            try:
                next_obs, env_reward, done, info = env.step(minerl_action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break
            if done:
                break
            # 3) Compute your custom reward
            reward, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            cumulative_reward += reward

            # 4) Single-step advantage (still naive)
            #    advantage = reward - V(s), but you might want a next-state bootstrap, etc.
            v_pred_val = v_pred.detach() 
            #if hasattr(v_pred, 'item') else v_pred
            advantage = reward - v_pred_val
            loss_rl = -advantage * log_prob

            # 5) KL regularization with the pretrained policy
            #    we do a forward pass on 'obs' for the pretrained policy distribution
            #    so we can measure distance.  We'll do the same "env_obs_to_agent" logic.
            with th.no_grad():
                obs_for_pretrained = agent._env_obs_to_agent(obs)
                obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                (old_pi_dist, _, _), _ = pretrained_policy.policy(
                    obs=obs_for_pretrained,
                    state_in=pretrained_policy.policy.initial_state(1),
                    first=th.tensor([[False]], dtype=th.bool, device="cuda")
                    )
        
                # for key, value  in pi_dist.items():
                #     if isinstance(value, th.Tensor):
                #         print(f"pi_dist[{key}].requires_grad:", value.requires_grad)
                # for key, value in old_pi_dist.items():
                #     if isinstance(value, th.Tensor):
                #         print(f"old_pi_dist[{key}].requires_grad:", value.requires_grad)
            old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)
            loss_kl = compute_kl_loss(pi_dist, old_pi_dist)
            total_loss = loss_rl + LAMBDA_KL * loss_kl

            # 6) Backprop and update
            optimizer.zero_grad()
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            obs = next_obs
            running_loss += total_loss.item()
            total_steps += 1

        print(f"Episode {episode} finished. Cumulative reward = {cumulative_reward}")

        if total_steps > 0:
            avg_loss = running_loss / total_steps
            print(f" Steps so far: {total_steps}, average training loss: {avg_loss:.4f}")

    # Save fine-tuned weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)

'''
'''last working
def train_rl(in_model, in_weights, out_weights, num_episodes=10):
    # Example hyperparameters
    LEARNING_RATE = 1e-5
    MAX_GRAD_NORM = 1.0      # For gradient clipping
    LAMBDA_KL = 1.0          # KL regularization weight
    BATCH_SIZE = 40          # Number of steps per batch

    env = HumanSurvival(**ENV_KWARGS).make()

    # Load parameters for both current agent and pretrained agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)

    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert not (agent_param.data_ptr() == pretrained_param.data_ptr()), "Weights are shared!"

    # Optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        visited_chunks = set()
        trajectories = []  # Collect data for batch updates

        while not done:
            env.render()

            # Forward pass
            minerl_action, pi_dist, v_pred, log_prob, new_hidden_state = \
                agent.get_action_and_training_info(obs, stochastic=True)

            # Environment step
            try:
                next_obs, env_reward, done, info = env.step(minerl_action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            # Compute custom reward
            reward, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            cumulative_reward += reward
            v_pred = v_pred.detach()
            log_prob = log_prob.detach()
            # Store trajectory data
            trajectories.append((
                obs, 
                minerl_action, 
                reward, 
                v_pred.detach(), 
                log_prob.detach(), 
                {k: v.detach() for k,v in pi_dist.items() if isinstance(v, th.Tensor)}
            ))
            obs = next_obs
            print(f"Allocated: {th.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Reserved: {th.cuda.memory_reserved() / 1e9:.2f} GB")

        print(f"Episode {episode} finished. Cumulative reward = {cumulative_reward}")

        # Process trajectories in batches
        for i in range(0, len(trajectories), BATCH_SIZE):
            batch = trajectories[i:i + BATCH_SIZE]

            # Extract batch data
            obs_batch, action_batch, reward_batch, v_pred_batch, log_prob_batch, pi_dist_batch = zip(*batch)

            # Convert to tensors
            obs_tensor = th.stack([th.tensor(obs, dtype=th.float32) for obs in obs_batch]).to("cuda")
            reward_tensor = th.tensor(reward_batch, dtype=th.float32).to("cuda")
            log_prob_tensor = th.stack(log_prob_batch).to("cuda")

            # Compute advantage
            v_pred_tensor = th.stack(v_pred_batch)
            advantage = reward_tensor - v_pred_tensor
            loss_rl = -(advantage * log_prob_tensor).mean()

            # Compute KL loss
            with th.no_grad():
                obs_for_pretrained = agent._env_obs_to_agent(obs_tensor)
                obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
                (old_pi_dist, _, _), _ = pretrained_policy.policy(
                    obs=obs_for_pretrained,
                    state_in=pretrained_policy.policy.initial_state(len(obs_batch)),
                    first=th.tensor([[False]] * len(obs_batch), dtype=th.bool, device="cuda")
                )
            old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)
            loss_kl = compute_kl_loss(pi_dist_batch, old_pi_dist)

            # Combine losses
            total_loss = loss_rl + LAMBDA_KL * loss_kl

            # Backprop and update
            optimizer.zero_grad()
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    # Save fine-tuned weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)
'''

def train_rl(in_model, in_weights, out_weights, num_episodes=10):
    # Example hyperparameters
    LEARNING_RATE = 1e-5
    MAX_GRAD_NORM = 1.0      # For gradient clipping
    LAMBDA_KL = 1.0          # KL regularization weight
    BATCH_SIZE = 1           # Number of episodes per batch (updated)

    env = HumanSurvival(**ENV_KWARGS).make()

    # Load parameters for both current agent and pretrained agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    agent.load_weights(in_weights)

    pretrained_policy = MineRLAgent(
        env, device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    pretrained_policy.load_weights(in_weights)

    for agent_param, pretrained_param in zip(agent.policy.parameters(), pretrained_policy.policy.parameters()):
        assert not (agent_param.data_ptr() == pretrained_param.data_ptr()), "Weights are shared!"

    # Optimizer
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE)

    # Store multiple episodes for batch processing
    batched_trajectories = []  # Collect multiple episodes here

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        visited_chunks = set()
        episode_trajectory = []  # Store one episode's data

        while not done:
            env.render()

            # Forward pass
            minerl_action, pi_dist, v_pred, log_prob, new_hidden_state = \
                agent.get_action_and_training_info(obs, stochastic=True)

            # Environment step
            try:
                next_obs, env_reward, done, info = env.step(minerl_action)
                if 'error' in info:
                    print(f"Error in info: {info['error']}. Ending episode.")
                    break
            except Exception as e:
                print(f"Error during env.step(): {e}")
                break

            # Compute custom reward
            reward, visited_chunks = custom_reward_function(obs, done, info, visited_chunks)
            cumulative_reward += reward

            # Store step data for this episode
            episode_trajectory.append((
                obs, 
                minerl_action, 
                reward, 
                v_pred.detach(), 
                log_prob.detach(), 
                {k: v.detach() for k, v in pi_dist.items() if isinstance(v, th.Tensor)}
            ))
            obs = next_obs

        print(f"Episode {episode} finished. Cumulative reward = {cumulative_reward}")

        # Add the finished episode to the batch
        batched_trajectories.append(episode_trajectory)

        # Process the batch if enough episodes are collected
        if len(batched_trajectories) >= BATCH_SIZE:
            process_batched_episodes(batched_trajectories, agent, pretrained_policy, optimizer, LAMBDA_KL, MAX_GRAD_NORM)
            batched_trajectories.clear()  # Clear batched trajectories after processing

    # Save fine-tuned weights
    print(f"Saving fine-tuned weights to {out_weights}")
    th.save(agent.policy.state_dict(), out_weights)

# def process_batched_episodes(batched_trajectories, agent, pretrained_policy, optimizer, LAMBDA_KL, MAX_GRAD_NORM):
#     # Process each episode in the batch
#     for episode_trajectory in batched_trajectories:
#         # Extract trajectory data for the episode
#         obs_batch, action_batch, reward_batch, v_pred_batch, log_prob_batch, pi_dist_batch = zip(*episode_trajectory)

#         # Convert to tensors
#         obs_tensor = th.stack([th.tensor(obs, dtype=th.float32) for obs in obs_batch]).to("cuda")
#         reward_tensor = th.tensor(reward_batch, dtype=th.float32).to("cuda")
#         log_prob_tensor = th.stack(log_prob_batch).to("cuda")
#         v_pred_tensor = th.stack(v_pred_batch)

#         # Compute advantage
#         advantage = reward_tensor - v_pred_tensor
#         loss_rl = -(advantage * log_prob_tensor).mean()

#         # Compute KL loss
#         with th.no_grad():
#             obs_for_pretrained = agent._env_obs_to_agent(obs_tensor)
#             obs_for_pretrained = tree_map(lambda x: x.unsqueeze(1), obs_for_pretrained)
#             (old_pi_dist, _, _), _ = pretrained_policy.policy(
#                 obs=obs_for_pretrained,
#                 state_in=pretrained_policy.policy.initial_state(len(obs_batch)),
#                 first=th.tensor([[False]] * len(obs_batch), dtype=th.bool, device="cuda")
#             )
#         old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)
#         loss_kl = compute_kl_loss(pi_dist_batch, old_pi_dist)

#         # Combine losses
#         total_loss = loss_rl + LAMBDA_KL * loss_kl

#         # Backprop and update
#         optimizer.zero_grad()
#         total_loss.backward()
#         th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
#         optimizer.step()
def process_batched_episodes(batched_trajectories, agent, pretrained_policy, optimizer, LAMBDA_KL, MAX_GRAD_NORM):
    optimizer.zero_grad()  # Clear previous gradients

    total_loss = 0.0  # Accumulate loss over the batch

    for episode_trajectory in batched_trajectories:
        # Extract trajectory data for the episode
        obs_batch, action_batch, reward_batch, v_pred_batch, log_prob_batch, pi_dist_batch = zip(*episode_trajectory)

        # Process each step in the trajectory
        for step in range(len(obs_batch)):
            # Extract step data
            obs = obs_batch[step]
            reward = reward_batch[step]
            v_pred = v_pred_batch[step]
            log_prob = log_prob_batch[step]
            pi_dist = pi_dist_batch[step]

            # Convert observation to agent-compatible format
            obs_tensor = agent._env_obs_to_agent(obs)

            # Compute advantage
            advantage = reward - v_pred
            loss_rl = -(advantage * log_prob)

            # Compute KL loss with pretrained policy
            with th.no_grad():
                obs_for_pretrained = tree_map(lambda x: x.unsqueeze(0), obs_tensor)
                (old_pi_dist, _, _), _ = pretrained_policy.policy(
                    obs=obs_for_pretrained,
                    state_in=pretrained_policy.policy.initial_state(1),
                    first=th.tensor([[False]], dtype=th.bool, device="cuda")
                )
            old_pi_dist = tree_map(lambda x: x.detach(), old_pi_dist)
            loss_kl = compute_kl_loss(pi_dist, old_pi_dist)

            # Combine losses
            total_loss += loss_rl + LAMBDA_KL * loss_kl  # Accumulate loss

    # Perform a single backward pass and optimization step
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()


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
