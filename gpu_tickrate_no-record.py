from argparse import ArgumentParser
import pickle
import numpy as np
import cv2
import torch
import math
import time
import os

import minerl
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

def euclidean_distance(pos1, pos2):
    return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2)))

def to_gpu_if_ndarray(obs: dict):
    """Converts any NumPy arrays in the obs dict to GPU torch tensors (float32)."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            # Fix negative stride or other memory issues
            val = np.ascontiguousarray(val)
            obs[key] = torch.tensor(val, device=device, dtype=torch.float32)
    return obs

def run_high_tick_rate(env, agent, frame_width, frame_height, tick_skip=5):
    """
    Runs the environment with a high tick rate, skipping multiple environment steps between overlays.
    Uses GPU for model inference if available.

    Args:
        env: The MineRL environment.
        agent: MineRLAgent utilizing GPU for inference (if available).
        frame_width: Display width in pixels.
        frame_height: Display height in pixels.
        tick_skip: Number of env steps before each overlay/visual update.

    Returns:
        A dictionary of run statistics.
    """
    start_time = time.time()
    step_count = total_reward = health_lost = hunger_lost = distance_traveled = 0.0
    lowest_health = 20.0

    obs = env.reset()
    if isinstance(obs, dict):
        obs = to_gpu_if_ndarray(obs)

    done = False

    # Extract initial stats
    life_stats = obs.get("life_stats", {})
    prev_health = life_stats.get("life", 20.0)
    prev_food   = life_stats.get("food", 20.0)

    location_stats = obs.get("location_stats", {})
    prev_pos = [
        float(location_stats.get("xpos", 0.0)),
        float(location_stats.get("ypos", 0.0)),
        float(location_stats.get("zpos", 0.0))
    ]

    while not done:
        elapsed_seconds = time.time() - start_time

        # Perform multiple steps for a high tick rate
        action = agent.get_action(obs)  # forward pass on GPU
        for _ in range(tick_skip):
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Convert new obs data to GPU
            if isinstance(obs, dict):
                obs = to_gpu_if_ndarray(obs)

            # Update stats
            life_stats = obs.get("life_stats", {})
            current_health = life_stats.get("life", prev_health)
            current_food   = life_stats.get("food", prev_food)

            if current_health < prev_health:
                health_lost += (prev_health - current_health)
            if current_health < lowest_health:
                lowest_health = current_health

            if current_food < prev_food:
                hunger_lost += (prev_food - current_food)

            prev_health = current_health
            prev_food   = current_food

            location_stats = obs.get("location_stats", {})
            current_pos = [
                float(location_stats.get("xpos", prev_pos[0])),
                float(location_stats.get("ypos", prev_pos[1])),
                float(location_stats.get("zpos", prev_pos[2]))
            ]
            distance_traveled += euclidean_distance(prev_pos, current_pos)
            prev_pos = current_pos

            if done:
                break

        # Show overlay every 'tick_skip' steps
        #  Render environment as an RGB array
        frame = env.render(mode='rgb_array')
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Build overlay stats
        stats = {
            "Time Elapsed": f"{elapsed_seconds:.2f}s",
            "Steps": int(step_count),
            "Total Reward": f"{total_reward:.2f}",
            "Health Lost": f"{health_lost:.2f}",
            "Lowest Health": f"{lowest_health:.2f}",
            "Hunger Lost": f"{hunger_lost:.2f}",
            "Distance Traveled": f"{distance_traveled:.2f}"
        }
        # Add text overlay
        y_offset = 30
        for i, (label, value) in enumerate(stats.items()):
            cv2.putText(
                frame,
                f"{label}: {value}",
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Display the overlay in OpenCV window
        cv2.imshow("MineRL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early.")
            break

        if done or step_count >= 2000:
            print("Ending run.")
            break

    return {
        "Elapsed Time": f"{elapsed_seconds:.2f}s",
        "Steps": int(step_count),
        "Total Reward": f"{total_reward:.2f}",
        "Health Lost": f"{health_lost:.2f}",
        "Lowest Health": f"{lowest_health:.2f}",
        "Hunger Lost": f"{hunger_lost:.2f}",
        "Distance Traveled": f"{distance_traveled:.2f}"
    }

def main(model, weights):
    # No video recording
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Running agent on: {device_str}")

    env = HumanSurvival(**ENV_KWARGS).make()

    # Load agent
    agent_params = pickle.load(open(model, "rb"))
    policy_kwargs = agent_params["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    from agent import MineRLAgent
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # High tick rate setup
    frame_width, frame_height = 800, 600

    print("---Launching MineRL environment (be patient)---")
    stats = run_high_tick_rate(env, agent, frame_width, frame_height, tick_skip=5)

    # Summarize stats
    summary = "\n".join([f"{k}: {v}" for k, v in stats.items()])
    print("---Run Complete---")
    print(summary)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment (GPU, high tick rate, overlay)")
    parser.add_argument("--weights", type=str, required=True, help="Path to '.weights' file.")
    parser.add_argument("--model", type=str, required=True, help="Path to '.model' file.")
    args = parser.parse_args()

    main(args.model, args.weights)
