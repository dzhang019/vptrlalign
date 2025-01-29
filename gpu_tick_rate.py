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

################################################################################
# If the environment's 'agent.py' doesn't already map model and data to GPU,
# we can ensure it here by explicitly converting observations. 
################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

def euclidean_distance(pos1, pos2):
    return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2)))

def to_gpu_if_ndarray(obs: dict):
    """
    Ensures any NumPy arrays in the obs dict are converted into torch tensors on GPU.
    This helps fix negative stride errors using np.ascontiguousarray.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = torch.tensor(
                np.ascontiguousarray(val),
                device=device,
                dtype=torch.float32
            )
    return obs

def render_and_record(env, agent, out, frame_width, frame_height, tick_skip=5):
    """
    High tick rate loop:
      - Processes `tick_skip` steps of the environment per render.
      - Renders + overlays stats occasionally to reduce overhead & speed up the loop.

    Args:
      env: The MineRL environment.
      agent: A MineRLAgent with a GPU-based model if CUDA is available.
      out: cv2.VideoWriter for saving frames to an MP4 file.
      frame_width: The width of each rendered frame (pixels).
      frame_height: The height of each rendered frame (pixels).
      tick_skip: Number of environment steps to run before rendering.

    Returns:
      dict of final stats from the run.
    """
    start_time = time.time()

    # Tracking variables
    step_count = 0
    total_reward = 0.0
    health_lost = 0.0
    hunger_lost = 0.0
    distance_traveled = 0.0
    lowest_health = 20.0

    # Reset env & convert obs to GPU if available
    obs = env.reset()
    if isinstance(obs, dict):
        obs = to_gpu_if_ndarray(obs)

    done = False

    # Initialize "previous" state trackers
    life_stats = obs.get("life_stats", {})
    prev_health = life_stats.get("life", 20.0)
    prev_food = life_stats.get("food", 20.0)

    location_stats = obs.get("location_stats", {})
    prev_pos = [
        float(location_stats.get("xpos", 0.0)),
        float(location_stats.get("ypos", 0.0)),
        float(location_stats.get("zpos", 0.0))
    ]

    while not done:
        elapsed_seconds = time.time() - start_time

        # For a high tick rate, we run multiple env steps between renders
        for _ in range(tick_skip):
            # Agent picks an action. We'll let MineRLAgent handle GPU inference
            action = agent.get_action(obs)  # This should do GPU-based forward pass

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Convert new obs to GPU-friendly format
            if isinstance(obs, dict):
                obs = to_gpu_if_ndarray(obs)

            # Update trackers
            life_stats = obs.get("life_stats", {})
            current_health = life_stats.get("life", prev_health)
            current_food = life_stats.get("food", prev_food)

            if current_health < prev_health:
                health_lost += (prev_health - current_health)
            if current_health < lowest_health:
                lowest_health = current_health

            if current_food < prev_food:
                hunger_lost += (prev_food - current_food)

            prev_health = current_health
            prev_food = current_food

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

        # Render a frame now that we've done some steps
        frame = env.render(mode='rgb_array')

        # Standard fix for alpha channels + resizing
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Add overlay
        stats = {
            "Time Elapsed": f"{elapsed_seconds:.2f}s",
            "Steps": step_count,
            "Total Reward": f"{total_reward:.2f}",
            "Health Lost": f"{health_lost:.2f}",
            "Lowest Health": f"{lowest_health:.2f}",
            "Hunger Lost": f"{hunger_lost:.2f}",
            "Distance Traveled": f"{distance_traveled:.2f}"
        }
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

        # Write frame to video, show live
        out.write(frame)
        cv2.imshow("MineRL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early.")
            break

        if done or step_count >= 2000:
            print("Ending run.")
            break

    return {
        "Elapsed Time": f"{elapsed_seconds:.2f}s",
        "Steps": step_count,
        "Total Reward": f"{total_reward:.2f}",
        "Health Lost": f"{health_lost:.2f}",
        "Lowest Health": f"{lowest_health:.2f}",
        "Hunger Lost": f"{hunger_lost:.2f}",
        "Distance Traveled": f"{distance_traveled:.2f}"
    }

def main(model, weights):
    # Create output folder for videos + logs
    output_folder = "minerl_runs"
    os.makedirs(output_folder, exist_ok=True)
    video_filename = os.path.join(output_folder, f"run_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    # Create environment
    env = HumanSurvival(**ENV_KWARGS).make()

    # Load agent
    agent_params = pickle.load(open(model, "rb"))
    policy_kwargs = agent_params["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    # Create agent with GPU device
    from agent import MineRLAgent
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # Set up video writing
    frame_width, frame_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 30, (frame_width, frame_height))

    print("---Launching MineRL environment (be patient)---")
    # We'll skip ~5 steps for a high effective tick rate
    stats = render_and_record(env, agent, out, frame_width, frame_height, tick_skip=5)

    # Cleanup
    out.release()
    summary = "\n".join([f"{key}: {value}" for key, value in stats.items()])
    with open(os.path.join(output_folder, "run_log.txt"), "a") as f:
        f.write(summary + "\n")

    print("---Run Complete---")
    print(summary)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")
    parser.add_argument("--weights", type=str, required=True, help="Path to '.weights' file.")
    parser.add_argument("--model", type=str, required=True, help="Path to '.model' file.")
    args = parser.parse_args()
    main(args.model, args.weights)
