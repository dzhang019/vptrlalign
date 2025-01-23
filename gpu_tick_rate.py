from argparse import ArgumentParser
import pickle
import numpy as np
import cv2
import gym
import minerl
import math
import time
import os
import torch  # Ensure PyTorch is installed

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

def euclidean_distance(pos1, pos2):
    """Compute Euclidean distance between two 3D points [x, y, z]."""
    return math.sqrt(
        (pos2[0] - pos1[0])**2 +
        (pos2[1] - pos1[1])**2 +
        (pos2[2] - pos1[2])**2
    )

def render_and_record(env, agent, out, frame_width, frame_height, device):
    """
    Handles the main loop for rendering the environment, updating stats, and recording the video.

    Args:
        env: The MineRL environment.
        agent: The pretrained MineRL agent.
        out: cv2.VideoWriter object for recording video.
        frame_width: Width of the video frames.
        frame_height: Height of the video frames.
        device: The device (CPU or GPU) for computations.

    Returns:
        A dictionary with final run statistics.
    """
    start_time = time.time()
    # Initialize trackers
    step_count = total_reward = health_lost = hunger_lost = distance_traveled = 0.0
    lowest_health = 20.0

    obs = env.reset()
    done = False

    # Initialize health, hunger, and position
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

        # Agent action and environment step
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Update health and hunger
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

        # Update distance traveled
        location_stats = obs.get("location_stats", {})
        current_pos = [
            float(location_stats.get("xpos", prev_pos[0])),
            float(location_stats.get("ypos", prev_pos[1])),
            float(location_stats.get("zpos", prev_pos[2]))
        ]

        distance_traveled += euclidean_distance(prev_pos, current_pos)
        prev_pos = current_pos

        # Render frame
        frame = env.render(mode='rgb_array')
        frame = cv2.resize((frame[..., :3] if frame.shape[-1] == 4 else frame), (frame_width, frame_height))

        # Add overlay
        stats = {
            "Time Elapsed": f"{elapsed_seconds:.2f} s",
            "Steps": int(step_count),
            "Total Reward": f"{total_reward:.2f}",
            "Health Lost": f"{health_lost:.2f}",
            "Lowest Health": f"{lowest_health:.2f}",
            "Hunger Lost": f"{hunger_lost:.2f}",
            "Distance Traveled": f"{distance_traveled:.2f}"
        }

        for i, (label, value) in enumerate(stats.items()):
            cv2.putText(
                frame, f"{label}: {value}", (10, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Write frame to video
        out.write(frame)
        cv2.imshow("MineRL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early.")
            break

        if done or step_count >= 1000:
            print("Ending run.")
            break

    return {
        "Elapsed Time": f"{elapsed_seconds:.2f} s",
        "Steps": int(step_count),
        "Total Reward": f"{total_reward:.2f}",
        "Health Lost": f"{health_lost:.2f}",
        "Lowest Health": f"{lowest_health:.2f}",
        "Hunger Lost": f"{hunger_lost:.2f}",
        "Distance Traveled": f"{distance_traveled:.2f}"
    }

def main(model, weights):
    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recording the runs to a folder
    output_folder = "minerl_runs"
    os.makedirs(output_folder, exist_ok=True)
    video_filename = os.path.join(output_folder, f"run_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    env = HumanSurvival(**ENV_KWARGS).make()
    agent_params = pickle.load(open(model, "rb"))
    policy_kwargs = agent_params["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device=device)
    agent.load_weights(weights)

    # More recording setup
    frame_width, frame_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 30, (frame_width, frame_height))

    print("---Launching MineRL environment (be patient)---")

    # Start the main loop for rendering and recording
    stats = render_and_record(env, agent, out, frame_width, frame_height, device)

    # Save run summary
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
