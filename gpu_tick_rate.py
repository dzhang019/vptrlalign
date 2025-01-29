from argparse import ArgumentParser
import pickle
import numpy as np
import cv2
import gym
import minerl
import torch  # Added for GPU support
import math
import time
import os

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

def euclidean_distance(pos1, pos2):
    return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2)))

def render_and_record(env, agent, out, frame_width, frame_height, tick_skip=4):
    """
    Runs the main loop at a high tick rate, rendering every `tick_skip` steps.

    Args:
        env: The MineRL environment.
        agent: The pretrained MineRL agent.
        out: cv2.VideoWriter object for recording video.
        frame_width: Width of the video frames.
        frame_height: Height of the video frames.
        tick_skip: Number of steps to skip rendering for a high tick rate.

    Returns:
        A dictionary with final run statistics.
    """
    start_time = time.time()
    step_count = total_reward = health_lost = hunger_lost = distance_traveled = 0.0
    lowest_health = 20.0

    obs = env.reset()
    done = False

    # Move obs to GPU (if applicable)
    if isinstance(obs, dict):
        for key in obs:
            if isinstance(obs[key], np.ndarray):
                obs[key] = torch.tensor(obs[key], device=device, dtype=torch.float32)

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

        # Get agent action (move data to GPU)
        action = agent.get_action(obs)
        
        # Take multiple steps per rendered frame to increase tick rate
        for _ in range(tick_skip):
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Move obs to GPU (if applicable)
            if isinstance(obs, dict):
                for key in obs:
                    if isinstance(obs[key], np.ndarray):
                        obs[key] = torch.tensor(obs[key], device=device, dtype=torch.float32)

            if done:
                break

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

        # Render only every tick_skip steps to increase speed
        if step_count % tick_skip == 0:
            frame = env.render(mode='rgb_array')
            frame = cv2.resize((frame[..., :3] if frame.shape[-1] == 4 else frame), (frame_width, frame_height))
            
            # Overlay stats
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

            out.write(frame)
            cv2.imshow("MineRL", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting early.")
                break

        if done or step_count >= 5000:
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
    output_folder = "minerl_runs"
    os.makedirs(output_folder, exist_ok=True)
    video_filename = os.path.join(output_folder, f"run_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    env = HumanSurvival(**ENV_KWARGS).make()
    agent_params = pickle.load(open(model, "rb"))
    policy_kwargs = agent_params["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    frame_width, frame_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 30, (frame_width, frame_height))

    print("---Launching MineRL environment (be patient)---")
    stats = render_and_record(env, agent, out, frame_width, frame_height, tick_skip=4)

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
