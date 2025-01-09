import minerl
import gym

env = gym.make("MineRLBasaltFindCave-v0")

for _ in range(5):
    try:
        print("Resetting environment...")
        obs = env.reset()  # Reset the environment
        print("Environment reset successful.")
    except Exception as e:
        print(f"Environment reset failed: {e}")
        break