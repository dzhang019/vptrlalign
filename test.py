import gym
import minerl
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom environment
from custom_minerl_env import register_custom_env, LogsAndIronSwordEnv

# Register the environment
register_custom_env()

def main():
    # Create the environment
    env = gym.make("LogsAndIronSword-v0")
    print("Environment created successfully!")
    
    # Print observation and action spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset the environment
    obs = env.reset()
    print("Environment reset")
    
    # Test the environment with random actions
    print("Running test episode with random actions...")
    total_reward = 0.0
    done = False
    step = 0
    
    while not done and step < 1000:  # Limit to 1000 steps for the test
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step
        next_obs, reward, done, info = env.step(action)
        
        # Update total reward
        total_reward += reward
        
        # Print reward if non-zero
        if reward != 0:
            print(f"Step {step}, Reward: {reward}, Total: {total_reward}")
            
            # Print inventory if available
            if "inventory" in next_obs:
                inventory = next_obs["inventory"]
                for item in ["log", "planks", "stick", "crafting_table", "wooden_pickaxe", 
                            "stone_pickaxe", "iron_ore", "iron_ingot", "iron_sword"]:
                    if item in inventory and inventory[item] > 0:
                        print(f"  {item}: {inventory[item]}")
        
        # Render the environment (if available)
        env.render()
        
        # Update observation
        obs = next_obs
        step += 1
        
        # Small delay to make rendering visible
        time.sleep(0.01)
    
    print(f"Episode finished after {step} steps with total reward: {total_reward}")
    
    # Close the environment
    env.close()

def train_with_ppo():
    """Train an agent using Stable Baselines 3 PPO"""
    
    # Create a vectorized environment
    def make_env():
        return gym.make("LogsAndIronSword-v0")
    
    env = DummyVecEnv([make_env])
    
    # Create the PPO agent
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log="./logs/"
    )
    
    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=1000000)
    
    # Save the trained model
    model.save("logs_and_sword_ppo_agent")
    
    print("Training completed!")
    
    # Test the trained agent
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
    # Uncomment to train with PPO:
    # train_with_ppo()
