from argparse import ArgumentParser
import pickle
import time
import numpy as np
 
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
 
from agent import MineRLAgent, ENV_KWARGS
 
def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, device = "cuda", policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
 
    print("---Launching MineRL environment (be patient)---")
    start_time = time.time()
    obs = env.reset()
    reset_time = time.time() - start_time
    print(f"Environment reset took {reset_time:.3f}s")
    
    # Tracking variables
    step_times = []
    action_gen_times = []
    env_step_times = []
    render_times = []
    
    step_count = 0
    total_start_time = time.time()
    last_report_time = total_start_time
    
    while True:
        iteration_start = time.time()
        
        # Generate action
        action_start = time.time()
        minerl_action = agent.get_action(obs)
        action_time = time.time() - action_start
        action_gen_times.append(action_time)
        
        # Take environment step
        env_step_start = time.time()
        obs, reward, done, info = env.step(minerl_action)
        env_step_time = time.time() - env_step_start
        env_step_times.append(env_step_time)
        
        # Render
        render_start = time.time()
        env.render()
        render_time = time.time() - render_start
        render_times.append(render_time)
        
        # Track full step time
        step_time = time.time() - iteration_start
        step_times.append(step_time)
        
        # Increment step counter
        step_count += 1
        
        # Print detailed timing every step
        print(f"Step {step_count}: Total={step_time:.3f}s (Action={action_time:.3f}s, Env={env_step_time:.3f}s, Render={render_time:.3f}s)")
        
        # Print summary stats every 40 steps
        if step_count % 40 == 0:
            elapsed = time.time() - last_report_time
            avg_step = np.mean(step_times[-40:])
            avg_action = np.mean(action_gen_times[-40:])
            avg_env = np.mean(env_step_times[-40:])
            avg_render = np.mean(render_times[-40:])
            
            print(f"\n==== TIMING SUMMARY FOR 40 STEPS ====")
            print(f"Total time for 40 steps: {elapsed:.3f}s")
            print(f"Average step time: {avg_step:.3f}s")
            print(f"Average action generation time: {avg_action:.3f}s")
            print(f"Average environment step time: {avg_env:.3f}s")
            print(f"Average render time: {avg_render:.3f}s")
            print(f"Effective steps per second: {40/elapsed:.2f}")
            print(f"Total steps so far: {step_count}")
            print(f"Total runtime: {time.time() - total_start_time:.1f}s")
            print(f"=======================================\n")
            
            last_report_time = time.time()
            
        # Handle episode completion
        if done:
            print(f"Episode complete after {step_count} steps")
            obs = env.reset()
  
if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")
 
    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering to measure raw performance")
 
    args = parser.parse_args()
    
    main(args.model, args.weights)