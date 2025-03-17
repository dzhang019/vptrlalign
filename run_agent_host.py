from argparse import ArgumentParser
import pickle
import socket
import time
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def start_minecraft_server(port=8888):
    """Start a Minecraft server instance for the interactor to connect to."""
    from minerl.env.malmo import InstanceManager, MinecraftInstance
    
    interactor_port = 31415
    try:
        InstanceManager.add_existing_instance(interactor_port)
        instance = InstanceManager.get_instance(-1)
        print(f"Connected to existing Minecraft instance on port {interactor_port}")
    except AssertionError:
        print(f"Starting new Minecraft instance on port {interactor_port}")
        instance = MinecraftInstance(interactor_port)
        instance.launch(daemonize=True)
    
    # Allow time for the server to initialize
    time.sleep(5)
    return instance, port

def main(model, weights, port=8888, debug=False):
    """
    Run the MineRL agent and host a server for interactor connections.
    
    Args:
        model: Path to the model file
        weights: Path to the weights file
        port: Port for the interactor to connect on
        debug: Enable debug logging
    """
    # Setup logging if debug mode is enabled
    if debug:
        import logging
        import coloredlogs
        coloredlogs.install(level=logging.DEBUG)
    
    # Start the Minecraft server for interactor connections
    instance, server_port = start_minecraft_server(port)
    local_ip = get_local_ip()
    
    # Print connection information for users
    print("\n" + "="*80)
    print(f"HOSTING MINECRAFT SERVER")
    try:
        from minerl.env.malmo import malmo_version
        print(f"Minecraft version: Compatible with Malmo {malmo_version}")
        # MineRL typically uses Minecraft 1.11.2 or 1.12.2 with Malmo
        if malmo_version.startswith("0.37."):
            print(f"This typically corresponds to Minecraft version 1.12.2")
        elif malmo_version.startswith("0.36."):
            print(f"This typically corresponds to Minecraft version 1.11.2")
    except ImportError:
        print("Could not determine Minecraft version")
        
    print(f"Local IP address: {local_ip}")
    print(f"Connection Port: {server_port}")
    print(f"To connect from another machine:")
    print(f"  1. Run the interactor script: python interactor.py {server_port} -i {local_ip}")
    print(f"  2. Open Minecraft client and connect to Direct Server: {local_ip}:{server_port}")
    print(f"To connect from this machine:")
    print(f"  1. Run the interactor script: python interactor.py {server_port} -i 127.0.0.1")
    print(f"  2. Open Minecraft client and connect to Direct Server: localhost:{server_port}")
    print("="*80 + "\n")
    
    # Initialize environment and agent
    print("Starting MineRL environment...")
    env = HumanSurvival(**ENV_KWARGS).make()
    
    print("Loading agent model...")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    
    print("Launching MineRL environment (be patient)...")
    obs = env.reset()
    
    try:
        print("Agent is now running. Connect with the interactor to take control.")
        print("Press Ctrl+C to stop the agent.")
        
        while True:
            minerl_action = agent.get_action(obs)
            obs, reward, done, info = env.step(minerl_action)
            env.render()
            
            if done:
                print("Environment reset due to terminal state.")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nStopping agent")
    finally:
        print("Cleaning up environment...")
        env.close()

if __name__ == "__main__":
    parser = ArgumentParser("Run a MineRL agent that hosts a server for interactor connections")
    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--port", type=int, default=8888, help="Port to host the server on (default: 8888)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    main(args.model, args.weights, args.port, args.debug)