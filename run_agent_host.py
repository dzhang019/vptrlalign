from argparse import ArgumentParser
import pickle
import socket
import time
import threading
import struct
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

def setup_interactor_thread(instance, client_port=8888):
    """Set up the interactor functionality in a background thread."""
    local_ip = get_local_ip()
    
    def interactor_worker():
        print(f"\nINTERACTOR: Setting up connection on port {client_port}")
        print(f"INTERACTOR: Run the following command in a separate terminal:")
        print(f"  python -m interactor {client_port} -i 127.0.0.1   (for local connection)")
        print(f"  python -m interactor {client_port} -i {local_ip}   (for network connection)")
        print(f"INTERACTOR: Then connect your Minecraft client to localhost:{client_port} or {local_ip}:{client_port}")
        
    thread = threading.Thread(target=interactor_worker)
    thread.daemon = True
    thread.start()
    return thread

def main(model, weights, port=8888, use_interactor=False, debug=False):
    """
    Run the MineRL agent with optional interactor setup.
    
    Args:
        model: Path to the model file
        weights: Path to the weights file
        port: Port for the interactor to use
        use_interactor: Whether to attempt integrated interactor setup
        debug: Enable debug logging
    """
    # Setup logging if debug mode is enabled
    if debug:
        import logging
        import coloredlogs
        coloredlogs.install(level=logging.DEBUG)
    
    # Start the Minecraft instance
    from minerl.env.malmo import InstanceManager, MinecraftInstance, malmo_version
    
    # Print Minecraft version information
    print(f"Using Minecraft version compatible with Malmo version: {malmo_version}")
    
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
    
    local_ip = get_local_ip()
    
    # Print connection information for users
    print("\n" + "="*80)
    print(f"MINERL AGENT")
    
    # Print Minecraft version again for clarity
    try:
        print(f"Minecraft version: Compatible with Malmo {malmo_version}")
        # MineRL typically uses Minecraft 1.11.2 or 1.12.2 with Malmo
        if malmo_version.startswith("0.37."):
            print(f"This typically corresponds to Minecraft version 1.12.2")
        elif malmo_version.startswith("0.36."):
            print(f"This typically corresponds to Minecraft version 1.11.2")
    except ImportError:
        print("Could not determine Minecraft version")
    
    # Set up interactor if requested
    if use_interactor:
        setup_interactor_thread(instance, port)
        print("\nINTERACTOR INFO:")
        print(f"Local IP address: {local_ip}")
        print(f"Connection Port: {port}")
        print(f"Follow the instructions above to connect to the agent.")
    
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
        print("Agent is now running.")
        if use_interactor:
            print("Follow the interactor instructions above to connect and control the agent.")
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
    parser = ArgumentParser("Run a MineRL agent with optional interactor setup")
    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--port", type=int, default=8888, help="Port for interactor to use (default: 8888)")
    parser.add_argument("--interactor", action="store_true", help="Enable interactor setup instructions")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    main(args.model, args.weights, args.port, args.interactor, args.debug)