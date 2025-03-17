#!/usr/bin/env python3
# Modified interactor script with enhanced debugging

import argparse
import os
import socket
import struct
import time
import logging
import sys

# Set up logging
import coloredlogs
coloredlogs.install(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from minerl.env._multiagent import _MultiAgentEnv
    from minerl.env.malmo import InstanceManager, MinecraftInstance, malmo_version
    from minerl.env import comms
    logger.info(f"Successfully imported MineRL modules. Malmo version: {malmo_version}")
except ImportError as e:
    logger.error(f"Failed to import MineRL modules: {e}")
    sys.exit(1)

def get_socket(instance):
    """Create and return a socket connected to the given instance."""
    logger.debug(f"Creating socket to connect to {instance.host}:{instance.port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(60)  # 60 second timeout
    
    try:
        sock.connect((instance.host, instance.port))
        logger.debug(f"Socket connected successfully to {instance.host}:{instance.port}")
        return sock
    except Exception as e:
        logger.error(f"Failed to connect socket: {e}")
        raise

def request_interactor(instance, ip):
    """Request an interactor connection with enhanced error handling."""
    logger.debug(f"Requesting interactor for {ip} on instance {instance}")
    
    try:
        sock = get_socket(instance)
    except Exception as e:
        logger.error(f"Failed to get socket: {e}")
        return False
    
    # Send hello message - THIS MUST MATCH EXACTLY WHAT'S IN THE CODE
    try:
        logger.debug("Sending hello message...")
        # This must match exactly how the hello is defined in _MultiAgentEnv._TO_MOVE_hello
        comms.send_message(sock, ("<MalmoEnv" + malmo_version + "/>").encode())
        logger.debug("Hello message sent successfully")
    except Exception as e:
        logger.error(f"Error sending hello message: {e}")
        sock.close()
        return False
    
    # Send interaction request
    try:
        message = f"<Interact>{ip}</Interact>".encode()
        logger.debug(f"Sending interaction request: {message}")
        comms.send_message(sock, message)
        logger.debug("Interaction request sent")
    except Exception as e:
        logger.error(f"Error sending interaction request: {e}")
        sock.close()
        return False
    
    # Get reply
    try:
        logger.debug("Waiting for reply...")
        reply = comms.recv_message(sock)
        
        if reply is None:
            logger.error("Received None instead of bytes. Connection might be closed.")
            sock.close()
            return False
        
        logger.debug(f"Got reply: {reply} (length: {len(reply)})")
        
        ok, = struct.unpack('!I', reply)
        if not ok:
            logger.error("Server reported failure")
            sock.close()
            return False
            
        logger.debug("Interaction request successful")
    except Exception as e:
        logger.error(f"Error receiving/processing reply: {e}")
        sock.close()
        return False
    
    sock.close()
    return True

INTERACTOR_PORT = 31415

def run_interactor(ip, port, interactor_port=INTERACTOR_PORT):
    """Run the interactor with enhanced error handling and debugging."""
    logger.info(f"Running interactor for {ip}:{port} (interactor port: {interactor_port})")
    
    # Try to find existing instance
    try:
        logger.debug(f"Trying to connect to existing instance on port {interactor_port}")
        InstanceManager.add_existing_instance(interactor_port)
        instance = InstanceManager.get_instance(-1)
        logger.info(f"Connected to existing instance: {instance}")
    except AssertionError as e:
        logger.warning(f"No existing interactor found on port {interactor_port}. Starting a new one.")
        try:
            instance = MinecraftInstance(interactor_port)
            instance.launch(daemonize=True)
            # Give it time to start up
            time.sleep(10)
            logger.info(f"Started new instance: {instance}")
        except Exception as e:
            logger.error(f"Failed to start new instance: {e}")
            return False
    
    # Request interactor
    success = request_interactor(instance, f'{ip}:{port}')
    
    if success:
        logger.info(f"Interactor successfully set up for {ip}:{port}")
        return True
    else:
        logger.error(f"Failed to set up interactor for {ip}:{port}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Connect to an agent server.')
    parser.add_argument('port', type=int, default=8888,
                        help='The minecraft server port to connect to.')
    parser.add_argument('-i', '--ip', default='127.0.0.1',
                        help='The ip to connect to.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging.')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse args
    opts = parse_args()
    
    # Set debug level if requested
    if opts.debug:
        coloredlogs.install(level=logging.DEBUG)
    
    # Run interactor
    success = run_interactor(ip=opts.ip, port=opts.port)
    
    # Exit with appropriate status
    sys.exit(0 if success else 1)