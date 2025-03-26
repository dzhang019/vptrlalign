import matplotlib.pyplot as plt
import json
import numpy as np
import os

def parse_jsonl_file(file_path):
    """Parse JSON lines from a file."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON in {file_path}.")
        return []

def plot_rl_metrics(episode_file, loss_file):
    """Plot reward, episode length, and loss metrics as a function of update number."""
    # Load data from files
    episode_data = parse_jsonl_file(episode_file) if episode_file else []
    loss_data = parse_jsonl_file(loss_file) if loss_file else []
    
    print(f"Found episode data: {len(episode_data)} entries from {episode_file}")
    print(f"Found loss data: {len(loss_data)} entries from {loss_file}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 12))
    
    # First set - Episode metrics
    ax1 = fig.add_subplot(3, 1, 1)  # Reward
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)  # Episode length
    
    # Second set - Loss metrics
    ax3 = fig.add_subplot(3, 1, 3)  # Losses
    
    # ---- Plot Episode Data ----
    if episode_data:
        updates_ep = [entry['update'] for entry in episode_data]
        rewards = [entry['total_reward'] for entry in episode_data]
        lengths = [entry['length'] for entry in episode_data]
        
        # Plot reward
        ax1.plot(updates_ep, rewards, 'o-', color='blue', linewidth=2, markersize=6)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.set_title('RL Episode Metrics by Update', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add best fit line for rewards
        if len(updates_ep) > 1:
            update_array_ep = np.array(updates_ep)
            z1 = np.polyfit(update_array_ep, rewards, 1)
            p1 = np.poly1d(z1)
            ax1.plot(update_array_ep, p1(update_array_ep), 'r--', 
                    label=f'Trend: {z1[0]:.5f}x + {z1[1]:.2f}')
            ax1.legend()
        
        # Plot episode length
        ax2.plot(updates_ep, lengths, 'o-', color='green', linewidth=2, markersize=6)
        ax2.set_ylabel('Episode Length', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add best fit line for lengths
        if len(updates_ep) > 1:
            z2 = np.polyfit(update_array_ep, lengths, 1)
            p2 = np.poly1d(z2)
            ax2.plot(update_array_ep, p2(update_array_ep), 'r--', 
                    label=f'Trend: {z2[0]:.2f}x + {z2[1]:.2f}')
            ax2.legend()
    else:
        ax1.text(0.5, 0.5, "No episode data found", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No episode data found", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    
    # ---- Plot Loss Data ----
    if loss_data:
        updates_loss = [entry['update'] for entry in loss_data]
        policy_losses = [entry['policy_loss'] for entry in loss_data]
        value_losses = [entry['value_loss'] for entry in loss_data]
        kl_losses = [entry['kl_loss'] for entry in loss_data]
        lambdas = [entry['lambda_kl'] for entry in loss_data]
        
        # Plot losses
        ax3.plot(updates_loss, policy_losses, 'o-', color='blue', linewidth=2, markersize=6, label='Policy Loss')
        ax3.plot(updates_loss, value_losses, 'o-', color='green', linewidth=2, markersize=6, label='Value Loss')
        ax3.plot(updates_loss, kl_losses, 'o-', color='purple', linewidth=2, markersize=6, label='KL Loss')
        
        # Create a second y-axis for lambda_kl
        ax3_twin = ax3.twinx()
        ax3_twin.plot(updates_loss, lambdas, 'o-', color='red', linewidth=2, markersize=6, label='Lambda KL')
        ax3_twin.set_ylabel('Lambda KL', fontsize=12, color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')
        
        # Set labels and grid for losses
        ax3.set_xlabel('Update Number', fontsize=12)
        ax3.set_ylabel('Loss Values', fontsize=12)
        ax3.set_title('Training Losses by Update', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Combine legends from both y-axes
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax3.text(0.5, 0.5, "No loss data found", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    
    # Add overall x-axis label for first two plots
    ax2.set_xlabel('Update Number', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('rl_training_metrics.png', dpi=300)
    plt.show()

def print_current_dir_info():
    """Print information about the current working directory to help locate files."""
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    print("\nContents of current directory:")
    for item in os.listdir(cwd):
        if os.path.isdir(os.path.join(cwd, item)):
            print(f"  - {item}/ (directory)")
        else:
            print(f"  - {item}")
    
    debug_logs_path = os.path.join(cwd, "debug_logs")
    if os.path.isdir(debug_logs_path):
        print("\nContents of debug_logs directory:")
        for item in os.listdir(debug_logs_path):
            if os.path.isdir(os.path.join(debug_logs_path, item)):
                print(f"  - {item}/ (directory)")
            else:
                print(f"  - {item}")
    
    return cwd

def main():
    # Print info about current directory
    base_dir = print_current_dir_info()
    
    # Set file paths for the specific jsonl files
    episode_file = os.path.join(base_dir, "debug_logs", "episodes.jsonl")
    loss_file = os.path.join(base_dir, "debug_logs", "training_debug.jsonl")
    
    # Check if the files exist
    episode_exists = os.path.isfile(episode_file)
    loss_exists = os.path.isfile(loss_file)
    
    print(f"\nChecking file paths:")
    print(f"  Episode file: {episode_file} - {'Found' if episode_exists else 'Not found'}")
    print(f"  Loss file: {loss_file} - {'Found' if loss_exists else 'Not found'}")
    
    if not episode_exists or not loss_exists:
        print("\nSuggested actions:")
        print("  1. Make sure the files are in the correct location")
        print("  2. You may need to modify the file paths in the script")
        
        # Try alternative paths - maybe they're directly under debug_logs
        print("\nTrying alternative paths...")
        
        episode_alt = os.path.join(base_dir, "debug_logs", "episodes.jsonl")
        loss_alt = os.path.join(base_dir, "debug_logs", "training_debug.jsonl")
        
        episode_alt_exists = os.path.isfile(episode_alt)
        loss_alt_exists = os.path.isfile(loss_alt)
        
        print(f"  Alternative episode file: {episode_alt} - {'Found' if episode_alt_exists else 'Not found'}")
        print(f"  Alternative loss file: {loss_alt} - {'Found' if loss_alt_exists else 'Not found'}")
        
        if episode_alt_exists:
            episode_file = episode_alt
        if loss_alt_exists:
            loss_file = loss_alt
    
    # Only proceed if at least one file is found
    if os.path.isfile(episode_file) or os.path.isfile(loss_file):
        plot_rl_metrics(episode_file, loss_file)
    else:
        print("\nNo data files found. Cannot generate plots.")

if __name__ == "__main__":
    main()