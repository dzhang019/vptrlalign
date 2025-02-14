from argparse import ArgumentParser
import pickle
import time
import concurrent.futures
import torch as th
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from agent_mod import MineRLAgent, ENV_KWARGS
from lib.tree_util import tree_map
from lib.policy_mod import compute_kl_loss
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

# Optimized hyperparameters
NUM_ENVS = 8
ROLLOUT_STEPS = 32
BATCH_SIZE = 256
GRADIENT_ACCUM_STEPS = 2

th.backends.cudnn.benchmark = True
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

def load_model_parameters(path):
    params = pickle.load(open(path, "rb"))
    policy_args = params["model"]["args"]["net"]["args"]
    pi_head_args = params["model"]["args"]["pi_head_opts"]
    pi_head_args["temperature"] = float(pi_head_args["temperature"])
    return policy_args, pi_head_args

class GPUBatchManager:
    """Optimized batch processing with pinned memory and async transfers"""
    def __init__(self, num_envs, obs_space):
        self.num_envs = num_envs
        self.current_batch = 0
        
        # Pre-allocate pinned memory buffers
        self.obs_buffer = th.empty(
            (num_envs, *obs_space.shape), 
            dtype=th.uint8,
            pin_memory=True
        )
        
    def batch_obs(self, obs_list):
        """Batch observations using zero-copy pinned memory"""
        for i, obs in enumerate(obs_list):
            self.obs_buffer[i] = th.from_numpy(obs["pov"]).permute(2, 0, 1)
        return self.obs_buffer.to("cuda", non_blocking=True)

def train_rl(in_model, in_weights, out_weights, out_episodes, num_iterations=100):
    # Mixed precision setup
    scaler = GradScaler()
    th.autograd.set_detect_anomaly(False)
    
    # Environment setup
    dummy_env = HumanSurvival(**ENV_KWARGS).make()
    policy_kwargs, pi_head_kwargs = load_model_parameters(in_model)
    
    # Create agent with GPU-optimized init
    agent = MineRLAgent(dummy_env, device="cuda", 
                       policy_kwargs=policy_kwargs,
                       pi_head_kwargs=pi_head_kwargs).cuda()
    agent.load_weights(in_weights)
    
    # Create parallel envs with threadpool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ENVS) as executor:
        envs = [HumanSurvival(**ENV_KWARGS).make() for _ in range(NUM_ENVS)]
        obs_list = [env.reset() for env in envs]
        done_list = [False] * NUM_ENVS
        
        # GPU batch manager for efficient transfers
        batch_manager = GPUBatchManager(NUM_ENVS, dummy_env.observation_space)
        
        # Hidden states directly initialized on GPU
        hidden_states = [
            tree_map(lambda x: x.cuda(), agent.policy.initial_state(1))
            for _ in range(NUM_ENVS)
        ]
        
        # Optimized optimizer setup
        optimizer = th.optim.AdamW(agent.policy.parameters(), lr=1e-6, fused=True)
        
        # Training loop with async environment stepping
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Phase 1: Parallel environment steps
            futures = []
            for env_i in range(NUM_ENVS):
                if not done_list[env_i]:
                    # Async environment step
                    fut = executor.submit(
                        envs[env_i].step,
                        agent.policy.last_action[env_i] if hasattr(agent.policy, "last_action") 
                        else {}
                    )
                    futures.append((env_i, fut))
            
            # Batch process observations on GPU while envs are stepping
            batched_obs = batch_manager.batch_obs(obs_list)
            
            # Phase 2: Parallel policy inference
            with th.no_grad(), autocast():
                actions, hidden_states, log_probs, v_preds = agent.batch_policy(
                    batched_obs, 
                    th.stack([h[0] for h in hidden_states]),
                    th.stack([h[1] for h in hidden_states])
                )
            
            # Phase 3: Process environment results
            new_obs_list = [None] * NUM_ENVS
            for env_i, fut in futures:
                next_obs, reward, done, _ = fut.result()
                new_obs_list[env_i] = next_obs
                done_list[env_i] = done
            
            # Phase 4: Overlapped training update
            if iteration % GRADIENT_ACCUM_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)
                
            with autocast():
                # Compute losses here
                loss = compute_losses()
                scaler.scale(loss).backward()
                
            if (iteration + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
            
            # Phase 5: Async next obs batch prep
            next_batched_obs = batch_manager.batch_obs(new_obs_list)
            
            print(f"Iter {iteration}: {time.time()-start_time:.2f}s")

def compute_losses():
    # Implement your loss calculations here
    return th.tensor(0.0)  # Dummy return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True)
    parser.add_argument("--in-weights", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-episodes", default="episodes.txt")
    parser.add_argument("--num-iterations", type=int, default=100)
    args = parser.parse_args()
    
    train_rl(
        args.in_model,
        args.in_weights,
        args.out_weights,
        args.out_episodes,
        args.num_iterations
    )
