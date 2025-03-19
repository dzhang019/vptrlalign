"""
Implementation of Learning Without Forgetting (LwF) for MineRL agent training.
This extends the existing PPG implementation with knowledge distillation loss
to prevent catastrophic forgetting.
"""
import torch as th
import torch.nn.functional as F

class LwFHandler:
    """
    Handler for Learning Without Forgetting.
    
    This class manages the distillation process from the teacher model (original pretrained model)
    to the student model (model being fine-tuned).
    """
    def __init__(self, teacher_model, temperature=2.0, lambda_distill=0.5):
        """
        Initialize the LwF handler.
        
        Args:
            teacher_model: The pretrained model to distill knowledge from
            temperature: Temperature for softening the distributions
            lambda_distill: Weight of the distillation loss
        """
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        
        # Freeze the teacher model's parameters
        for param in teacher_model.policy.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(self, student_model, obs_batch, hidden_state=None):
        """
        Compute the distillation loss between teacher and student models.
        
        Args:
            student_model: The model being trained
            obs_batch: Batch of observations
            hidden_state: Hidden state for recurrent policies
            
        Returns:
            distillation_loss: The knowledge distillation loss
        """
        # Get teacher outputs
        with th.no_grad():
            if hidden_state is None:
                teacher_hidden = self.teacher_model.policy.initial_state(batch_size=1)
            else:
                teacher_hidden = hidden_state
                
            teacher_outputs = self.teacher_model.get_action_and_training_info(
                minerl_obs=obs_batch,
                hidden_state=teacher_hidden,
                stochastic=False,
                taken_action=None
            )
            
            # Extract teacher policy distributions
            teacher_actions = teacher_outputs[0]  # Action dict
        
        # Get student outputs (with gradients)
        if hidden_state is None:
            student_hidden = student_model.policy.initial_state(batch_size=1)
        else:
            student_hidden = hidden_state
            
        student_outputs = student_model.get_action_and_training_info(
            minerl_obs=obs_batch,
            hidden_state=student_hidden,
            stochastic=False,
            taken_action=None
        )
        
        # Extract student policy distributions
        student_actions = student_outputs[0]  # Action dict
        
        # Compute distillation loss across all action types
        distillation_losses = []
        
        # For each action type in the action space
        for action_name in student_actions:
            # Skip actions that aren't present in both models
            if action_name not in teacher_actions:
                continue
                
            # Handle different action types appropriately
            if isinstance(student_actions[action_name], th.Tensor) and student_actions[action_name].dtype == th.float32:
                # Continuous actions - use MSE loss
                student_action = student_actions[action_name]
                teacher_action = teacher_actions[action_name]
                
                # Apply temperature scaling
                # For continuous actions, we don't apply temperature directly but can use MSE
                action_loss = F.mse_loss(student_action, teacher_action)
                distillation_losses.append(action_loss)
                
            elif isinstance(student_actions[action_name], th.Tensor) and student_actions[action_name].dtype == th.int64:
                # Discrete actions - use KL divergence with softened distributions
                student_action = student_actions[action_name].float()
                teacher_action = teacher_actions[action_name].float()
                
                # Compute KL-divergence between softened distributions
                student_logits = student_action / self.temperature
                teacher_logits = teacher_action / self.temperature
                
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                # KL divergence (using direct formula for numerical stability)
                kl_div = th.sum(teacher_probs * (th.log(teacher_probs + 1e-10) - th.log(student_probs + 1e-10)))
                distillation_losses.append(kl_div)
                
        # Average the losses across all action types
        if distillation_losses:
            distillation_loss = sum(distillation_losses) / len(distillation_losses)
            return distillation_loss * (self.temperature ** 2) * self.lambda_distill
        else:
            # Return zero loss if no actions were processed
            return th.tensor(0.0, device="cuda")

def run_policy_update_with_lwf(agent, pretrained_policy, rollouts, optimizer, scaler, 
                          lwf_handler, train_unroll_fn, compute_kl_loss_fn, value_loss_coef=0.5, 
                          lambda_kl=0.2, max_grad_norm=1.0):
    """
    Extended version of run_policy_update that includes Learning Without Forgetting.
    
  Args:
        agent: The agent being trained
        pretrained_policy: Reference policy for KL divergence
        rollouts: List of rollouts to use for optimization
        optimizer: The optimizer to use
        scaler: Gradient scaler for mixed precision training
        lwf_handler: LwF handler for distillation loss
        train_unroll_fn: Function to process rollouts into transitions
        compute_kl_loss_fn: Function to compute KL divergence loss
        value_loss_coef: Coefficient for value function loss
        lambda_kl: Coefficient for KL divergence loss
        max_grad_norm: Maximum gradient norm for clipping
    """
    # Track statistics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl_loss = 0.0
    total_distill_loss = 0.0
    num_valid_envs = 0
    total_transitions = 0
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Process each environment's rollouts
    for env_idx, env_rollout in enumerate(rollouts):
        # Skip empty rollouts
        if len(env_rollout["obs"]) == 0:
            print(f"[Policy Update] Environment {env_idx} has no transitions, skipping")
            continue
        
        # Process rollout into transitions
        env_transitions = train_unroll_fn(
            agent,
            pretrained_policy,
            env_rollout,
            gamma=0.9999,
            lam=0.95
        )
        
        if len(env_transitions) == 0:
            continue
        
        # Extract data for this environment
        env_advantages = th.cat([th.tensor(t["advantage"], device="cuda").unsqueeze(0) 
                                for t in env_transitions])
        env_returns = th.tensor([t["return"] for t in env_transitions], device="cuda")
        env_log_probs = th.cat([t["log_prob"].unsqueeze(0) for t in env_transitions])
        env_v_preds = th.cat([t["v_pred"].unsqueeze(0) for t in env_transitions])
        
        # Normalize advantages
        env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-8)
        
        # Compute losses
        with th.autocast(device_type='cuda'):
            # Policy loss (Actor)
            policy_loss = -(env_advantages * env_log_probs).mean()
            
            # Value function loss (Critic)
            value_loss = ((env_v_preds - env_returns) ** 2).mean()
            
            # KL divergence loss between current and initial policy
            kl_losses = []
            for t in env_transitions:
                kl_loss = compute_kl_loss_fn(t["cur_pd"], t["old_pd"])
                kl_losses.append(kl_loss)
            kl_loss = th.stack(kl_losses).mean()
            
            # Calculate distillation loss (Learning Without Forgetting)
            distill_losses = []
            # Sample observations for distillation (use a subset to save computation)
            distill_indices = th.randperm(len(env_transitions))[:min(len(env_transitions), 10)]
            for idx in distill_indices:
                t = env_transitions[idx]
                hid_t = None  # Simplified for this example - full implementation would track hidden states
                distill_loss = lwf_handler.compute_distillation_loss(
                    student_model=agent, 
                    obs_batch=t["obs"],
                    hidden_state=hid_t
                )
                distill_losses.append(distill_loss)
            
            if distill_losses:
                distill_loss = th.stack(distill_losses).mean()
            else:
                distill_loss = th.tensor(0.0, device="cuda")
            
            # Total loss
            env_loss = policy_loss + (value_loss_coef * value_loss) + (lambda_kl * kl_loss) + distill_loss
        
        # Backward pass
        scaler.scale(env_loss).backward()
        
        # Update statistics
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl_loss += kl_loss.item()
        total_distill_loss += distill_loss.item()
        num_valid_envs += 1
        total_transitions += len(env_transitions)
    
    # Skip update if no valid transitions
    if num_valid_envs == 0:
        print("[Policy Update] No valid transitions, skipping update")
        return 0.0, 0.0, 0.0, 0.0, 0
    
    # Apply gradients
    scaler.unscale_(optimizer)
    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    # Compute averages
    avg_policy_loss = total_policy_loss / num_valid_envs
    avg_value_loss = total_value_loss / num_valid_envs
    avg_kl_loss = total_kl_loss / num_valid_envs
    avg_distill_loss = total_distill_loss / num_valid_envs
    
    return avg_policy_loss, avg_value_loss, avg_kl_loss, avg_distill_loss, total_transitions
