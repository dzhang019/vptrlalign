import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy_mod import MinecraftAgentPolicy
from lib.torch_util import default_device_type, set_default_torch_device
from lib.tree_util import tree_map
print(f"GPU detected: {th.cuda.is_available()}")

# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

# Valid MineRL environment parameters
ENV_KWARGS = {
    'fovRange': [70, 70],
    'gammaRange': [2, 2],
    'texturePack': "default",
    'renderResolution': 128
}
TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def validate_env(env):
    """Check that the MineRL environment is setup correctly, and raise if not"""
    for key, value in ENV_KWARGS.items():
        if key == "frameskip":
            continue
        if getattr(env.task, key) != value:
            raise ValueError(f"MineRL environment setting {key} does not match {value}")
    action_names = set(env.action_space.spaces.keys())
    if action_names != set(TARGET_ACTION_SPACE.keys()):
        raise ValueError(f"MineRL action space does match. Expected actions {set(TARGET_ACTION_SPACE.keys())}")

    for ac_space_name, ac_space_space in TARGET_ACTION_SPACE.items():
        if env.action_space.spaces[ac_space_name] != ac_space_space:
            raise ValueError(f"MineRL action space setting {ac_space_name} does not match {ac_space_space}")


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


class MineRLAgent:
    """Agent class that interfaces between the MineRL environment and the policy"""
    def __init__(self, env, device=None, policy_kwargs=None, pi_head_kwargs=None):
        # Validate that the environment matches expected configuration
        validate_env(env)

        # Set up device
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        
        # Set up action mappings
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # Use default hyperparameters if not provided
        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        # Create policy
        agent_kwargs = dict(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            action_space=action_space
        )
        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        
        # Initialize hidden state
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)
        
        # Check if model has auxiliary value head
        self.has_aux_head = hasattr(self.policy, 'aux_value_head')
        print(f"Agent initialized with {'an auxiliary' if self.has_aux_head else 'no auxiliary'} value head")

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        weights = th.load(path, map_location=self.device)
        
        # Print keys for debugging
        weight_keys = list(weights.keys())
        print(f"Loading weights with keys: {weight_keys[:5]}... (total: {len(weight_keys)} keys)")
        
        # Check for auxiliary value head weights
        aux_keys = [k for k in weight_keys if 'aux' in k.lower()]
        if aux_keys and not self.has_aux_head:
            print(f"WARNING: Found auxiliary value head weights {aux_keys} but model doesn't have aux_value_head")
        elif not aux_keys and self.has_aux_head:
            print(f"WARNING: Model has auxiliary value head but weights don't contain aux keys")
            
        # Load weights, allowing for missing keys (strict=False)
        self.policy.load_state_dict(weights, strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation
        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Convert MineRL action to agent's format.
        
        Args:
            minerl_action_transformed: Action in MineRL format
            to_torch: Whether to convert to torch tensors (otherwise numpy)
            check_if_null: Whether to check if action is null
            
        Returns:
            Action in agent's format (dict with buttons and camera)
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action

    def get_action(self, minerl_obs):
        """
        Basic agent action selection.
        
        Args:
            minerl_obs: Observation from MineRL environment
            
        Returns:
            Action in MineRL format
        """
        agent_input = self._env_obs_to_agent(minerl_obs)
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True
        )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action

    def get_action_and_training_info(self, minerl_obs, hidden_state, stochastic=True, taken_action=None):
        """
        Single-step logic that wraps policy.act_train(...).
        You can call this for:
        - Environment stepping: pass 'taken_action=None' to sample an action.
        - Re-forward at train time: pass 'taken_action' for the forced action.

        Args:
            minerl_obs: Observation from MineRL environment
            hidden_state: Hidden state for current step
            stochastic: Whether to sample actions stochastically
            taken_action: Optional forced action in MineRL format
            
        Returns:
            minerl_action: Action in MineRL format
            pi_dist: Policy distribution
            vpred: Value prediction
            aux_vpred: Auxiliary value prediction (if model has it)
            log_prob: Log probability of action
            new_hidden_state: New hidden state
        """
        # Convert MineRL observation to agent's input
        agent_input = self._env_obs_to_agent(minerl_obs)

        # Convert forced action to agent's format if provided
        forced_action_torch = None
        if taken_action is not None:
            forced_action_torch = self._env_action_to_agent(taken_action, to_torch=True)

        # Forward pass through policy
        agent_action, new_hidden_state, result = self.policy.act_train(
            obs=agent_input,
            first=self._dummy_first,
            state_in=hidden_state,
            stochastic=stochastic,
            taken_action=forced_action_torch,
            return_pd=True
        )

        # Convert agent's action to MineRL format
        minerl_action = self._agent_action_to_env(agent_action)

        # Extract results
        log_prob = result["log_prob"]
        vpred = result["vpred"]
        pi_dist = result["pd"]
        
        # Include auxiliary value prediction if available
        if "aux_vpred" in result:
            aux_vpred = result["aux_vpred"]
            return minerl_action, pi_dist, vpred, aux_vpred, log_prob, new_hidden_state
        else:
            # For backwards compatibility, return without aux_vpred
            return minerl_action, pi_dist, vpred, log_prob, new_hidden_state

    def get_sequence_and_training_info(
        self,
        minerl_obs_list,        # List of T raw MineRL obs
        initial_hidden_state,   # The hidden state at step=0
        stochastic=True,
        taken_actions_list=None
    ):
        """
        Process a sequence of observations and optionally forced actions.
        
        Args:
            minerl_obs_list: List of T MineRL observations
            initial_hidden_state: Hidden state for first timestep
            stochastic: Whether to sample actions stochastically
            taken_actions_list: Optional list of T forced actions
            
        Returns:
            pi_dist_seq: Sequence of T policy distributions
            vpred_seq: Sequence of T value predictions
            aux_vpred_seq: Sequence of T auxiliary value predictions (if model has them)
            log_prob_seq: Sequence of T log probabilities
            final_hidden_state: Final hidden state after processing sequence
        """
        # Move initial hidden state to correct device
        initial_hidden_state = tree_map(lambda x: x.to(self.device), initial_hidden_state)
        
        # Convert each observation to agent's format
        agent_inputs_list = []
        for minerl_obs in minerl_obs_list:
            agent_input = self._env_obs_to_agent(minerl_obs)   # {"img": Tensor[1, C, H, W]}
            agent_inputs_list.append(agent_input["img"])       # List of [1, C, H, W]

        # Stack along time dimension
        stacked_img = th.cat(agent_inputs_list, dim=0).to(self.device)  # [T, C, H, W]
        stacked_obs = {"img": stacked_img}

        # Convert forced actions if provided
        forced_actions_torch = None
        if taken_actions_list is not None:
            forced_actions_list_torch = []
            for taken_action in taken_actions_list:
                fa_torch = self._env_action_to_agent(taken_action, to_torch=True)
                forced_actions_list_torch.append(fa_torch)
            forced_actions_torch = forced_actions_list_torch

        # Forward pass through policy
        if self.has_aux_head:
            # With auxiliary value head
            results = self.policy.act_train_sequence(
                stacked_obs,
                initial_hidden_state,
                forced_actions=forced_actions_torch,
                stochastic=stochastic
            )
            pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hidden_state = results
            return pi_dist_seq, vpred_seq, aux_vpred_seq, log_prob_seq, final_hidden_state
        else:
            # Without auxiliary value head (for backwards compatibility)
            # This case might need adjustment based on your policy implementation
            results = self.policy.act_train_sequence(
                stacked_obs,
                initial_hidden_state,
                forced_actions=forced_actions_torch,
                stochastic=stochastic
            )
            
            # If 5 outputs are returned but we don't have an aux head, it may be a different format
            if len(results) == 5:
                pi_dist_seq, vpred_seq, _, log_prob_seq, final_hidden_state = results
            else:
                pi_dist_seq, vpred_seq, log_prob_seq, final_hidden_state = results
                
            return pi_dist_seq, vpred_seq, log_prob_seq, final_hidden_state
