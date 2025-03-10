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

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

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
    def __init__(self, env, device=None, policy_kwargs=None, pi_head_kwargs=None):
        validate_env(env)

        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
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
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
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
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True
        )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action
    '''
    def get_action_and_training_info(self, minerl_obs, stochastic=True):
        """
        Similar to get_action(...), but also returns additional info
        for RL training (policy distribution, value prediction, log-prob).

        This avoids doing a second forward pass for training:
        you get everything from the same call that determines the action.
        """
        # 1) Convert the MineRL environment observation into the agent's observation
        agent_input = self._env_obs_to_agent(minerl_obs)

        # 2) Call policy.act(...) exactly like in get_action(...), 
        #    but set return_pd=True so we get distribution info (pd).
        agent_action, new_hidden_state, result = self.policy.act_train(
            agent_input,
            self._dummy_first,     # same first-flag usage as get_action
            self.hidden_state,     # same hidden-state usage as get_action
            stochastic=stochastic,
            return_pd=True         # <--- We need the distribution for training
        )

        # 3) Convert the agent_action (PyTorch dict) to a MineRL env action
        minerl_action = self._agent_action_to_env(agent_action)

        # 4) Update the agent's hidden state internally (same as get_action does)
        self.hidden_state = tree_map(lambda x : x.detach(), new_hidden_state)


        # 5) Extract extras from 'result':
        #    By default, result["log_prob"] and result["vpred"] are 1D (batch_size=1).
        log_prob = result["log_prob"]
        vpred = result["vpred"]
        # pd was only returned because return_pd=True
        pi_dist = result.get("pd", None)

        # Return everything needed for both environment stepping + RL training
        return minerl_action, pi_dist, vpred, log_prob, new_hidden_state
        '''
    def get_action_and_training_info_old(self, minerl_obs, hidden_state, stochastic=True):
        # 1) Convert obs
        agent_input = self._env_obs_to_agent(minerl_obs)

        # 2) Single-step forward pass with the provided hidden_state
        #    (We do NOT use self.hidden_state or overwrite it internally)
        agent_action, new_hidden_state, result = self.policy.act_train(
            agent_input,
            self._dummy_first,
            hidden_state,  # use the passed-in hidden_state
            stochastic=stochastic,
            return_pd=True
        )

        minerl_action = self._agent_action_to_env(agent_action)

        log_prob = result["log_prob"]
        vpred = result["vpred"]
        pi_dist = result.get("pd", None)

        # Return everything, including new_hidden_state
        return minerl_action, pi_dist, vpred, log_prob, new_hidden_state
    
    def get_action_and_training_info(self, minerl_obs, hidden_state, stochastic=True, taken_action=None):
        """
        Single-step logic that wraps policy.act_train(...).
        You can call this for:
        - Environment stepping: pass 'taken_action=None' to sample an action.
        - Re-forward at train time: pass 'taken_action' for the forced action.

        Args:
        minerl_obs: the raw MineRL observation (dict with 'pov' etc.)
        hidden_state: the Transformer/LSTM hidden state for the current step
        stochastic: whether to sample (if no forced action)
        taken_action: if not None, a *factored* MineRL action that we force
                        for log_prob calculation.
        Returns:
        minerl_action: an action in MineRL environment format
        pi_dist: the distribution (if return_pd was True)
        vpred: predicted value (tensor)
        log_prob: log-prob of the chosen or forced action
        new_hidden_state: the updated hidden state after this step
        """

        # 1) Convert MineRL observation to agent's input
        agent_input = self._env_obs_to_agent(minerl_obs)

        # 2) Policy call. 
        #    If 'taken_action' is not None, we must convert that to the model's format 
        #    before passing it. But let's do that internally if needed:
        forced_action_torch = None
        if taken_action is not None:
            # Use the existing _env_action_to_agent(...) to convert to Tensors:
            forced_action_torch = self._env_action_to_agent(taken_action, to_torch=True)

        # 3) Single-step forward pass
        agent_action, new_hidden_state, result = self.policy.act_train(
            obs=agent_input,
            first=self._dummy_first,   # or pass a real 'first' if you track episode starts
            state_in=hidden_state,
            stochastic=stochastic,
            taken_action=forced_action_torch,  # forced action if not None
            return_pd=True
        )

        # 4) Convert the policy's action => MineRL env format
        minerl_action = self._agent_action_to_env(agent_action)

        log_prob = result["log_prob"]
        vpred = result["vpred"]
        pi_dist = result.get("pd", None)

        return minerl_action, pi_dist, vpred, log_prob, new_hidden_state
    def get_sequence_and_training_info(
        self,
        minerl_obs_list,        # List of T raw MineRL obs
        initial_hidden_state,   # The RNN/transformer hidden state at step=0
        stochastic=True,
        taken_actions_list=None
    ):
        """
        Multi-step logic for training:
        - Takes a full list of T environment observations
        - Optionally a list of forced actions (taken_actions_list) if you want
        to compute log-prob for each step.
        - Returns pi_dist_seq, vpred_seq, log_prob_seq, final_hidden_state
        each of which is length T or shaped [T, ...].
        """
        # print(f"Initial hidden state type: {type(initial_hidden_state)}")
        # print(f"Initial hidden state: {initial_hidden_state}")
        # def inspect_and_move(x):
        #     print(f"Leaf type: {type(x)}, value: {x}")
        #     return x.to(self.device)
        # initial_hidden_state = tree_map(inspect_and_move, initial_hidden_state)
        initial_hidden_state = tree_map(lambda x: x.to(self.device), initial_hidden_state)
        # 1) Convert each MineRL observation to agent input & stack
        #    Suppose you have T dicts like {"pov": ...}
        #    We'll produce a single batch dimension with shape [T, C, H, W].
        agent_inputs_list = []
        for minerl_obs in minerl_obs_list:
            agent_input = self._env_obs_to_agent(minerl_obs)   # shape { "img": Tensor[1, C, H, W] }
            agent_inputs_list.append(agent_input["img"])       # each is shape [1, C, H, W]

        # Stack into shape [T, C, H, W] (removing the batch=1 dimension)
        # So we do cat or stack along dim=0
        # Example if each agent_inputs_list[i] is [1, C, H, W], we remove the "1" dimension:
        stacked_img = th.cat(agent_inputs_list, dim=0).to(self.device)  # shape [T, C, H, W]

        # We now have a dict with a single "img" key for the entire sequence
        stacked_obs = {"img": stacked_img}

        # 2) If we have forced actions, convert them
        forced_actions_torch = None
        if taken_actions_list is not None:
            forced_actions_list_torch = []
            for taken_action in taken_actions_list:
                # Convert to model's format
                fa_torch = self._env_action_to_agent(taken_action, to_torch=True)
                forced_actions_list_torch.append(fa_torch)
            # We'll handle the shape inside act_train_sequence
            forced_actions_torch = forced_actions_list_torch

        # 3) Call the new multi-step forward
        pi_dist_seq, vpred_seq, log_prob_seq, final_hidden_state = self.policy.act_train_sequence(
            stacked_obs,              # shape [T, ...]
            initial_hidden_state,
            forced_actions=forced_actions_torch,
            stochastic=stochastic
        )

        return pi_dist_seq, vpred_seq, log_prob_seq, final_hidden_state


