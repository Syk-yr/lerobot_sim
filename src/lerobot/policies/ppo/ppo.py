from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.ppo import config, logger
from lerobot.policies.ppo.base import Agent
from lerobot.policies.ppo.random_memory import Memory
from lerobot.policies.ppo.model import Model
from lerobot.policies.ppo.kl_adaptive import KLAdaptiveLR
from lerobot.policies.ppo.running_standard_scaler import RunningStandardScaler
from lerobot.policies.ppo.gaussian import gaussian_model
from lerobot.policies.ppo.deterministic import deterministic_model
from lerobot.policies.ppo.gaussian import GaussianMixin
from lerobot.policies.ppo.runner import Runner
import gymnasium as gym
import torch
from math import inf
from gymnasium.spaces import Box
import numpy as np

# fmt: off
# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "seed": 42,
    "rollouts": 24,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 4,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-4,                  # learning rate
    "learning_rate_scheduler": 'KLAdaptiveLR',        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": 'RunningStandardScaler',             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {"size": Box(-inf, inf, shape=(2018,), dtype=np.float32)},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": 'RunningStandardScaler',             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {"size": 1},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 1.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": True,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.001,      # entropy loss scaling factor
    "value_loss_scale": 2.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "rewards_shaper_scale": 0.01,
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "lift_obj",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 0,   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 0,      # interval for checkpoints (timesteps)
    }
}
# [end-config-dict-torch]
# fmt: on
agent_cfg = {
    'seed': 42,
    'models': {
        'separate': False, 
        'policy': {
            'class': 'GaussianMixin', 
            'clip_actions': False, 
            'clip_log_std': True, 
            'min_log_std': -20.0, 
            'max_log_std': 2.0, 
            'initial_log_std': 0.0, 
            'network': [{
                'name': 'net',
                'input':'STATES',
                'layers': [512,256,128,64], 
                'activations':'elu',
            }],
            'output': 'ACTIONS'
        }, 
        'value': {
            'class': 'DeterministicMixin', 
            'clip_actions': False, 
            'network': [{
                'name': 'net',
                'input':'STATES',
                'layers': [512,256,128,64], 
                'activations':'elu',
            }],
            'output': 'ONE'
        }
    },
    'memory': {
        'class': 'RandomMemory', 
        'memory_size': -1
    },
    'agent': {
        'class': 'PPO', 
        'rollouts': 24, 
        'learning_epochs': 8, 
        'mini_batches': 4, 
        'discount_factor': 0.99, 
        'lambda': 0.95, 
        'learning_rate': 0.0001, 
        'learning_rate_scheduler': 'KLAdaptiveLR', 
        'learning_rate_scheduler_kwargs': {
            'kl_threshold': 0.01
        }, 
        'state_preprocessor': 
        'RunningStandardScaler', 
        'state_preprocessor_kwargs': None, 
        'value_preprocessor': 'RunningStandardScaler', 
        'value_preprocessor_kwargs': None, 
        'random_timesteps': 0, 
        'learning_starts': 0, 
        'grad_norm_clip': 1.0, 
        'ratio_clip': 0.2, 
        'value_clip': 0.2, 
        'clip_predicted_values': True, 
        'entropy_loss_scale': 0.001, 
        'value_loss_scale': 2.0, 
        'kl_threshold': 0.0, 
        'rewards_shaper_scale': 0.01, 
        'time_limit_bootstrap': False, 
        'experiment': {
            'directory': 'lift_obj', 
            'experiment_name': '', 
            'write_interval': 0, 
            'checkpoint_interval': 0
        }
    },
    'trainer': {
        'class': 'SequentialTrainer', 
        'timesteps': 72000, 
        'environment_info': 'log',
        "close_environment_at_exit":False
    },
}


class PPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type


        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor


    def act(self, states: torch.Tensor) -> torch.Tensor:
        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, outputs


from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.policies.ppo.images import Image_Features
class PPO_Policy():
    def __init__(self):
        # super().__init__(cfg)
        self.pt_path = "/home/yk/lerobot/src/lerobot/policies/ppo/2025-07-16_11-55-39_ppo_torch/checkpoints/agent_72000.pt"
        self.image_features = Image_Features()
        observation_space = Box(-inf, inf, shape=(2018,), dtype=np.float32)
        action_space = Box(-inf, inf, shape=(6,), dtype=np.float32)
        self.runner = Runner(observation_space, action_space, agent_cfg)
        self.last_action = torch.zeros((1,6), device="cuda")

    def reset(self):
        self.runner.agent.load(self.pt_path)
        self.runner.agent.set_running_mode("eval")

    def transform_image(self, img):
        import torch.nn.functional as F
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)  # (1, 3, 224, 224)
        img = img.permute(0, 2, 3, 1).contiguous()  # (1, 224, 224, 3)
        return img
    def select_action(self, obs):
        # 将1*3*480*640格式的图像obs["observation.images.front"] 转换为 1*224*224*3格式的图像
        # 假设输入为 torch.Tensor，形状为(1, 3, 480, 640)
        image_data1 = self.image_features.act(self.transform_image(obs["observation.images.front"]))
        image_data2 = self.image_features.act(self.transform_image(obs["observation.images.wrist"]))
        state = torch.cat([obs["observation.state"],obs["observation.vel"],self.last_action],dim=1)
        obs = torch.cat([state, image_data2, image_data1], dim=1)
        outputs = self.runner.agent.act(obs)
        actions = outputs[-1].get("mean_actions", outputs[0])
        self.last_action = actions
        return actions

if __name__ == "__main__":
    pt_path = "/home/yk/lerobot/src/lerobot/policies/ppo/2025-07-16_11-55-39_ppo_torch/checkpoints/agent_72000.pt"
    observation_space = Box(-inf, inf, shape=(2018,), dtype=np.float32)
    action_space = Box(-inf, inf, shape=(6,), dtype=np.float32)
    runner = Runner(observation_space, action_space, agent_cfg)
    runner.agent.load(pt_path)
    runner.agent.set_running_mode("eval")
    image_features = Image_Features()
    image_data1 = image_features.act(torch.randn((1,224,224,3),device="cuda"))
    image_data2 = image_features.act(torch.randn((1,224,224,3),device="cuda"))
    obs = torch.cat([image_data1, image_data2,torch.randn((1,18),device="cuda")], dim=1)
    outputs = runner.agent.act(obs)
    actions = outputs[-1].get("mean_actions", outputs[0])

