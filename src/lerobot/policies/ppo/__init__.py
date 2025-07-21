from typing import Union

import logging
import os
import sys

import numpy as np

from typing import Optional

import os
import random
import sys
import time

import numpy as np


__all__ = ["__version__", "logger", "config"]


# read library version from metadata
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("skrl")
except ImportError:
    __version__ = "unknown"


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {
        logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
        logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
        logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
        logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
        logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m",
    }

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)


_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("skrl")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)


# machine learning framework configuration
class _Config(object):
    def __init__(self) -> None:
        """Machine learning framework specific configuration."""

        class PyTorch(object):
            def __init__(self) -> None:
                """PyTorch configuration."""
                # torch.distributed config
                self._local_rank = int(os.getenv("LOCAL_RANK", "0"))
                self._rank = int(os.getenv("RANK", "0"))
                self._world_size = int(os.getenv("WORLD_SIZE", "1"))
                self._is_distributed = self._world_size > 1
                # device
                self._device = f"cuda:{self._local_rank}"

                # set up distributed runs
                if self._is_distributed:
                    import torch

                    logger.info(
                        f"Distributed (rank: {self._rank}, local rank: {self._local_rank}, world size: {self._world_size})"
                    )
                    torch.distributed.init_process_group("nccl", rank=self._rank, world_size=self._world_size)
                    torch.cuda.set_device(self._local_rank)

            @staticmethod
            def parse_device(device: Union[str, "torch.device", None], validate: bool = True) -> "torch.device":
                """Parse the input device and return a :py:class:`~torch.device` instance.

                :param device: Device specification. If the specified device is ``None`` or it cannot be resolved,
                               the default available device will be returned instead.
                :param validate: Whether to check that the specified device is valid. Since PyTorch does not check if
                                 the specified device index is valid, a tensor is created for the verification.

                :return: PyTorch device.
                """
                import torch

                _device = None
                if isinstance(device, torch.device):
                    _device = device
                elif isinstance(device, str):
                    try:
                        _device = torch.device(device)
                    except RuntimeError as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                if _device is None:
                    _device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )  # torch.get_default_device() was introduced in version 2.3.0
                # validate device
                if validate:
                    try:
                        torch.zeros((1,), device=_device)
                    except Exception as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                        _device = PyTorch.parse_device(None)
                return _device

            @property
            def device(self) -> "torch.device":
                """Default device.

                The default device, unless specified, is ``cuda:0`` (or ``cuda:LOCAL_RANK`` in a distributed environment)
                if CUDA is available, ``cpu`` otherwise.
                """
                self._device = self.parse_device(self._device, validate=False)
                return self._device

            @device.setter
            def device(self, device: Union[str, "torch.device"]) -> None:
                self._device = device

            @property
            def local_rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

                This property reads from the ``LOCAL_RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._local_rank

            @property
            def rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

                This property reads from the ``RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._rank

            @property
            def world_size(self) -> int:
                """The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

                This property reads from the ``WORLD_SIZE`` environment variable (``1`` if it doesn't exist).

                Read-only attribute.
                """
                return self._world_size

            @property
            def is_distributed(self) -> bool:
                """Whether if running in a distributed environment.

                This property is ``True`` when the PyTorch's distributed environment variable ``WORLD_SIZE > 1``.

                Read-only attribute.
                """
                return self._is_distributed

        self.torch = PyTorch()


config = _Config()
from lerobot.policies.ppo.spaces import unflatten_tensorized_space
def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    # generate a random seed
    seed = int(seed)

    # set different seeds in distributed runs
    if config.torch.is_distributed:
        seed += config.torch.rank

    logger.info(f"Seed: {seed}")

    # numpy
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
            # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8

            logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch seeding error: {e}")

    return seed





# agent_cfg = {
#     'seed': 42,
#     'models': {
#         'separate': False, 
#         'policy': {
#             'class': 'GaussianMixin', 
#             'clip_actions': False, 
#             'clip_log_std': True, 
#             'min_log_std': -20.0, 
#             'max_log_std': 2.0, 
#             'initial_log_std': 0.0, 
#             'network': [512,256,128,64], 
#             'output': 'ACTIONS'
#         }, 
#         'value': {
#             'class': 'DeterministicMixin', 
#             'clip_actions': False, 
#             'network': [512,256,128,64], 
#             'output': 'ONE'
#         }
#     },
#     'memory': {
#         'class': 'RandomMemory', 
#         'memory_size': -1
#     },
#     'agent': {
#         'class': 'PPO', 
#         'rollouts': 24, 
#         'learning_epochs': 8, 
#         'mini_batches': 4, 
#         'discount_factor': 0.99, 
#         'lambda': 0.95, 
#         'learning_rate': 0.0001, 
#         'learning_rate_scheduler': 'KLAdaptiveLR', 
#         'learning_rate_scheduler_kwargs': {
#             'kl_threshold': 0.01
#         }, 
#         'state_preprocessor': 
#         'RunningStandardScaler', 
#         'state_preprocessor_kwargs': None, 
#         'value_preprocessor': 'RunningStandardScaler', 
#         'value_preprocessor_kwargs': None, 
#         'random_timesteps': 0, 
#         'learning_starts': 0, 
#         'grad_norm_clip': 1.0, 
#         'ratio_clip': 0.2, 
#         'value_clip': 0.2, 
#         'clip_predicted_values': True, 
#         'entropy_loss_scale': 0.001, 
#         'value_loss_scale': 2.0, 
#         'kl_threshold': 0.0, 
#         'rewards_shaper_scale': 0.01, 
#         'time_limit_bootstrap': False, 
#         'experiment': {
#             'directory': 'lift_obj', 
#             'experiment_name': '', 
#             'write_interval': 0, 
#             'checkpoint_interval': 0
#         }
#     },
#     'trainer': {
#         'class': 'SequentialTrainer', 
#         'timesteps': 72000, 
#         'environment_info': 'log',
#         "close_environment_at_exit":False
#     },
# }
