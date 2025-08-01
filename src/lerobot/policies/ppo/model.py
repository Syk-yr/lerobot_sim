from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import collections
import gymnasium
from packaging import version

import torch

from lerobot.policies.ppo import config, logger
from lerobot.policies.ppo.spaces import (
    compute_space_size,
    flatten_tensorized_space,
    sample_space,
)

from lerobot.policies.ppo.spaces import unflatten_tensorized_space

class Model(torch.nn.Module):
    def __init__(
        self,
        observation_space: Union[int, Sequence[int], gymnasium.Space],
        action_space: Union[int, Sequence[int], gymnasium.Space],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (torch.device): Device to be used for the computations
        - ``observation_space`` (int, sequence of int, gymnasium.Space): Observation/state space
        - ``action_space`` (int, sequence of int, gymnasium.Space): Action space
        - ``num_observations`` (int): Number of elements in the observation/state space
        - ``num_actions`` (int): Number of elements in the action space

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gymnasium.Space
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional

        Custom models should override the ``act`` method::

            import torch
            from skrl.models.torch import Model

            class CustomModel(Model):
                def __init__(self, observation_space, action_space, device="cuda:0"):
                    Model.__init__(self, observation_space, action_space, device)

                    self.layer_1 = nn.Linear(self.num_observations, 64)
                    self.layer_2 = nn.Linear(64, self.num_actions)

                def act(self, inputs, role=""):
                    x = F.relu(self.layer_1(inputs["states"]))
                    x = F.relu(self.layer_2(x))
                    return x, None, {}
        """
        super(Model, self).__init__()

        self.device = config.torch.parse_device(device)

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else compute_space_size(observation_space)
        self.num_actions = None if action_space is None else compute_space_size(action_space)

        self._random_distribution = None

    def init_state_dict(self, role: str, inputs: Mapping[str, torch.Tensor] = {}) -> None:
        if not inputs:
            inputs = {
                "states": flatten_tensorized_space(
                    sample_space(self.observation_space, backend="native", device=self.device)
                ),
                "taken_actions": flatten_tensorized_space(
                    sample_space(self.action_space, backend="native", device=self.device)
                ),
            }
        # init parameters
        self.to(device=self.device)
        # self.compute = unflatten_tensorized_space
        self.compute(inputs=inputs, role=role)

    def tensor_to_space(
        self, tensor: torch.Tensor, space: gymnasium.Space, start: int = 0
    ) -> Union[torch.Tensor, dict]:
        return unflatten_tensorized_space(space, tensor)

    def random_act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, None, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dict
        """
        # discrete action space (Discrete)
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            return torch.randint(self.action_space.n, (inputs["states"].shape[0], 1), device=self.device), None, {}
        # continuous action space (Box)
        elif isinstance(self.action_space, gymnasium.spaces.Box):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(self.action_space.low[0], device=self.device, dtype=torch.float32),
                    high=torch.tensor(self.action_space.high[0], device=self.device, dtype=torch.float32),
                )

            return (
                self._random_distribution.sample(sample_shape=(inputs["states"].shape[0], self.num_actions)),
                None,
                {},
            )
        else:
            raise NotImplementedError(f"Action space type ({type(self.action_space)}) not supported")

    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: ``"normal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all parameters with an orthogonal distribution with a gain of 0.5
            >>> model.init_parameters("orthogonal_", gain=0.5)

            # initialize all parameters as a sparse matrix with a sparsity of 0.1
            >>> model.init_parameters("sparse_", sparsity=0.1)
        """
        for parameters in self.parameters():
            exec(f"torch.nn.init.{method_name}(parameters, *args, **kwargs)")

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: ``"orthogonal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all weights with uniform distribution in range [-0.1, 0.1]
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # initialize all weights with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_weights(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_weights(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.weight, *args, **kwargs)")

        _update_weights(self.children(), method_name, args, kwargs)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """Initialize the model biases according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: ``"constant_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all biases with a constant value (0)
            >>> model.init_biases(method_name="constant_", val=0)

            # initialize all biases with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_biases(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_biases(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.bias, *args, **kwargs)")

        _update_biases(self.children(), method_name, args, kwargs)

    def get_specification(self) -> Mapping[str, Any]:
        """Returns the specification of the model

        The following keys are used by the agents for initialization:

        - ``"rnn"``: Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells

          - ``"sizes"``: List of RNN shapes (number of layers, number of environments, number of features in the RNN state).
            There must be as many tuples as there are states in the recurrent layer/cell. E.g., LSTM has 2 states (hidden and cell).

        :return: Dictionary containing advanced specification of the model
        :rtype: dict

        Example::

            # model with a LSTM layer.
            # - number of layers: 1
            # - number of environments: 4
            # - number of features in the RNN state: 64
            >>> model.get_specification()
            {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        """
        return {}

    def forward(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Forward pass of the model

        This method calls the ``.act()`` method and returns its outputs

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        return self.act(inputs, role)

    def compute(self, inputs, role=""):
        if role == "policy":
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            net = self.net_container(states)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                net = self.net_container(states)
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        This method is currently implemented by the helper models (**GaussianModel**, etc.).
        The classes that inherit from the latter must only implement the ``.compute()`` method

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Child class must implement this method

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        logger.warning("Make sure to place Mixins before Model during model definition")
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation.
            See `torch.nn.Module.train <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path

        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: ``None``).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")

            # save an older version of the model to the specified path
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)
        """
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str

        Example::

            # load the model onto the CPU
            >>> model = Model(observation_space, action_space, device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(observation_space, action_space, device="cuda:1")
            >>> model.load("model.pt")
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def migrate(
        self,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        path: Optional[str] = None,
        name_map: Mapping[str, str] = {},
        auto_mapping: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Migrate the specified extrernal model's state dict to the current model

        The final storage device is determined by the constructor of the model

        Only one of ``state_dict`` or ``path`` can be specified.
        The ``path`` parameter allows automatic loading the ``state_dict`` only from files generated
        by the *rl_games* and *stable-baselines3* libraries at the moment

        For ambiguous models (where 2 or more parameters, for source or current model, have equal shape)
        it is necessary to define the ``name_map``, at least for those parameters, to perform the migration successfully

        :param state_dict: External model's state dict to migrate from (default: ``None``)
        :type state_dict: Mapping[str, torch.Tensor], optional
        :param path: Path to the external checkpoint to migrate from (default: ``None``)
        :type path: str, optional
        :param name_map: Name map to use for the migration (default: ``{}``).
                         Keys are the current parameter names and values are the external parameter names
        :type name_map: Mapping[str, str], optional
        :param auto_mapping: Automatically map the external state dict to the current state dict (default: ``True``)
        :type auto_mapping: bool, optional
        :param verbose: Show model names and migration (default: ``False``)
        :type verbose: bool, optional

        :raises ValueError: If neither or both of ``state_dict`` and ``path`` parameters have been set
        :raises ValueError: If the correct file type cannot be identified from the ``path`` parameter

        :return: True if the migration was successful, False otherwise.
                 Migration is successful if all parameters of the current model are found in the external model
        :rtype: bool

        Example::

            # migrate a rl_games checkpoint with unambiguous state_dict
            >>> model.migrate(path="./runs/Ant/nn/Ant.pth")
            True

            # migrate a rl_games checkpoint with ambiguous state_dict
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", verbose=False)
            [skrl:WARNING] Ambiguous match for log_std_parameter <- [value_mean_std.running_mean, value_mean_std.running_var, a2c_network.sigma]
            [skrl:WARNING] Ambiguous match for net.0.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.2.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.4.weight <- [a2c_network.value.weight, a2c_network.mu.weight]
            [skrl:WARNING] Ambiguous match for net.4.bias <- [a2c_network.value.bias, a2c_network.mu.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.0.bias -> [net.0.bias, net.2.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.2.bias -> [net.0.bias, net.2.bias]
            False
            >>> name_map = {"log_std_parameter": "a2c_network.sigma",
            ...             "net.0.bias": "a2c_network.actor_mlp.0.bias",
            ...             "net.2.bias": "a2c_network.actor_mlp.2.bias",
            ...             "net.4.weight": "a2c_network.mu.weight",
            ...             "net.4.bias": "a2c_network.mu.bias"}
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", name_map=name_map, verbose=True)
            [skrl:INFO] Models
            [skrl:INFO]   |-- current: 7 items
            [skrl:INFO]   |    |-- log_std_parameter : torch.Size([1])
            [skrl:INFO]   |    |-- net.0.weight : torch.Size([32, 4])
            [skrl:INFO]   |    |-- net.0.bias : torch.Size([32])
            [skrl:INFO]   |    |-- net.2.weight : torch.Size([32, 32])
            [skrl:INFO]   |    |-- net.2.bias : torch.Size([32])
            [skrl:INFO]   |    |-- net.4.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- net.4.bias : torch.Size([1])
            [skrl:INFO]   |-- source: 15 items
            [skrl:INFO]   |    |-- value_mean_std.running_mean : torch.Size([1])
            [skrl:INFO]   |    |-- value_mean_std.running_var : torch.Size([1])
            [skrl:INFO]   |    |-- value_mean_std.count : torch.Size([])
            [skrl:INFO]   |    |-- running_mean_std.running_mean : torch.Size([4])
            [skrl:INFO]   |    |-- running_mean_std.running_var : torch.Size([4])
            [skrl:INFO]   |    |-- running_mean_std.count : torch.Size([])
            [skrl:INFO]   |    |-- a2c_network.sigma : torch.Size([1])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.weight : torch.Size([32, 4])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.bias : torch.Size([32])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.weight : torch.Size([32, 32])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.bias : torch.Size([32])
            [skrl:INFO]   |    |-- a2c_network.value.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- a2c_network.value.bias : torch.Size([1])
            [skrl:INFO]   |    |-- a2c_network.mu.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- a2c_network.mu.bias : torch.Size([1])
            [skrl:INFO] Migration
            [skrl:INFO]   |-- map:  log_std_parameter <- a2c_network.sigma
            [skrl:INFO]   |-- auto: net.0.weight <- a2c_network.actor_mlp.0.weight
            [skrl:INFO]   |-- map:  net.0.bias <- a2c_network.actor_mlp.0.bias
            [skrl:INFO]   |-- auto: net.2.weight <- a2c_network.actor_mlp.2.weight
            [skrl:INFO]   |-- map:  net.2.bias <- a2c_network.actor_mlp.2.bias
            [skrl:INFO]   |-- map:  net.4.weight <- a2c_network.mu.weight
            [skrl:INFO]   |-- map:  net.4.bias <- a2c_network.mu.bias
            False

            # migrate a stable-baselines3 checkpoint with unambiguous state_dict
            >>> model.migrate(path="./ddpg_pendulum.zip")
            True

            # migrate from any exported model by loading its state_dict (unambiguous state_dict)
            >>> state_dict = torch.load("./external_model.pt")
            >>> model.migrate(state_dict=state_dict)
            True
        """
        if (state_dict is not None) + (path is not None) != 1:
            raise ValueError("Exactly one of state_dict or path may be specified")

        # load state_dict from path
        if path is not None:
            state_dict = {}
            # rl_games checkpoint
            if path.endswith(".pt") or path.endswith(".pth"):
                checkpoint = torch.load(path, map_location=self.device)
                if type(checkpoint) is dict:
                    state_dict = checkpoint.get("model", {})
            # stable-baselines3
            elif path.endswith(".zip"):
                import zipfile

                try:
                    archive = zipfile.ZipFile(path, "r")
                    with archive.open("policy.pth", mode="r") as file:
                        state_dict = torch.load(file, map_location=self.device)
                except KeyError as e:
                    logger.warning(str(e))
                    state_dict = {}
            else:
                raise ValueError("Cannot identify file type")

        # show state_dict
        if verbose:
            logger.info("Models")
            logger.info(f"  |-- current: {len(self.state_dict().keys())} items")
            for name, tensor in self.state_dict().items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info(f"  |-- source: {len(state_dict.keys())} items")
            for name, tensor in state_dict.items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info("Migration")

        # migrate the state_dict to current model
        new_state_dict = collections.OrderedDict()
        match_counter = collections.defaultdict(list)
        used_counter = collections.defaultdict(list)
        for name, tensor in self.state_dict().items():
            for external_name, external_tensor in state_dict.items():
                # mapped names
                if name_map.get(name, "") == external_name:
                    if tensor.shape == external_tensor.shape:
                        new_state_dict[name] = external_tensor
                        match_counter[name].append(external_name)
                        used_counter[external_name].append(name)
                        if verbose:
                            logger.info(f"  |-- map:  {name} <- {external_name}")
                        break
                    else:
                        logger.warning(
                            f"Shape mismatch for {name} <- {external_name} : {tensor.shape} != {external_tensor.shape}"
                        )
                # auto-mapped names
                if auto_mapping and name not in name_map:
                    if tensor.shape == external_tensor.shape:
                        if name.endswith(".weight"):
                            if external_name.endswith(".weight"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        elif name.endswith(".bias"):
                            if external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        else:
                            if not external_name.endswith(".weight") and not external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")

        # show ambiguous matches
        status = True
        for name, tensor in self.state_dict().items():
            if len(match_counter.get(name, [])) > 1:
                logger.warning("Ambiguous match for {} <- [{}]".format(name, ", ".join(match_counter.get(name, []))))
                status = False
        # show missing matches
        for name, tensor in self.state_dict().items():
            if not match_counter.get(name, []):
                logger.warning(f"Missing match for {name}")
                status = False
        # show multiple uses
        for name, tensor in state_dict.items():
            if len(used_counter.get(name, [])) > 1:
                logger.warning("Multiple use of {} -> [{}]".format(name, ", ".join(used_counter.get(name, []))))
                status = False

        # load new state dict
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()

        return status

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (``parameters.requires_grad = False``)
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``)

        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them (default: ``True``)
        :type freeze: bool, optional

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, polyak: float = 1) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: Model used to update the internal parameters
        :type model: torch.nn.Module (skrl.models.torch.Model)
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        :type polyak: float, optional

        Example::

            # hard update (from source model)
            >>> model.update_parameters(source_model)

            # soft update (from source model)
            >>> model.update_parameters(source_model, polyak=0.005)
        """
        with torch.no_grad():
            # hard update
            if polyak == 1:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.copy_(model_parameters.data)
            # soft update (use in-place operations to avoid creating new parameters)
            else:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.mul_(1 - polyak)
                    parameters.data.add_(polyak * model_parameters.data)

    def broadcast_parameters(self, rank: int = 0):
        """Broadcast model parameters to the whole group (e.g.: across all nodes) in distributed runs

        After calling this method, the distributed model will contain the broadcasted parameters from ``rank``

        :param rank: Worker/process rank from which to broadcast model parameters (default: ``0``)
        :type rank: int

        Example::

            # broadcast model parameter from worker/process with rank 1
            >>> if config.torch.is_distributed:
            ...     model.broadcast_parameters(rank=1)
        """
        object_list = [self.state_dict()]
        torch.distributed.broadcast_object_list(object_list, rank)
        self.load_state_dict(object_list[0])

    def reduce_parameters(self):
        """Reduce model parameters across all workers/processes in the whole group (e.g.: across all nodes)

        After calling this method, the distributed model parameters will be bitwise identical for all workers/processes

        Example::

            # reduce model parameter across all workers/processes
            >>> if config.torch.is_distributed:
            ...     model.reduce_parameters()
        """
        # batch all_reduce ops: https://github.com/entity-neural-network/incubator/pull/220
        gradients = []
        for parameters in self.parameters():
            if parameters.grad is not None:
                gradients.append(parameters.grad.view(-1))
        gradients = torch.cat(gradients)

        torch.distributed.all_reduce(gradients, op=torch.distributed.ReduceOp.SUM)

        offset = 0
        for parameters in self.parameters():
            if parameters.grad is not None:
                parameters.grad.data.copy_(
                    gradients[offset : offset + parameters.numel()].view_as(parameters.grad.data)
                    / config.torch.world_size
                )
                offset += parameters.numel()
