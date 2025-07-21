from typing import Any, Mapping, Type, Union

import copy

from lerobot.policies.ppo import logger, set_seed
from lerobot.policies.ppo.base import Agent
from lerobot.policies.ppo.base import Model

from lerobot.policies.ppo.kl_adaptive import KLAdaptiveLR  # noqa
from lerobot.policies.ppo.running_standard_scaler import RunningStandardScaler



class Runner:
    def __init__(self, observation_space, action_space,cfg: Mapping[str, Any]) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        self._cfg = cfg

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        self._models = self._generate_models(observation_space, action_space, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(observation_space, action_space, copy.deepcopy(self._cfg), self._models)

    @property
    def agent(self):
        """Agent instance"""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file

        :param path: File path

        :return: Loaded configuration, or an empty dict if an error has occurred
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "gaussianmixin":
            from lerobot.policies.ppo.gaussian import gaussian_model as component
        elif name == "deterministicmixin":
            from lerobot.policies.ppo.deterministic import deterministic_model as component
        elif name == "shared":
            from lerobot.policies.ppo.shared import shared_model as component
        # memory
        elif name == "randommemory":
            from lerobot.policies.ppo.random_memory import RandomMemory as component
        # agent
        elif name in ["ppo", "ppo_default_config"]:
            from lerobot.policies.ppo.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(
        self, observation_space, action_space, cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        multi_agent=False
        device = 'cuda'
        possible_agents =  ["agent"]
        observation_spaces =  {"agent": observation_space}
        action_spaces =  {"agent": action_space}
        state_spaces = {"agent": None}
        agent_class = 'ppo'

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            # shared models
            roles = list(models_cfg.keys())
            if len(roles) != 2:
                raise ValueError(
                    "Runner currently only supports shared models, made up of exactly two models. "
                    "Set 'separate' field to True to create non-shared models for the given cfg"
                )
            # get shared model structure and parameters
            structure = []
            parameters = []
            for role in roles:
                # get instantiator function and remove 'class' key
                model_structure = models_cfg[role].get("class")
                if not model_structure:
                    raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                del models_cfg[role]["class"]
                structure.append(model_structure)
                parameters.append(self._process_cfg(models_cfg[role]))
            model_class = self._component("Shared")
    
            # instantiate model
            models[agent_id][roles[0]] = model_class(
                observation_space=observation_spaces[agent_id],
                action_space=action_spaces[agent_id],
                device=device,
                structure=structure,
                roles=roles,
                parameters=parameters,
            )
            models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # initialize lazy modules' parameters
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role)

        return models

    def _generate_agent(
        self,
        observation_space,
        action_space,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """

        device = "cuda"
        num_envs = 1
        possible_agents =  ["agent"]
        observation_spaces =  {"agent": observation_space}
        action_spaces = {"agent": action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError(f"No 'class' field defined in 'agent' cfg")

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # single-agent configuration and instantiation
        
        if agent_class in ["ppo"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"].get("noise_kwargs", {}).update({"device": device})
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)


    def run(self, mode: str = "train") -> None:
        """Run the training/evaluation

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation (default: ``"train"``)

        :raises ValueError: The specified running mode is not valid
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
