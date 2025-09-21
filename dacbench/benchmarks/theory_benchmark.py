import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import gymnasium as gym
import numpy as np
import pandas as pd
from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.theory import (
    RLSTheoryEnv,
    RLSTheoryEnvDiscrete,
    OLLGATheoryEnv,
    OLLGATheoryEnvDiscrete,
    OLLGAFactTheoryEnvDiscrete,
    OLLGATheoryPPOEnvDiscrete,
    OLLGAFactL1TheoryEnvDiscrete,
)

RLS_INFO = {
    "identifier": "RLSTheory",
    "name": "DAC benchmark with RLS algorithm and LeadingOnes problem",
    "reward": "Negative number of iterations until solution",
    "state_description": "specified by user",
}

RLS_THEORY_DEFAULTS = {
    "observation_description": "n, f(x)",  # examples: n, f(x), delta_f(x), optimal_k, k, k_{t-0..4}, f(x)_{t-1}, f(x)_{t-0..4}
    "reward_range": [-np.inf, np.inf],  # the true reward range is instance dependent
    "reward_choice": "imp_minus_evals",  # possible values: see envs/theory.py for more details
    "cutoff": 1e6,  # if using as a "train" environment, a cutoff of 0.8*n^2 where n is problem size will be used (for more details, please see https://arxiv.org/abs/2202.03259)
    # see get_environment function of TheoryBenchmark on how to specify a train/test environment
    "seed": 0,
    "seed_action_space": False,  # set this one to True for reproducibility when random action is sampled in the action space with gym.action_space.sample()
    "problem": "LeadingOnes",  # possible values: "LeadingOnes"
    "instance_set_path": "lo_rls_50.csv",  # if the instance list file cannot be found in the running directory, it will be looked up in <DACBench>/dacbench/instance_sets/theory/
    "discrete_action": True,  # action space is discrete
    "action_choices": [1, 2, 4, 8, 16],  # portfolio of k values
    "benchmark_info": RLS_INFO,
    "name": "LeadingOnesDAC",
}


class RLSTheoryBenchmark(AbstractBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize a theory benchmark

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        super(RLSTheoryBenchmark, self).__init__()

        self.config = objdict(RLS_THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()

        # initialise action space and environment class
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert "action_choices" in self.config, "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (
                "max_action" not in self.config
            ), "ERROR: min_action and max_action should not be used for discrete action space"
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "RLSTheoryEnvDiscrete"

            # action choices can be a dictionary, where each item represent a list of actions for each instance, in this case, we need to make sure the number of actions is the same for all instances
            if isinstance(self.config["action_choices"], dict):
                cur_size = None  # number of actions of current instance
                for inst_id, ls_acts in self.config["action_choices"].items():
                    assert isinstance(ls_acts, np.ndarray) or isinstance(ls_acts, list)
                    assert (cur_size is None) or (len(ls_acts) == cur_size)
                    cur_size = len(ls_acts)
                n_acts = cur_size

            # the case where we have a single list of actions. For convenience, we will convert action_choices to a dictionary.
            else:
                assert isinstance(self.config["action_choices"], np.ndarray) or isinstance(
                    self.config["action_choices"], list
                )
                n_acts = len(self.config["action_choices"])
                action_choices = {
                    inst_id: self.config["action_choices"]
                    for inst_id in self.config["instance_set"].keys()
                }
                self.config["action_choices"] = action_choices

            action = CSH.UniformIntegerHyperparameter(name="", lower=0, upper=n_acts)

        else:
            assert (
                "action_chocies" not in self.config
            ), "ERROR: action_choices is only used for discrete action space"
            assert ("min_action" in self.config) and (
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "RLSTheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add_hyperparameter(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class == RLSTheoryEnv or self.env_class == RLSTheoryEnvDiscrete

        self.config["observation_space"] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )

    def create_observation_space_from_description(
        self, obs_description, env_class=RLSTheoryEnvDiscrete
    ):
        """
        Create a gym observation space (Box only) based on a string containing observation variable names, e.g. "n, f(x), k, k_{t-1}"
        Return:
            A gym.spaces.Box observation space
        """
        obs_var_names = [s.strip() for s in obs_description.split(",")]
        low = []
        high = []
        for var_name in obs_var_names:
            l, h = env_class.get_obs_domain_from_name(var_name)
            low.append(l)
            high.append(h)
        obs_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32)
        )
        return obs_space

    def get_environment(self, test_env=False):
        """
        Return an environment with current configuration

        Parameters:
            test_env:   whether the enviroment is used for train an agent or for testing.
                        if test_env=False:
                            cutoff time for an episode is set to 0.8*n^2 (n: problem size)
                            if an action is out of range, stop the episode immediately and return a large negative reward (see envs/theory.py for more details)
                        otherwise: benchmark's original cutoff time is used, and out-of-range action will be clipped to nearest valid value and the episode will continue.
        """

        env = self.env_class(self.config, test_env)

        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """
        Read instance set from file
            we look at the current directory first, if the file doesn't exist, we look in <DACBench>/dacbench/instance_sets/theory/
        """
        assert self.config.instance_set_path
        if os.path.isfile(self.config.instance_set_path):
            path = self.config.instance_set_path
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../instance_sets/rls_theory/"
                + self.config.instance_set_path
            )

        self.config["instance_set"] = pd.read_csv(path, index_col=0).to_dict(orient="index")

        assert len(self.config["instance_set"].items()) > 0, "ERROR: empty instance set"
        assert (
            "initObj" in self.config["instance_set"][0].keys()
        ), "ERROR: initial solution (initObj) must be specified in instance set"
        assert (
            "size" in self.config["instance_set"][0].keys()
        ), "ERROR: problem size must be specified in instance set"

        for key, val in self.config["instance_set"].items():
            self.config["instance_set"][key] = objdict(val)


OLLGA_INFO = {
    "identifier": "OLLGATheory",
    "name": "DAC benchmark with OLLGA algorithm and OneMax problem",
    "reward": "Negative number of iterations until solution",
    "state_description": "specified by user",
}

OLLGA_THEORY_DEFAULTS = {
    "observation_description": "n, f(x)",  # examples: n, f(x), delta_f(x), optimal_k, k, k_{t-0..4}, f(x)_{t-1}, f(x)_{t-0..4}
    "reward_range": [-np.inf, np.inf],  # the true reward range is instance dependent
    "reward_choice": "imp_minus_evals",  # possible values: see envs/theory.py for more details
    "cutoff": 1e6,  # if using as a "train" environment, a cutoff of 0.8*n^2 where n is problem size will be used (for more details, please see https://arxiv.org/abs/2202.03259)
    # see get_environment function of TheoryBenchmark on how to specify a train/test environment
    "seed": 0,
    "seed_action_space": False,  # set this one to True for reproducibility when random action is sampled in the action space with gym.action_space.sample()
    "problem": "OneMax",  # possible values: "OneMax"
    "instance_set_path": "om_ollga_50.csv",  # if the instance list file cannot be found in the running directory, it will be looked up in <DACBench>/dacbench/instance_sets/theory/
    "discrete_action": True,  # action space is discrete
    "action_choices": [1, 2, 4, 8, 16],  # portfolio of k values
    "benchmark_info": OLLGA_INFO,
    "name": "OneMaxDAC",
}


class OLLGATheoryBenchmark(AbstractBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize a theory benchmark

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        super(OLLGATheoryBenchmark, self).__init__()

        self.config = objdict(OLLGA_THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()
        # initialise action space and environment class
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert "action_choices" in self.config, "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (
                "max_action" not in self.config
            ), "ERROR: min_action and max_action should not be used for discrete action space"
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "OLLGATheoryEnvDiscrete"

            # action choices can be a dictionary, where each item represent a list of actions for each instance, in this case, we need to make sure the number of actions is the same for all instances
            if isinstance(self.config["action_choices"], dict):
                cur_size = None  # number of actions of current instance
                for inst_id, ls_acts in self.config["action_choices"].items():
                    assert isinstance(ls_acts, np.ndarray) or isinstance(ls_acts, list)
                    assert (cur_size is None) or (len(ls_acts) == cur_size)
                    cur_size = len(ls_acts)
                n_acts = cur_size

            # the case where we have a single list of actions. For convenience, we will convert action_choices to a dictionary.
            else:
                assert isinstance(self.config["action_choices"], np.ndarray) or isinstance(
                    self.config["action_choices"], list
                )
                n_acts = len(self.config["action_choices"])
                action_choices = {
                    inst_id: self.config["action_choices"]
                    for inst_id in self.config["instance_set"].keys()
                }
                self.config["action_choices"] = action_choices

            action = CSH.UniformIntegerHyperparameter(name="", lower=0, upper=n_acts)

        else:
            assert (
                "action_chocies" not in self.config
            ), "ERROR: action_choices is only used for discrete action space"
            assert ("min_action" in self.config) and (
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "OLLGATheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add_hyperparameter(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class == OLLGATheoryEnv or self.env_class == OLLGATheoryEnvDiscrete

        self.config["observation_space"] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )

    def create_observation_space_from_description(
        self, obs_description, env_class=OLLGATheoryEnvDiscrete
    ):
        """
        Create a gym observation space (Box only) based on a string containing observation variable names, e.g. "n, f(x), k, k_{t-1}"
        Return:
            A gym.spaces.Box observation space
        """
        obs_var_names = [s.strip() for s in obs_description.split(",")]
        low = []
        high = []
        for var_name in obs_var_names:
            l, h = env_class.get_obs_domain_from_name(var_name)
            low.append(l)
            high.append(h)
        obs_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32)
        )
        return obs_space

    def get_environment(self, test_env=False):
        """
        Return an environment with current configuration

        Parameters:
            test_env:   whether the enviroment is used for train an agent or for testing.
                        if test_env=False:
                            cutoff time for an episode is set to 0.8*n^2 (n: problem size)
                            if an action is out of range, stop the episode immediately and return a large negative reward (see envs/theory.py for more details)
                        otherwise: benchmark's original cutoff time is used, and out-of-range action will be clipped to nearest valid value and the episode will continue.
        """

        env = self.env_class(self.config, test_env)

        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """
        Read instance set from file
            we look at the current directory first, if the file doesn't exist, we look in <DACBench>/dacbench/instance_sets/theory/
        """
        assert self.config.instance_set_path
        if os.path.isfile(self.config.instance_set_path):
            path = self.config.instance_set_path
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../instance_sets/ollga_theory/"
                + self.config.instance_set_path
            )

        self.config["instance_set"] = pd.read_csv(path, index_col=0).to_dict(orient="index")

        assert len(self.config["instance_set"].items()) > 0, "ERROR: empty instance set"
        assert (
            "initObj" in self.config["instance_set"][0].keys()
        ), "ERROR: initial solution (initObj) must be specified in instance set"
        assert (
            "size" in self.config["instance_set"][0].keys()
        ), "ERROR: problem size must be specified in instance set"

        for key, val in self.config["instance_set"].items():
            self.config["instance_set"][key] = objdict(val)


class OLLGAFactTheoryBenchmark(OLLGATheoryBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize a theory benchmark

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        super(OLLGAFactTheoryBenchmark, self).__init__()
        self.config = objdict(OLLGA_THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()

        # initialise action space and environment class
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert "action_choices" in self.config, "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (
                "max_action" not in self.config
            ), "ERROR: min_action and max_action should not be used for discrete action space"
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "OLLGAFactTheoryEnvDiscrete"

            # action choices can be a dictionary, where each item represent a list of actions for each instance, in this case, we need to make sure the number of actions is the same for all instances
            if isinstance(self.config["action_choices"], dict):
                cur_size = None  # number of actions of current instance
                for inst_id, ls_acts in self.config["action_choices"].items():
                    assert isinstance(ls_acts, np.ndarray) or isinstance(ls_acts, list)
                    assert (cur_size is None) or (len(ls_acts) == cur_size)
                    cur_size = len(ls_acts)
                n_acts = cur_size

            # the case where we have a single list of actions. For convenience, we will convert action_choices to a dictionary.
            else:
                assert isinstance(self.config["action_choices"], np.ndarray) or isinstance(
                    self.config["action_choices"], list
                )
                n_acts = len(self.config["action_choices"][0])
                action_choices = {
                    inst_id: self.config["action_choices"]
                    for inst_id in self.config["instance_set"].keys()
                }
                self.config["action_choices"] = action_choices

            action = CSH.UniformIntegerHyperparameter(name="", lower=0, upper=n_acts)

        else:
            assert (
                "action_chocies" not in self.config
            ), "ERROR: action_choices is only used for discrete action space"
            assert ("min_action" in self.config) and (
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "OLLGATheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add_hyperparameter(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class == OLLGATheoryEnv or self.env_class == OLLGAFactTheoryEnvDiscrete

        self.config["observation_space"] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )


class OLLGAFactL1TheoryBenchmark(OLLGATheoryBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize a theory benchmark

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        super(OLLGAFactL1TheoryBenchmark, self).__init__()
        self.config = objdict(OLLGA_THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()

        # initialise action space and environment class
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert "action_choices" in self.config, "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (
                "max_action" not in self.config
            ), "ERROR: min_action and max_action should not be used for discrete action space"
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "OLLGAFactL1TheoryEnvDiscrete"

            # action choices can be a dictionary, where each item represent a list of actions for each instance, in this case, we need to make sure the number of actions is the same for all instances
            if isinstance(self.config["action_choices"], dict):
                cur_size = None  # number of actions of current instance
                for inst_id, ls_acts in self.config["action_choices"].items():
                    assert isinstance(ls_acts, np.ndarray) or isinstance(ls_acts, list)
                    assert (cur_size is None) or (len(ls_acts) == cur_size)
                    cur_size = len(ls_acts)
                n_acts = cur_size

            # the case where we have a single list of actions. For convenience, we will convert action_choices to a dictionary.
            else:
                assert isinstance(self.config["action_choices"], np.ndarray) or isinstance(
                    self.config["action_choices"], list
                )
                n_acts = len(self.config["action_choices"][0])
                action_choices = {
                    inst_id: self.config["action_choices"]
                    for inst_id in self.config["instance_set"].keys()
                }
                self.config["action_choices"] = action_choices

            action = CSH.UniformIntegerHyperparameter(name="", lower=0, upper=n_acts)

        else:
            assert (
                "action_chocies" not in self.config
            ), "ERROR: action_choices is only used for discrete action space"
            assert ("min_action" in self.config) and (
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "OLLGATheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add_hyperparameter(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class == OLLGATheoryEnv or self.env_class == OLLGAFactL1TheoryEnvDiscrete

        self.config["observation_space"] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )


class OLLGATheoryPPOBenchmark(OLLGATheoryBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize a theory benchmark

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        super(OLLGATheoryPPOBenchmark, self).__init__()
        self.config = objdict(OLLGA_THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()

        # initialise action space and environment class
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert "action_choices" in self.config, "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (
                "max_action" not in self.config
            ), "ERROR: min_action and max_action should not be used for discrete action space"
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "OLLGATheoryPPOEnvDiscrete"

            # action choices can be a dictionary, where each item represent a list of actions for each instance, in this case, we need to make sure the number of actions is the same for all instances
            if isinstance(self.config["action_choices"], dict):
                cur_size = None  # number of actions of current instance
                for inst_id, ls_acts in self.config["action_choices"].items():
                    assert isinstance(ls_acts, np.ndarray) or isinstance(ls_acts, list)
                    assert (cur_size is None) or (len(ls_acts) == cur_size)
                    cur_size = len(ls_acts)
                n_acts = cur_size

            # the case where we have a single list of actions. For convenience, we will convert action_choices to a dictionary.
            else:
                assert isinstance(self.config["action_choices"], np.ndarray) or isinstance(
                    self.config["action_choices"], list
                )
                n_acts = len(self.config["action_choices"][0])
                action_choices = {
                    inst_id: self.config["action_choices"]
                    for inst_id in self.config["instance_set"].keys()
                }
                self.config["action_choices"] = action_choices

            action = CSH.UniformIntegerHyperparameter(name="", lower=0, upper=n_acts)

        else:
            assert (
                "action_chocies" not in self.config
            ), "ERROR: action_choices is only used for discrete action space"
            assert ("min_action" in self.config) and (
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "OLLGATheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        # cfg_space.add_hyperparameter(action)
        # self.config["config_space"] = cfg_space
        self.config["action_space_class"] = "MultiDiscrete"
        self.config["action_space_args"] = [
            [len(item) for item in self.config["action_choices"][0]]
        ]

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class == OLLGATheoryEnv or self.env_class == OLLGATheoryPPOEnvDiscrete

        self.config["observation_space"] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )
