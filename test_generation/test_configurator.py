import glob
import logging
import os
from typing import List, Tuple, Union, cast

import gym
import numpy as np
import torch as th
from test_generation.utils.file_utils import (
    get_coef_number,
    get_one_hot_for_mutant_coef,
)
from log import Log
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from test_utils import get_trained_model
from training.training_type import TrainingType

from test_generation.env_configuration import EnvConfiguration
from test_generation.env_wrapper import EnvWrapper
from test_generation.preprocessor import preprocess_data
from test_generation.test_generation_config import SAMPLING_SIZE
from test_generation.test_generator import TestGenerator


class TestConfigurator:
    def __init__(
        self,
        env_name: str,
        render: bool = False,
        seed: int = -1,
        num_envs: int = 1,
        folder: str = "logs",
        algo: str = "ppo",
        exp_id: int = 0,
        device: Union[th.device, str] = "auto",
        testing_policy_for_training_name: str = "mlp",
        testing_strategy_name: str = "nn",
        model_checkpoint: int = -1,
        exp_name: str = None,
        training_progress_filter: int = None,
        layers: int = 4,
        num_episodes: int = 1000,
        num_runs_each_env_config: int = 10,
        budget: int = -1,
        sampling_size: int = SAMPLING_SIZE,
        register_environment: bool = False,
        training_type: TrainingType = 0,
        test_multiple_agents: bool = False,
        parallelize: bool = False,
        **env_kwargs,
    ):
        self.logger = Log("TestConfigurator", parallelize=parallelize)

        self.env_name = env_name
        self.num_envs = num_envs
        assert self.num_envs == 1, "Num envs must be = 1. Found: {}".format(
            self.num_envs
        )
        self.folder = folder
        self.algo = algo
        self.exp_id = exp_id
        self.seed = seed
        self.testing_policy_for_training_name = testing_policy_for_training_name
        self.layers = layers
        self.model_checkpoint = model_checkpoint
        self.training_progress_filter = training_progress_filter
        self.num_episodes = num_episodes
        self.num_runs_each_env_config = num_runs_each_env_config
        self.budget = budget
        self.sampling_size = sampling_size
        self.env: VecEnv = None
        self.model: BaseAlgorithm = None
        self.env_kwargs = env_kwargs
        self.register_environment = register_environment
        self.time_wrapper = False
        self.exp_name = exp_name
        self.render = render
        self.device = device
        if "timeout_steps" in self.env_kwargs.keys():
            self.time_wrapper = True

        self.testing_strategy_for_logs = testing_strategy_name
        if testing_strategy_name == "random":
            self.testing_strategy_for_logs = "random"
        elif testing_strategy_name == "prioritized_replay":
            assert (
                self.training_progress_filter is not None
            ), "Training progress filter must not be None"
            self.testing_strategy_for_logs = "{}-{}".format(
                testing_strategy_name, self.training_progress_filter
            )
        elif testing_strategy_name == "nn":
            if self.training_progress_filter is not None:
                self.testing_strategy_for_logs = "{}-{}-{}-{}".format(
                    self.testing_policy_for_training_name,
                    testing_strategy_name,
                    self.training_progress_filter,
                    self.layers,
                )
            else:
                self.testing_strategy_for_logs = "{}-{}-{}".format(
                    self.testing_policy_for_training_name,
                    testing_strategy_name,
                    self.layers,
                )

        self.testing_strategy_name = testing_strategy_name

        if (
            self.budget != -1
            and testing_strategy_name != "random"
            and testing_strategy_name != "prioritized_replay"
        ):
            self.testing_strategy_for_logs += "-budget-{}".format(self.budget)

        if self.exp_name is not None:
            self.testing_strategy_for_logs += "-{}".format(self.exp_name)

        self.run_folders = []
        self.folder_name_with_agents = None

        self.training_type = training_type
        self.test_multiple_agents = test_multiple_agents
        if self.test_multiple_agents:
            # FIXME: old feature to remove
            self.folder_name_with_agents = f"{folder}{os.path.sep}{algo}{os.path.sep}{env_name}_{self.training_type.name}"
            assert os.path.exists(
                self.folder_name_with_agents
            ), f"{self.folder_name_with_agents} does not exist"
            runs_folders = glob.glob(
                os.path.join(self.folder_name_with_agents, "run_*")
            )
            self.runs_folders = sorted(
                runs_folders,
                key=lambda run_folder: int(run_folder[run_folder.rindex("_") + 1 :]),
            )
            assert (
                len(runs_folders) > 0
            ), f"There are no runs in {self.folder_name_with_agents}"
        elif training_type != TrainingType.mutant:
            self.initialize()

        self.test_generator: TestGenerator = None

    def initialize(
        self,
        folder_with_agent: str = None,
        reset_random_gen: bool = False,
        configurations: List[EnvConfiguration] = None,
        parallelize: bool = False,
        mutant_name: str = None,
        mutant_configuration: str = None,
        num_run: int = -1,
        encode_run_and_conf: bool = False,
    ) -> None:
        self.env, self.model, log_path = get_trained_model(
            exp_id=self.exp_id,
            env_name=self.env_name,
            folder=self.folder,
            algo=self.algo,
            seed=self.seed,
            device=self.device,
            test_generation=True,
            register_environment=self.register_environment,
            time_wrapper=self.time_wrapper,
            folder_with_agent=folder_with_agent,
            reset_random_gen=reset_random_gen,
            parallelize=parallelize,
            **self.env_kwargs,
        )

        coef_one_hot = None
        if encode_run_and_conf:
            coef_num = get_coef_number(
                filepath=folder_with_agent, mutant_name=mutant_name
            )
            if coef_num is not None:
                coef_one_hot = get_one_hot_for_mutant_coef(
                    coef_number=coef_num,
                    mutant_name=mutant_name,
                    log_path=os.path.join(self.folder, self.algo),
                )
            else:
                coef_one_hot = coef_num

        # trick to extract the test_generator object from the environment

        env_unwrapped = self.env.unwrapped.envs[0]
        while not isinstance(env_unwrapped, EnvWrapper):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped = cast(EnvWrapper, env_unwrapped)
        self.test_generator = cast(TestGenerator, env_unwrapped.test_generator)

        logging.basicConfig(
            filename=os.path.join(
                log_path, f"testing-{self.testing_strategy_for_logs}.txt"
            ),
            filemode="w",
            level=logging.DEBUG,
        )

        preprocessed_dataset = None
        if self.testing_strategy_name != "random":
            # FIXME: we are not loading all the data the predictor was trained on, but only the ones of the single
            #   agent we are considering
            preprocessed_dataset = preprocess_data(
                env_name=self.env_name,
                log_paths=[log_path],
                training_progress_filter=self.training_progress_filter,
                policy_name=self.testing_policy_for_training_name,
                train_from_multiple_runs=self.test_multiple_agents,
                testing_mode=True,
            )

        if "original" in folder_with_agent:
            self.logger.warn("Setting num_run to -1 for original agent")
            num_run = -1

        print(folder_with_agent)

        self.test_generator.update_state_variables_to_enable_test_generation(
            num_episodes=self.num_episodes,
            training_progress_filter=self.training_progress_filter,
            layers=self.layers,
            num_runs_each_env_config=self.num_runs_each_env_config,
            testing_policy_for_training_name=self.testing_policy_for_training_name,
            testing_strategy_name=self.testing_strategy_name,
            testing_strategy_name_logs=self.testing_strategy_for_logs,
            budget=self.budget,
            preprocessed_dataset=preprocessed_dataset,
            sampling_size=self.sampling_size,
            configurations=configurations,
            mutant_name=mutant_name,
            mutant_configuration=mutant_configuration,
            num_run=num_run,
            encode_run_and_conf=encode_run_and_conf,
            coef_one_hot=coef_one_hot,
        )

        self.deterministic = True

    def test_single_episode(
        self, episode_num: int, num_trials: int = -1
    ) -> Tuple[bool, EnvConfiguration, List[float]]:
        obs = self.env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        fitness_values = []

        if episode_num != -1:
            self.test_generator.num_episodes = episode_num
        if num_trials != -1:
            self.test_generator.num_trials = num_trials

        while not done:
            action, state = self.model.predict(
                obs, state=state, deterministic=self.deterministic
            )
            # Clip Action to avoid out of bound errors
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )

            obs, reward, done, _info = self.env.step(action)

            if _info[0].get("fitness", None) is not None:
                fitness_values.append(_info[0]["fitness"])

            if self.render:
                self.env.render()

            episode_reward += reward[0]
            episode_length += 1
            if done:
                self.test_generator.num_episodes += 1
                self.logger.debug("Episode #{}".format(episode_num + 1))
                self.logger.debug("Episode Reward: {:.2f}".format(episode_reward))
                self.logger.debug("Episode Length: {}".format(episode_length))
                if episode_length < 5:
                    self.logger.warn("Very short episode")
                is_success = _info[0].get("is_success", None)
                if is_success is not None:
                    self.logger.debug("Failure: {}".format(not is_success))
                    if is_success == 0:
                        return (
                            True,
                            self.test_generator.get_current_env_config(),
                            fitness_values,
                        )
                    return (
                        False,
                        self.test_generator.get_current_env_config(),
                        fitness_values,
                    )

    def close_env(self):
        assert self.num_envs == 1, "Num envs must be = 1. Found: {}".format(
            self.num_envs
        )
        self.env.close()

    def get_num_evaluation_predictions(self) -> int:
        if self.test_generator.trained_policy is not None:
            return self.test_generator.trained_policy.get_num_evaluation_predictions()
        return 0
