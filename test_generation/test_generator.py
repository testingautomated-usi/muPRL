import importlib
import json
import os
import random
import time
from typing import List, Tuple, Union

import numpy as np
from config import ENV_NAMES, LOG_PATH_CONFIGS_DIRECTORY
from log import Log
from PIL import Image
from training.training_type import TrainingType

from test_generation.dataset import Data, Dataset
from test_generation.env_configuration import EnvConfiguration
from test_generation.failure_predictor import FailurePredictor
from test_generation.policies.testing_policy import TestingPolicy
from test_generation.test_generation_config import SAMPLING_SIZE
from test_generation.training_logs import TrainingLogs


class TestGenerator:
    def __init__(
        self,
        env_name: str,
        log_path: str,
        storing_logs: bool,
        seed: int = None,
        evaluate_agent_during_training: bool = False,
        parallelize: bool = False,
    ):
        self.logger = Log("test_generator", parallelize=parallelize)
        self.env_name = env_name
        assert self.env_name in ENV_NAMES, "Env not supported: {}".format(env_name)
        self.log_path = log_path
        self.storing_logs = storing_logs
        self.seed = seed
        # to take into account multiple environments
        self.file_num_first = 0
        self.is_training = self.storing_logs

        self.training_progress_filter: int = None
        self.testing_policy_for_training_name: str = None
        self.testing_strategy_name: str = None
        self.testing_strategy_name_logs: str = None
        self.layers: int = 4
        self.sampling_size: int = None
        self.num_runs_each_env_config: int = None

        self.evaluate_agent_during_training = evaluate_agent_during_training

        # testing variables
        self.preprocessed_dataset = None
        self.failed_data_items: List[Data] = []
        self.sorted_failed_data_items: List[Data] = []
        self.idx_data: int = 0
        self.indices_data_selected: List[int] = []
        self.weights_data: List[float] = []
        self.trained_policy: TestingPolicy = None
        self.current_env_config: EnvConfiguration = None
        self.failure_probability_env_configurations: List[EnvConfiguration] = []
        self.num_trials: int = -1
        self.num_episodes: int = -1
        self.budget = -1
        self.all_configurations = []
        self.predictions = []
        self.mutant_name = None
        self.mutant_configuration = None
        self.trained_policy_save_path = None
        self.run_num = -1
        self.encode_run_and_conf = False

        # self.archive: List[EnvConfiguration] = []

    @staticmethod
    def load_generator(env_name: str) -> EnvConfiguration:
        env_name = env_name.split("-")[0]

        env_filename = f"envs.{env_name.lower()}.{env_name.lower()}_env_configuration"

        envlib = importlib.import_module(env_filename)

        target_env_name = env_name.replace("_", "") + "EnvConfiguration"

        for name, cls in envlib.__dict__.items():
            if name.lower() == target_env_name.lower() and issubclass(
                cls, EnvConfiguration
            ):
                return cls

        raise RuntimeError(
            "In %s.py, there should be a subclass of EnvConfiguration with class name that matches %s in lowercase."
            % (env_filename, target_env_name)
        )

    def update_state_variables_to_enable_test_generation(
        self,
        num_episodes: int,
        training_progress_filter: int,
        testing_policy_for_training_name: str,
        testing_strategy_name: str,
        testing_strategy_name_logs: str,
        layers: int,
        sampling_size: int,
        num_runs_each_env_config: int,
        budget: int,
        preprocessed_dataset: Dataset,
        configurations: List[EnvConfiguration],
        mutant_name: str,
        mutant_configuration: str,
        num_run: int,
        encode_run_and_conf: bool,
        coef_one_hot: List[int],
    ) -> None:
        self.num_episodes = num_episodes
        self.training_progress_filter = training_progress_filter
        self.testing_policy_for_training_name = testing_policy_for_training_name
        self.testing_strategy_name = testing_strategy_name
        self.testing_strategy_name_logs = testing_strategy_name_logs
        self.layers = layers
        self.sampling_size = sampling_size
        self.num_runs_each_env_config = num_runs_each_env_config
        self.budget = budget
        self.preprocessed_dataset = preprocessed_dataset

        self.failed_data_items: List[Data] = []
        self.sorted_failed_data_items: List[Data] = []
        self.idx_data: int = 0
        self.indices_data_selected: List[int] = []
        self.weights_data: List[float] = []
        self.current_env_config: EnvConfiguration = None
        self.failure_probability_env_configurations: List[EnvConfiguration] = []

        self.mutant_name = mutant_name
        self.mutant_configuration = mutant_configuration

        self.run_num = num_run
        self.coef_one_hot = coef_one_hot
        self.encode_run_and_conf = encode_run_and_conf

        if configurations is not None and len(configurations) > 0:
            self.all_configurations.extend(configurations)

    def sample_test_env_configuration(
        self,
        testing_policy_for_training_name: str,
        testing_strategy_name: str,
        sampling_size: int = SAMPLING_SIZE,
    ) -> Union[EnvConfiguration, Tuple[EnvConfiguration, float, float], None]:
        if testing_strategy_name == "random":
            return self.generate_random_env_configuration()

        start_time = time.perf_counter()
        if testing_strategy_name == "prioritized_replay":
            (
                env_config,
                max_prediction,
            ) = self.sample_test_env_configuration_prioritized()
        elif testing_strategy_name == "nn":
            env_config, max_prediction = self.sample_test_env_configuration_nn(
                testing_policy_for_training_name=testing_policy_for_training_name,
                sampling_size=sampling_size,
            )
        else:
            raise NotImplementedError(
                "Unknown test strategy name: {}".format(testing_strategy_name)
            )

        return env_config, time.perf_counter() - start_time, max_prediction

    def sample_test_env_configuration_prioritized(
        self,
    ) -> Tuple[EnvConfiguration, float]:
        assert (
            self.preprocessed_dataset is not None
        ), "Preprocessed dataset cannot be None"

        if len(self.failed_data_items) == 0:
            self.failed_data_items = [
                data_item
                for data_item in self.preprocessed_dataset.get()
                if data_item.label == 1
            ]
            self.weights_data = [
                data_item.training_progress for data_item in self.failed_data_items
            ]

        idx_data = random.choices(
            population=np.arange(0, len(self.failed_data_items)),
            weights=self.weights_data,
        )[0]

        if len(self.indices_data_selected) == len(self.failed_data_items):
            self.logger.warn("Fallback to random generation")
            self.current_env_config = self.generate_random_env_configuration()
            self.logger.info(
                "Env configuration: {}".format(self.current_env_config.get_str())
            )
            return self.current_env_config, 0.0

        while idx_data in self.indices_data_selected:
            idx_data = random.choices(
                population=np.arange(0, len(self.failed_data_items)),
                weights=self.weights_data,
            )[0]

        self.logger.debug(
            "Index data: {}. {}/{}".format(
                idx_data, len(self.indices_data_selected), len(self.failed_data_items)
            )
        )
        self.indices_data_selected.append(idx_data)

        self.current_env_config = self.failed_data_items[
            idx_data
        ].training_logs.get_config()
        self.logger.info(
            "Env configuration: {}; Training progress: {}".format(
                self.current_env_config.get_str(),
                self.failed_data_items[idx_data].training_logs.get_training_progress(),
            )
        )
        return self.current_env_config, 0.0

    def sample_test_env_configuration_nn(
        self,
        testing_policy_for_training_name: str,
        sampling_size: int,
    ) -> Tuple[EnvConfiguration, float]:
        assert (
            self.preprocessed_dataset is not None
        ), "Preprocessed dataset cannot be None"
        if self.budget != -1:
            assert sampling_size > 0, "Sampling size must be > 0"

        num_generated = 0
        predictions = []
        env_configurations: List[EnvConfiguration] = []
        start_time = time.perf_counter()
        current_num_predictions = 0
        condition = (
            num_generated != sampling_size
            if self.budget == -1
            else time.perf_counter() - start_time < self.budget
        )
        while condition:
            env_configuration = self.generate_random_env_configuration()
            env_configurations.append(env_configuration)

            # if self.encode_run_and_conf:
            #     assert self.run_num != -1, "Run number has not been set"

            env_configuration_transformed = (
                self.preprocessed_dataset.transform_env_configuration(
                    env_configuration=env_configuration,
                    policy_name=testing_policy_for_training_name,
                    encode_run_and_conf=self.encode_run_and_conf,
                    run_number=self.run_num,
                    coef_one_hot=self.coef_one_hot,
                )
            )

            prediction = self.trained_policy.get_failure_class_prediction(
                env_config_transformed=env_configuration_transformed,
                dataset=self.preprocessed_dataset,
            )
            predictions.append(prediction)

            current_num_predictions += 1

            num_generated += 1
            condition = (
                num_generated != sampling_size
                if self.budget == -1
                else time.perf_counter() - start_time < self.budget
            )

        max_prediction_idx = np.argmax(predictions)
        max_env_config = env_configurations[max_prediction_idx]
        max_prediction = predictions[max_prediction_idx]

        self.current_env_config = max_env_config
        self.logger.info(
            "Env configuration: {}; max prediction: {}, # samples: {}".format(
                self.current_env_config.get_str(), max_prediction, num_generated
            )
        )
        self.predictions.append(max_prediction)

        return self.current_env_config, max_prediction

    def generate_random_env_configuration(self) -> EnvConfiguration:
        gnt_instance = self.load_generator(env_name=self.env_name)()
        env_config = gnt_instance.generate_configuration(
            evaluate_agent_during_training=self.evaluate_agent_during_training
        )

        if self.testing_strategy_name == "random":
            self.current_env_config = env_config
            # self.logger.info("Env configuration: {}".format(self.current_env_config.get_str()))

        return env_config

    def generate_env_configuration(self) -> EnvConfiguration:
        if self.is_training:
            return self.generate_random_env_configuration()

        if not self.is_training:
            if (
                self.testing_strategy_name != "random"
                and self.testing_strategy_name != "constant"
            ):
                assert (
                    self.preprocessed_dataset is not None
                ), "Preprocessed dataset cannot be None"

                if (
                    self.testing_strategy_name != "prioritized_replay"
                    and self.trained_policy is None
                ):
                    save_path = FailurePredictor.get_policy_save_path(
                        log_path=self.log_path,
                        training_progress_filter=self.training_progress_filter,
                        testing_policy_for_training_name=self.testing_policy_for_training_name,
                        layers=self.layers,
                        mutant_name=self.mutant_name,
                        mutant_configuration=self.mutant_configuration,
                    )

                    # look for the predictor in the TrainingType.original folder
                    if not os.path.exists(save_path):
                        # look for the failure predictor in the _original directory (I assume it exists, else it fails)
                        failure_predictor_name = save_path[save_path.rindex("/") + 1 :]
                        assert (
                            "run_" in save_path
                        ), f"The word 'run_' should be in save_path: {save_path}"
                        # the -1 is to remove the path separator
                        failure_predictor_directory = (
                            save_path[: save_path.rindex(self.env_name)]
                            + f"{self.env_name}_{TrainingType.original.name}"
                        )
                        assert os.path.exists(
                            failure_predictor_directory
                        ), f"Failure predictor directory {failure_predictor_directory} does not exist"
                        save_path = os.path.join(
                            failure_predictor_directory, failure_predictor_name
                        )
                        if (
                            self.mutant_name is not None
                            and self.mutant_configuration is not None
                        ):
                            if not os.path.exists(save_path):
                                save_path = save_path.replace(
                                    f"-{self.mutant_configuration}", ""
                                )
                                self.logger.warn(
                                    f"Falling back to predictor trained on training configurations {self.mutant_configuration} of {self.mutant_name}, i.e., {save_path}"
                                )
                                assert os.path.exists(
                                    save_path
                                ), f"{save_path} does not exist"
                        elif (
                            self.mutant_name is not None
                            and self.training_progress_filter is not None
                        ):
                            if not os.path.exists(save_path) and save_path.endswith(
                                f"{self.mutant_name}.pkl"
                            ):
                                save_path = save_path.replace(
                                    f"-{self.mutant_name}", ""
                                )
                                self.logger.warn(
                                    f"Falling back to predictor trained on all training configurations of {self.mutant_name}, i.e., {save_path}"
                                )
                                assert os.path.exists(
                                    save_path
                                ), f"{save_path} does not exist"
                        elif self.training_progress_filter is None:
                            if not os.path.exists(save_path) and save_path.endswith(
                                f"{self.mutant_name}.pkl"
                            ):
                                save_path = save_path.replace(
                                    f"-{self.mutant_name}", ""
                                )
                                self.logger.warn(
                                    f"Falling back to predictor trained on all testing configurations of the original agent, i.e., {save_path}"
                                )
                        else:
                            assert os.path.exists(
                                save_path
                            ), f"{save_path} does not exist"

                    mutant_name = self.mutant_name
                    # trick to support failure predictor only trained on original agent
                    if mutant_name is not None and mutant_name not in save_path:
                        mutant_name = None
                        self.coef_one_hot = None

                    input_size = self.preprocessed_dataset.get_num_features(
                        encode_run_and_conf=self.encode_run_and_conf,
                        mutant_name=mutant_name,
                        log_path=self.log_path.split(self.env_name)[0],
                    )

                    self.trained_policy = FailurePredictor.load_testing_policy(
                        test_policy_for_training_name=self.testing_policy_for_training_name,
                        input_size=input_size,
                        load_path=save_path,
                        layers=self.layers,
                    )
                    self.trained_policy_save_path = save_path

            if len(self.failure_probability_env_configurations) == 0:
                assert (
                    self.num_runs_each_env_config >= 1
                ), "Num runs for each env configuration must be >= 1. Found: {}".format(
                    self.num_runs_each_env_config
                )
                self.logger.info(
                    "Generating {} env configurations".format(self.num_episodes)
                )
                times_elapsed = []

                if len(self.all_configurations):
                    assert len(self.all_configurations) == self.num_episodes, (
                        f"Num existing configurations {len(self.all_configurations)} "
                        f"!= num episodes {self.num_episodes}"
                    )
                    self.logger.info(
                        f"Reusing existing configurations generated in a previous run. "
                        f"Num configurations: {len(self.all_configurations)}"
                    )

                for i in range(self.num_episodes):
                    start_time = time.perf_counter()

                    if len(self.all_configurations) == self.num_episodes:
                        env_config = self.all_configurations[i]
                        self.current_env_config = env_config
                        self.logger.info(
                            "Existing configuration {}: {}".format(
                                i, self.current_env_config.get_str()
                            )
                        )
                    else:
                        res = self.sample_test_env_configuration(
                            testing_policy_for_training_name=self.testing_policy_for_training_name,
                            testing_strategy_name=self.testing_strategy_name,
                            sampling_size=self.sampling_size,
                        )

                        if isinstance(res, tuple):
                            env_config = res[0]
                        else:
                            env_config = res

                        self.all_configurations.append(env_config)

                    self.logger.info(
                        "Times elapsed: {} s".format(
                            round(time.perf_counter() - start_time, 2)
                        )
                    )
                    times_elapsed.append(round(time.perf_counter() - start_time, 2))
                    for j in range(self.num_runs_each_env_config):
                        self.failure_probability_env_configurations.append(env_config)

                self.logger.info(
                    "Time elapsed (s): {}, Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                        times_elapsed,
                        np.mean(times_elapsed),
                        np.std(times_elapsed),
                        np.min(times_elapsed),
                        np.max(times_elapsed),
                    )
                )
                # return the first configuration; in any case it will not be used, since the reset will be called
                # immediately after and a new configuration (below) will be taken
                self.current_env_config = self.failure_probability_env_configurations[
                    self.idx_data
                ]
                self.idx_data += 1
                self.logger.info(
                    "{}/{}".format(
                        self.idx_data, len(self.failure_probability_env_configurations)
                    )
                )
                return self.current_env_config

            self.current_env_config = self.failure_probability_env_configurations[
                self.idx_data
            ]
            self.idx_data += 1
            self.logger.info(
                "{}/{}".format(
                    self.idx_data, len(self.failure_probability_env_configurations)
                )
            )
            return self.current_env_config

        raise RuntimeError("Cannot generate env configuration")

    def store_testing_logs(self, training_logs: TrainingLogs) -> None:
        if not self.is_training:
            assert (
                self.testing_strategy_name_logs is not None
            ), "Testing strategy name for logs is not assigned"
            assert self.num_episodes != -1, "Num episodes not assigned"
            if self.num_trials != -1:
                filepath = os.path.join(
                    self.log_path,
                    self.testing_strategy_name_logs,
                    "trial-{}".format(self.num_trials),
                )
                os.makedirs(filepath, exist_ok=True)
            else:
                filepath = os.path.join(self.log_path, self.testing_strategy_name_logs)
                os.makedirs(filepath, exist_ok=True)

            env_image_array = training_logs.get_testing_image()
            if env_image_array is not None:
                env_image = Image.fromarray(env_image_array.astype(np.uint8))
                filepath_image = os.path.join(
                    filepath,
                    "{}-log-image-{}.png".format(
                        self.num_episodes, int(training_logs.get_label())
                    ),
                )
                env_image.save(fp=filepath_image)

            filepath_json = os.path.join(
                filepath,
                "{}-log-{}.json".format(
                    self.num_episodes, int(training_logs.get_label())
                ),
            )
            json_string = json.dumps(training_logs.to_dict(), indent=4)
            with open(filepath_json, "w+", encoding="utf-8") as f:
                f.write(json_string)

    def store_training_logs(self, training_logs: TrainingLogs) -> None:
        if self.storing_logs:
            file_num_last = 0
            if not os.path.exists(
                os.path.join(self.log_path, LOG_PATH_CONFIGS_DIRECTORY)
            ):
                os.makedirs(os.path.join(self.log_path, LOG_PATH_CONFIGS_DIRECTORY))
            log_path = os.path.join(
                self.log_path,
                LOG_PATH_CONFIGS_DIRECTORY,
                "policy_log_{}_{}.json".format(self.file_num_first, file_num_last),
            )
            while os.path.exists(log_path):
                file_num_last += 1
                log_path = os.path.join(
                    self.log_path,
                    LOG_PATH_CONFIGS_DIRECTORY,
                    "policy_log_{}_{}.json".format(self.file_num_first, file_num_last),
                )

            json_string = json.dumps(training_logs.to_dict(), indent=4)
            with open(log_path, "w+", encoding="utf-8") as f:
                f.write(json_string)

    def store_all_training_logs(self, all_training_logs: List[TrainingLogs]) -> None:
        if self.storing_logs:
            if not os.path.exists(
                os.path.join(self.log_path, LOG_PATH_CONFIGS_DIRECTORY)
            ):
                os.makedirs(os.path.join(self.log_path, LOG_PATH_CONFIGS_DIRECTORY))
            fp = os.path.join(
                self.log_path, LOG_PATH_CONFIGS_DIRECTORY, "training_logs.npz"
            )
            np.savez_compressed(fp, training_logs=all_training_logs)

    def store_environment_configurations(
        self, env_configs: List[EnvConfiguration], evaluation: bool = False
    ) -> None:
        if self.storing_logs:
            fp = os.path.join(
                self.log_path,
                "eval_env_configs.npz" if evaluation else "env_configs.npz",
            )
            np.savez_compressed(fp, env_configs=env_configs)

    def get_current_env_config(self) -> EnvConfiguration:
        return self.current_env_config
