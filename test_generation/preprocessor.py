import glob
import importlib
import os
from functools import reduce
from typing import Dict, List

import numpy as np
from config import ENV_NAMES, LOG_PATH_CONFIGS_DIRECTORY
from log import Log
from PIL import Image
from training.training_type import TrainingType

from test_generation.dataset import Dataset, TestingData, TrainingData
from test_generation.logs_factory import make_log
from test_generation.testing_logs import TestingLogs
from test_generation.training_logs import TrainingLogs
from test_generation.utils.file_utils import parse_experiment_file, read_logs

logger = Log("preprocessor")


def parse_training_logs(env_name: str, json_data: Dict) -> TrainingLogs:
    if env_name in ENV_NAMES:
        env_config = json_data["env_config"]
        del json_data["env_config"]
        if json_data["agent_state"] is None:
            json_data["agent_state"] = dict()
        return make_log(
            env_name=env_name,
            log_type="training",
            env_config=env_config,
            data=json_data,
        )

    raise NotImplementedError("Unknown env name: {}".format(env_name))


def parse_testing_logs(env_name: str, json_data) -> TestingLogs:
    if env_name in ENV_NAMES:
        env_config = json_data["env_config"]
        dynamic_info = json_data["dynamic_info"]
        return make_log(
            env_name=env_name,
            log_type="testing",
            env_config=env_config,
            data=json_data,
            dynamic_info=dynamic_info,
        )
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))


def sort_training_progress(env_name: str, log_path: str, filename: str) -> float:
    if filename.endswith(".json"):
        json_data = read_logs(log_path=log_path, filename=filename)
        training_logs = parse_training_logs(env_name=env_name, json_data=json_data)
        return training_logs.get_training_progress()
    return 0.0


def load_dataset(env_name: str) -> Dataset:
    env_name = env_name.split("-")[0]

    env_filename = f"envs.{env_name.lower()}.{env_name.lower()}_dataset"

    envlib = importlib.import_module(env_filename)

    target_env_name = env_name.replace("_", "") + "Dataset"

    for name, cls in envlib.__dict__.items():
        if name.lower() == target_env_name.lower() and issubclass(cls, Dataset):
            return cls

    raise RuntimeError(
        "In %s.py, there should be a subclass of Dataset with class name that matches %s in lowercase."
        % (env_filename, target_env_name)
    )


def preprocess_data(
    env_name: str,
    log_paths: List[str],
    training_progress_filter: int,
    policy_name: str,
    save_data: bool = False,
    train_from_multiple_runs: bool = False,
    testing_mode: bool = False,
    threshold_failure: float = 0.5,
    percentage_failure_discard: float = 0.9,
) -> Dataset:
    assert env_name in ENV_NAMES, "Unknown env name: {}".format(env_name)

    dataset = load_dataset(env_name=env_name)(policy=policy_name)
    all_episodes_unfiltered = 0
    num_failures = 0

    if training_progress_filter is None:
        logger.warn("Training progress filter is None. Preprocessing testing data.")

        for log_path in log_paths:
            replay_files = []

            if train_from_multiple_runs:
                runs_folders = glob.glob(os.path.join(log_path, "run_*"))
                assert len(runs_folders) > 0, f"No runs folders in {log_path}"
                for run_folder in runs_folders:
                    replay_file_matches = glob.glob(
                        os.path.join(run_folder, "*replay.txt")
                    )
                    assert (
                        len(replay_file_matches) == 1
                    ), f"No replay file in {run_folder} or more than one"
                    replay_files.append(replay_file_matches[0])
            else:
                replay_file_matches = glob.glob(os.path.join(log_path, "*replay.txt"))
                assert (
                    len(replay_file_matches) == 1
                ), f"No replay file in {log_path} or more than one"
                replay_files.append(replay_file_matches[0])

            for replay_file in replay_files:
                testing_configurations_outcomes = parse_experiment_file(
                    exp_file=replay_file, env_name=env_name, return_failures=False
                )

                # count number of failures in replay_file: if it is > percentage_failure_discard
                # then discard the configurations
                sum_failures = sum(
                    [
                        outcome > threshold_failure
                        for _, outcome in testing_configurations_outcomes
                    ]
                )
                percentage_failures = sum_failures / len(
                    testing_configurations_outcomes
                )
                if percentage_failures <= percentage_failure_discard or testing_mode:
                    for (
                        testing_configuration_outcome
                    ) in testing_configurations_outcomes:
                        dataset.add(
                            TestingData(
                                filename=replay_file,
                                testing_configuration=testing_configuration_outcome,
                            )
                        )
                else:
                    logger.warn(
                        f"Discarding the testing configurations of {replay_file} as percentage of failures {percentage_failures} > threshold {percentage_failure_discard}."
                    )

        labels = [data.get_label() for data in dataset.get()]

        if np.mean(labels) == 0.0 and not testing_mode:
            raise RuntimeError("No failures during replay")

        number_of_failures = sum([1 for data in dataset.get() if data.get_label() == 1])
        if number_of_failures < 50 and not testing_mode:
            raise RuntimeError(
                f"Number of failures is too low {number_of_failures} to learn a meaningful predictor"
            )

        if not testing_mode:
            logger.info("Number of samples: {}".format(len(labels)))
            logger.info("Data balancing: {}".format(np.mean(labels)))
            num_failures = reduce(
                lambda a, b: a + b, filter(lambda label: label == 1, labels)
            )
            num_successes = len(labels) - num_failures

            logger.info(f"Num failures: {num_failures}")
            logger.info(f"Num successes: {num_successes}")

    else:
        for log_path in log_paths:
            if not train_from_multiple_runs:
                log_path = os.path.join(log_path, LOG_PATH_CONFIGS_DIRECTORY)
                assert os.path.exists(log_path), "Log path {} does not exist".format(
                    log_path
                )

            if save_data and not train_from_multiple_runs:
                preprocessed_files_filepath = os.path.join(log_path, "preprocessed")
                if os.path.exists(preprocessed_files_filepath):
                    for filename in os.listdir(preprocessed_files_filepath):
                        os.remove(os.path.join(preprocessed_files_filepath, filename))

                os.makedirs(preprocessed_files_filepath, exist_ok=True)
                os.makedirs(
                    os.path.join(log_path, "preprocessed", "succeeded"), exist_ok=True
                )
                os.makedirs(
                    os.path.join(log_path, "preprocessed", "failed"), exist_ok=True
                )

            # TODO: refactor and do this in each training RL algo
            # to estimate the max ent_coef (assuming this is the name of the exploration param in all algos)
            max_exploration_coef = 0.0
            count = 0
            episode_indices = []

            if train_from_multiple_runs:
                runs_folders = glob.glob(os.path.join(log_path, "run_*"))
                # this happens when preprecessor is called during the testing phase, when the process loads the agent
                # from each training run directory
                if (
                    len(runs_folders) == 0
                    and TrainingType.original.name in log_path
                    and "run_" in log_path
                ):
                    # extract the parent directory to look for all the runs
                    # the -1 is to remove the path separator
                    parent_directory = log_path[: log_path.index("run_") - 1]
                    runs_folders = glob.glob(os.path.join(parent_directory, "run_*"))
                elif len(runs_folders) == 0:
                    raise RuntimeError(f"Could not find run folders in {log_path}")

                assert len(runs_folders) > 0, f"There are no runs in {log_path}"
                training_logs_all_runs = []
                for run_folder in runs_folders:
                    path_to_logs = os.path.join(
                        run_folder, LOG_PATH_CONFIGS_DIRECTORY, "training_logs.npz"
                    )
                    training_logs_run = np.load(path_to_logs, allow_pickle=True)[
                        "training_logs"
                    ]
                    for training_logs_run in training_logs_run:
                        # discard all configurations where agent only explores
                        if training_logs_run.agent_state is not None:
                            training_logs_all_runs.append(training_logs_run)

                sorted_by_training_progress = sorted(
                    training_logs_all_runs,
                    key=lambda training_logs_run: training_logs_run.agent_state[
                        "training_progress"
                    ],
                )
            else:
                if log_path.endswith(LOG_PATH_CONFIGS_DIRECTORY):
                    path_to_logs = os.path.join(log_path, "training_logs.npz")
                else:
                    path_to_logs = os.path.join(
                        log_path, LOG_PATH_CONFIGS_DIRECTORY, "training_logs.npz"
                    )

                # discard all configurations where agent only explores
                training_logs = list(
                    filter(
                        lambda training_log: training_log.agent_state is not None,
                        np.load(path_to_logs, allow_pickle=True)["training_logs"],
                    )
                )
                # already sorted
                sorted_by_training_progress = training_logs

            all_episodes_unfiltered += len(sorted_by_training_progress)

            for i, training_logs in enumerate(sorted_by_training_progress):
                exploration_coef = training_logs.get_exploration_coefficient()
                training_progress = training_logs.get_training_progress()
                if exploration_coef > max_exploration_coef:
                    max_exploration_coef = exploration_coef

                if (
                    training_progress_filter is None
                    or training_progress >= training_progress_filter
                ):
                    episode_indices.append(i)
                    data_point = TrainingData(index=i, training_logs=training_logs)
                    dataset.add(data_point)

                    if data_point.get_label() == 1:
                        num_failures += 1

                    if save_data:
                        image_array = training_logs.get_image()
                        if image_array is not None:
                            image = Image.fromarray(image_array.astype(np.uint8))
                            filepath = os.path.join(
                                log_path,
                                os.path.join(
                                    "preprocessed",
                                    "failed"
                                    if data_point.get_label() == 1
                                    else "succeeded",
                                    "env-{}-{}-{}.tiff".format(
                                        count,
                                        data_point.get_label(),
                                        round(training_progress, 0),
                                    ),
                                ),
                            )

                            image.save(fp=filepath)
                            count += 1

            min_episode = np.min(episode_indices)
            max_episode = np.max(episode_indices)

            # TODO: refactor and do this in each training RL algo
            if max_exploration_coef > 0.0:
                for data_point in dataset.get():
                    data_point.exploration_coefficient = round(
                        (data_point.exploration_coefficient / max_exploration_coef)
                        * 100,
                        2,
                    )

        labels = [data.get_label() for data in dataset.get()]
        logger.info("Number of samples: {}".format(len(labels)))
        logger.info("Data balancing: {}".format(np.mean(labels)))
        if train_from_multiple_runs:
            logger.info(
                "Training progress filter {}. Considering {} episodes instead of {}. "
                "Number of failures {}/{}".format(
                    training_progress_filter,
                    len(dataset.get()),
                    all_episodes_unfiltered,
                    num_failures,
                    len(dataset.get()),
                )
            )
        else:
            logger.info(
                "Training progress filter {}. Considering env configurations from episode {} to episode {}. "
                "Number of failures {}/{}".format(
                    training_progress_filter,
                    min_episode,
                    max_episode,
                    num_failures,
                    max_episode - min_episode + 1,
                )
            )

    return dataset
