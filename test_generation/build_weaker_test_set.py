import argparse
import logging
import os
import glob
import json
import numpy as np
from config import CARTPOLE_ENV_NAME, ENV_NAMES
from log import Log
from test_generation.env_configuration_factory import make_env_configuration

from test_generation.utils.env_utils import ALGOS
from test_generation.utils.file_utils import parse_experiment_file
from training.training_type import TrainingType


algos_list = list(ALGOS.keys())

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=algos_list,
)
parser.add_argument(
    "--env-name",
    type=str,
    default=CARTPOLE_ENV_NAME,
    choices=ENV_NAMES,
    help="environment ID",
)
parser.add_argument(
    "--num-configurations-to-select",
    type=int,
    default=100,
    help="Number of configurations to select",
)
args = parser.parse_args()

if __name__ == "__main__":

    confidence_type = "fitness"
    num_configurations_to_select = args.num_configurations_to_select

    logger = Log("build_weaker_test_set")
    log_filename = "build-weaker-test-set"

    original_agent_folder = os.path.join(
        args.folder, args.algo, f"{args.env_name}_{TrainingType.original.name}"
    )
    assert os.path.exists(
        original_agent_folder
    ), f"Original agent folder {original_agent_folder} does not exist"

    # save logs on original agent folder
    logging.basicConfig(
        filename=os.path.join(original_agent_folder, f"{log_filename}.txt"),
        filemode="w",
        level=logging.DEBUG,
    )

    weak_configurations = []
    original_run_folders = glob.glob(os.path.join(original_agent_folder, "run_*"))

    num_runs = len(original_run_folders)
    env_configs_confidence = {}
    env_configs_failure_probabilities_dict = {}

    for run_num, original_run_folder in enumerate(original_run_folders):
        weak_statistics = glob.glob(
            os.path.join(original_run_folder, "testing-random-*-weak-statistics.json")
        )
        assert (
            len(weak_statistics) == 1
        ), "There must be one filed named *weak-statistics.json"
        weak_statistics = weak_statistics[0]
        weak_file = glob.glob(
            os.path.join(original_run_folder, "testing-random-*-weak.txt")
        )
        assert len(weak_file) == 1, "There must be one filed named *weak.txt"
        weak_file = weak_file[0]

        with open(
            weak_statistics,
            "r+",
            encoding="utf-8",
        ) as f:
            statistics = json.load(f)
            if confidence_type == "fitness":
                confidence_values = statistics["fitness_values"]
                failure_probabilities_values = statistics["failure_probabilities"]
            else:
                raise NotImplementedError(
                    f"Confidence type {confidence_type} not implemented"
                )

        env_configs_failure_probabilities = parse_experiment_file(
            exp_file=weak_file, env_name=args.env_name, return_failures=False
        )

        confidence_values_to_append = []
        env_configs_to_append = []
        for i in range(len(confidence_values)):
            if (
                env_configs_failure_probabilities[i][0].get_str()
                not in env_configs_confidence
            ):
                env_configs_confidence[
                    env_configs_failure_probabilities[i][0].get_str()
                ] = []
                env_configs_failure_probabilities_dict[
                    env_configs_failure_probabilities[i][0].get_str()
                ] = []

            env_configs_confidence[
                env_configs_failure_probabilities[i][0].get_str()
            ].append(confidence_values[i])
            env_configs_failure_probabilities_dict[
                env_configs_failure_probabilities[i][0].get_str()
            ].append(env_configs_failure_probabilities[i][1])

    env_configs = []
    confidence_values = []
    for key in env_configs_confidence:
        # filter out configurations that are failures (can have a high confidence
        # even though they are failures, which is not what we want to select; this may
        # happen with LunarLander, when one of the failure mode is hovering over the target
        # i.e., the agent fails due to timeout)
        min_idx = np.argmin(env_configs_confidence[key])
        if env_configs_confidence[key][min_idx] > 0.0:
            # make sure that the configuration does not cause a failure
            # in at least 70% of the runs
            if (
                (
                    len(env_configs_confidence[key])
                    - len(
                        list(
                            filter(
                                lambda fp: fp > 0.5,
                                env_configs_failure_probabilities_dict[key],
                            )
                        )
                    )
                )
                / len(env_configs_confidence[key])
            ) > 0.5:
                confidence_values.append(np.min(env_configs_confidence[key]))
                env_configs.append(
                    make_env_configuration(env_name=args.env_name, env_config=key)
                )
        else:
            confidence_values.append(np.min(env_configs_confidence[key]))
            env_configs.append(
                make_env_configuration(env_name=args.env_name, env_config=key)
            )

    sorted_indices = np.argsort(confidence_values)[::-1]
    assert num_configurations_to_select <= len(sorted_indices), (
        f"Number of configurations to select {num_configurations_to_select} cannot be "
        f"greater than the number of configurations {len(sorted_indices)}."
    )

    for i in range(num_configurations_to_select):
        index = sorted_indices[i]
        logger.debug(
            f"Selecting configuration {index} with confidence {confidence_values[index]}"
        )
        weak_configurations.append(env_configs[index])

    logger.debug(f"Selected {len(weak_configurations)} env configs")

    numpy_dict = {"env_configs": np.asarray(weak_configurations)}
    np.savez_compressed(
        os.path.join(original_agent_folder, "weaker_configs.npz"), **numpy_dict
    )
