import json
import os
from typing import List, Optional, Tuple, Union

from config import ENV_NAMES
import re

from test_generation.env_configuration import EnvConfiguration
from test_generation.env_configuration_factory import make_env_configuration


def get_run_number(filepath: str, mutant_name: str) -> Optional[int]:
    run_pattern = re.compile(r"run_(\d+)")
    if "run" in filepath:
        if "original" in filepath and mutant_name is not None:
            # FIXME: assuming no mutant having value -1
            return -1
        run_match = run_pattern.search(filepath)
        assert (
            run_match is not None
        ), f"There should be a match in {filepath} for the word 'run_'."
        try:
            return int(run_match.group(1))
        except IndexError as _:
            raise RuntimeError(
                f"The first group in run_match {run_match} should be the run number."
            )
    return None


def get_one_hot_for_mutant_coef(
    coef_number: Union[int, float], mutant_name: str, log_path: str
) -> Optional[List[int]]:
    folders = [
        folder
        for folder in os.listdir(log_path)
        if os.path.isdir(os.path.join(log_path, folder))
    ]
    mutant_folders = filter(lambda fld: mutant_name in fld, folders)
    sorted_mutant_folders = sorted(
        mutant_folders, key=lambda mutant_folder: float(mutant_folder.split("_")[-1])
    )
    one_hot = [0] * len(sorted_mutant_folders)
    if coef_number == -1:
        return one_hot

    for i, mutant_folder in enumerate(sorted_mutant_folders):
        if str(coef_number) in mutant_folder:
            one_hot[i] = 1
            return one_hot

    raise RuntimeError(
        f"No match found for mutant configuration {coef_number} of mutant {mutant_name} in folder {log_path}."
    )


def get_coef_number(filepath: str, mutant_name: str) -> Optional[Union[int, float]]:
    coef_pattern = re.compile(r"_mutant_(\w+)_(\d+(\.\d+)?)")
    if "mutant" in filepath and mutant_name is not None:
        coef_match = coef_pattern.search(filepath)
        assert (
            coef_match is not None
        ), f"There should be a match in {filepath} for the word 'mutant_'."

        try:
            coef_value = coef_match.group(2)
            try:
                coef_value = int(coef_value)
                return coef_value
            except ValueError as _:
                pass

            try:
                coef_value = float(coef_value)
                return coef_value
            except ValueError as _:
                pass

            raise NotImplementedError(f"Unknown type: {type(coef_value)}")

        except IndexError as _:
            raise RuntimeError(
                f"The second group in coef_match {coef_match} should be the coef value."
            )
    if "original" in filepath and mutant_name is not None:
        # FIXME: assuming no mutant having value -1
        return -1
    return None


def read_logs(log_path: str, filename: str) -> json:
    assert os.path.exists(
        os.path.join(log_path, filename)
    ), "Filename {} not found".format(os.path.join(log_path, filename))
    with open(os.path.join(log_path, filename), "r+", encoding="utf-8") as f:
        return json.load(f)


def get_training_logs_path(
    folder: str, algo: str, env_id: str, exp_id: int, resume_dir: str = None
) -> str:
    if resume_dir is not None:
        log_path = os.path.join(folder, algo, resume_dir)
    else:
        log_path = os.path.join(folder, algo, "{}_{}".format(env_id, exp_id))
    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
    return log_path


# FIXME: to refactor (very fragile)
def parse_experiment_file(
    exp_file: str, env_name: str, return_failures: bool = True
) -> Union[List[EnvConfiguration], List[Tuple[EnvConfiguration, float]]]:
    assert os.path.exists(exp_file), "Exp file {} not found".format(exp_file)
    assert env_name in ENV_NAMES, "Env name {} not valid. Choose among: {}".format(
        env_name, ENV_NAMES
    )
    result = []
    f = False
    num_experiments = -1
    # for non deterministic environments
    num_env_configurations = -1

    pattern_num_experiments = r".*Num experiments: (\d+)/(\d+)"

    with open(exp_file, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            if "Num experiments" in line and num_experiments == -1:
                match = re.match(pattern_num_experiments, line)
                assert (
                    match is not None
                ), f"Pattern '{pattern_num_experiments}' does not match line '{line}'"
                num_experiments = int(match.group(2))

            if "Generating" in line:
                sentence = line.split(":")[2]
                assert (
                    "env configurations" in sentence
                ), f"Sentence {sentence} should contain the string 'env configurations'"
                num_env_configurations = int(
                    "".join(list(filter(str.isdigit, sentence)))
                )

            if return_failures and "FAIL -" in line:
                env_config_str = (
                    line.split(" - ")[1]
                    .split(":")[0]
                    .replace("Failure probability for env config ", "")
                )
                result.append(
                    make_env_configuration(env_config=env_config_str, env_name=env_name)
                )
            elif (
                not return_failures
                and f is True
                and "Failure probability for env config" in line
            ):
                env_config_str = (
                    line.split(":")[2]
                    .replace("FAIL - ", "")
                    .replace("Failure probability for env config ", "")
                )
                failure_probability = float(
                    line.split(":")[3].replace(" ", "").split("(")[1].split(",")[0]
                )
                env_config = make_env_configuration(
                    env_config=env_config_str, env_name=env_name
                )
                result.append((env_config, failure_probability))
            elif not return_failures and "Num experiments:" in line:
                match = re.match(pattern_num_experiments, line)
                assert (
                    match is not None
                ), f"Pattern {pattern_num_experiments} does not match line {line}"
                current_episodes_count = int(match.group(1))
                total_episodes_count = int(match.group(2))
                if current_episodes_count == total_episodes_count:
                    f = True

    if return_failures:
        assert len(result) > 0, "No failure in exp file {}".format(exp_file)
        return result

    assert len(result) > 0, "No env configuration in exp file {}".format(exp_file)
    assert num_experiments > 0, "Num experiments cannot be <= 0"
    assert (
        len(result) == num_experiments or len(result) == num_env_configurations
    ), f"Error when parsing file {exp_file}. Number of instantiated configurations {len(result)} does not correspond with the number of experiments {num_experiments} or with the number of declared configurations {num_env_configurations}."

    return result
