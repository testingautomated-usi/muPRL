import argparse
import glob
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
from typing import List, Tuple

import shutil
import numpy as np
import statsmodels.stats.proportion as smp
from config import LOG_PATH_CONFIGS_DIRECTORY
from my_experiment_manager import MyExperimentManager
from evaluate import check_killability_triviality
from joblib import Parallel, delayed
from log import Log, close_loggers
from mutation_utils import create_results_folders
from training.training_type import TrainingType

from test_generation.env_configuration import EnvConfiguration
from test_generation.test_configurator import TestConfigurator
from test_generation.testing_args import TestingArgs


class TestAgent:
    def __init__(
        self, test_configurator: TestConfigurator, all_args: argparse.Namespace
    ):
        self.args = all_args
        self.test_configurator = test_configurator

    @staticmethod
    def test_single_agent(
        test_configurator: TestConfigurator,
        run_folder: str,
        threshold_failure: float,
        log: Log,
        reset_random_gen: bool = False,
        env_configurations: List[EnvConfiguration] = None,
        parallelize: bool = False,
        mutant_name: str = None,
        mutant_configuration: str = None,
        num_run: int = -1,
        encode_run_and_conf: bool = False,
    ) -> Tuple[float, List[float], List[float]]:
        st = time.perf_counter()

        test_configurator.initialize(
            folder_with_agent=run_folder,
            reset_random_gen=reset_random_gen,
            configurations=env_configurations,
            parallelize=parallelize,
            mutant_name=mutant_name,
            mutant_configuration=mutant_configuration,
            num_run=num_run,
            encode_run_and_conf=encode_run_and_conf,
        )

        all_failure_probabilities = []

        num_experiments = 0
        num_failures = 0
        previous_env_config = None
        map_env_config_failure_prob = dict()
        map_env_config_min_fitness = dict()
        num_trials = 0
        episode_num = 0
        min_fitness_values = []

        while (
            num_experiments
            < test_configurator.num_runs_each_env_config
            * test_configurator.num_episodes
        ):
            if (
                num_experiments % test_configurator.num_runs_each_env_config == 0
                and num_experiments != 0
            ):
                map_env_config_failure_prob[previous_env_config.get_str()] = (
                    num_failures / test_configurator.num_runs_each_env_config,
                    smp.proportion_confint(
                        count=num_failures,
                        nobs=test_configurator.num_runs_each_env_config,
                        method="wilson",
                    ),
                )
                log.info(
                    "Failure probability for env config {}: {}".format(
                        previous_env_config.get_str(),
                        map_env_config_failure_prob[previous_env_config.get_str()],
                    )
                )
                log.info(
                    f"Failure probability {map_env_config_failure_prob[previous_env_config.get_str()]}"
                )
                num_failures = 0
                episode_num = 0
                num_trials += 1

                if len(min_fitness_values) > 0:
                    map_env_config_min_fitness[previous_env_config.get_str()] = float(
                        np.mean(min_fitness_values)
                    )
                    min_fitness_values.clear()

            failure, env_config, fitness_values = test_configurator.test_single_episode(
                episode_num=episode_num, num_trials=num_trials
            )

            if len(fitness_values) > 0:
                min_fitness_values.append(min(fitness_values))
                log.debug(f"Min fitness value: {min(fitness_values)}")

            previous_env_config = env_config
            if failure:
                num_failures += 1
            num_experiments += 1
            episode_num += 1
            log.debug(
                f"Num experiments: {num_experiments}/"
                f"{test_configurator.num_runs_each_env_config * test_configurator.num_episodes}"
            )

        if len(map_env_config_failure_prob) > 0:
            map_env_config_failure_prob[previous_env_config.get_str()] = (
                num_failures / test_configurator.num_runs_each_env_config,
                smp.proportion_confint(
                    count=num_failures,
                    nobs=test_configurator.num_runs_each_env_config,
                    method="wilson",
                ),
            )
        failure_probabilities = []

        if len(map_env_config_min_fitness) > 0:
            map_env_config_min_fitness[previous_env_config.get_str()] = np.mean(
                min_fitness_values
            )
            min_fitness_values.clear()

        num_failures = 0
        for key, value in map_env_config_failure_prob.items():
            if value[0] > threshold_failure:
                num_failures += 1
                log.info(
                    "FAIL - Failure probability for env config {}: {}".format(
                        key, value
                    )
                )
            else:
                log.info("Failure probability for env config {}: {}".format(key, value))
            failure_probabilities.append(value[0])

        all_failure_probabilities.extend(failure_probabilities)
        if len(failure_probabilities) > 0:
            log.info(
                "Failure probabilities: {}, Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                    failure_probabilities,
                    np.mean(failure_probabilities),
                    np.std(failure_probabilities),
                    np.min(failure_probabilities),
                    np.max(failure_probabilities),
                )
            )

        if len(min_fitness_values) > 0:
            log.info(
                "Min fitness values: Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}, Values: {}".format(
                    np.mean(min_fitness_values),
                    np.std(min_fitness_values),
                    np.min(min_fitness_values),
                    np.max(min_fitness_values),
                    min_fitness_values,
                )
            )
        elif len(map_env_config_min_fitness) > 0:
            mean_fitness_values = [
                mean_fitness_value
                for mean_fitness_value in map_env_config_min_fitness.values()
            ]
            log.info(
                "Min fitness values: Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}, Values: {}".format(
                    np.mean(mean_fitness_values),
                    np.std(mean_fitness_values),
                    np.min(mean_fitness_values),
                    np.max(mean_fitness_values),
                    mean_fitness_values,
                )
            )

        log.info("{}".format(map_env_config_failure_prob.keys()))

        log.info(
            "Number of evaluation predictions (i.e. number of times a model was used to make predictions): {}".format(
                test_configurator.get_num_evaluation_predictions()
            )
        )
        test_configurator.close_env()

        close_loggers()

        assert (
            len(all_failure_probabilities) > 0
        ), "Experiments did not produce any results"

        num_total_failures = len(
            list(filter(lambda v: v > 0.5, all_failure_probabilities))
        )
        if test_configurator.testing_strategy_name != "prioritized_replay":
            # in prioritized_replay there can be duplicated environment configurations
            log.info(
                f"Considering {test_configurator.num_episodes} "
                f"experiment(s) the {test_configurator.testing_strategy_for_logs} method generated: "
                f"{num_total_failures} failures / {len(all_failure_probabilities)} episodes"
            )
        log.info(
            "Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                np.mean(all_failure_probabilities),
                np.std(all_failure_probabilities),
                np.min(all_failure_probabilities),
                np.max(all_failure_probabilities),
            )
        )

        success_rate = (len(all_failure_probabilities) - num_total_failures) / len(
            all_failure_probabilities
        )
        min_fitness_s = None
        if len(min_fitness_values) > 0:
            min_fitness_s = min_fitness_values
        elif len(map_env_config_min_fitness) > 0:
            min_fitness_s = [
                mean_fitness_value
                for mean_fitness_value in map_env_config_min_fitness.values()
            ]

        # TODO: refactor with a class
        json_string = json.dumps(
            {
                "seed": test_configurator.seed,
                "failure_probabilities": all_failure_probabilities,
                "fitness_values": min_fitness_s if min_fitness_s is not None else [],
                "num_total_failures": num_total_failures,
                "success_rate": success_rate,
                "time_elapsed_s": round(time.perf_counter() - st, 2),
                "avg_min_fitness": (
                    np.mean(min_fitness_s) if min_fitness_s is not None else None
                ),
            },
            indent=4,
        )

        statistics_filename = (
            f"testing-{test_configurator.testing_strategy_for_logs}-statistics.json"
        )

        if "strong" in test_configurator.testing_strategy_for_logs:
            # at this point we have access to the path of the trained policy
            assert (
                test_configurator.test_generator.trained_policy_save_path is not None
            ), "Path to the trained policy is not available"
            if (
                mutant_name
                not in test_configurator.test_generator.trained_policy_save_path
            ):
                testing_strategy_for_logs_filename = os.path.join(
                    run_folder,
                    f"testing-{test_configurator.testing_strategy_for_logs}.txt",
                )
                testing_strategy_for_logs_filename_renamed = os.path.join(
                    run_folder,
                    f"testing-{test_configurator.testing_strategy_for_logs.replace(f'{mutant_name}-', '')}.txt",
                )
                statistics_filename = statistics_filename.replace(f"{mutant_name}-", "")
                os.rename(
                    testing_strategy_for_logs_filename,
                    testing_strategy_for_logs_filename_renamed,
                )

        with open(
            os.path.join(run_folder, statistics_filename), "w+", encoding="utf-8"
        ) as f:
            f.write(json_string)

        return success_rate, all_failure_probabilities, min_fitness_s


def run_tests_on_agent(
    mode: str,
    num_run: int,
    run_folder: str,
    threshold_failure: float,
    random_seed: int,
    test_agent: TestAgent,
    test_configurator: TestConfigurator,
    training_type: TrainingType,
    mutant_name: str,
    log: Log,
    mutant_configuration: str = None,
    env_configurations: List[EnvConfiguration] = None,
    num_configurations_run: int = 0,
    parallelize: bool = False,
    encode_run_and_conf: bool = False,
) -> None:
    test_configurator.seed = random_seed

    if mode == "replay":
        assert (
            num_configurations_run > 0
        ), "Replay requires to specify the number of episodes"
        test_configurator.testing_strategy_name = "random"
        test_configurator.testing_strategy_for_logs = f"random-{random_seed}-replay"
        test_configurator.num_episodes = num_configurations_run
    elif mode == "weak" or mode == "strong" or mode == "weaker":
        if mode == "weak":
            test_configurator.testing_strategy_name = "random"
        elif mode == "strong":
            test_configurator.testing_strategy_name = "nn"
        elif mode == "weaker":
            test_configurator.testing_strategy_name = "constant"
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")

        if mode == "weak":
            test_configurator.testing_strategy_for_logs = f"random-{random_seed}-weak"
        elif mode == "strong":
            # FIXME: investigate why this is useful
            if "strong" in test_configurator.testing_strategy_for_logs:
                testing_strategy_for_logs_original = (
                    test_configurator.testing_strategy_for_logs[
                        : test_configurator.testing_strategy_for_logs.rindex(
                            "-",
                            0,
                            test_configurator.testing_strategy_for_logs.rindex("-"),
                        )
                    ]
                )
            else:
                testing_strategy_for_logs_original = (
                    test_configurator.testing_strategy_for_logs
                )

            if training_type == TrainingType.original and mutant_name is not None:
                test_configurator.testing_strategy_for_logs = f"{testing_strategy_for_logs_original}-{random_seed}-{mutant_name}-strong"
            else:
                test_configurator.testing_strategy_for_logs = (
                    f"{testing_strategy_for_logs_original}-{random_seed}-strong"
                )
        elif mode == "weaker":
            test_configurator.testing_strategy_for_logs = "constant-weaker"
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

    testing_logs_statistics_file = os.path.join(
        run_folder,
        f"testing-{test_configurator.testing_strategy_for_logs}-statistics.json",
    )
    # this is to allow failure predictors trained on original agent
    mutant_name_str = mutant_name if mutant_name is not None else ""
    testing_logs_statistics_file_without_mutant_name = os.path.join(
        run_folder,
        f"testing-{test_configurator.testing_strategy_for_logs.replace(f'{mutant_name_str}-', '')}-statistics.json",
    )

    existing_condition = (
        (
            os.path.exists(testing_logs_statistics_file)
            or os.path.exists(testing_logs_statistics_file_without_mutant_name)
        )
        if "strong" in test_configurator.testing_strategy_for_logs
        else os.path.exists(testing_logs_statistics_file)
    )

    if training_type == TrainingType.mutant:
        if existing_condition:
            if not parallelize:
                log.info(
                    f"------ Skipping execution {mode} in mutant {mutant_name}. "
                    f"Delete the file "
                    f"{testing_logs_statistics_file} or {testing_logs_statistics_file_without_mutant_name} to run "
                    f"the configurations again. ------"
                )
            else:
                print(
                    f"------ Skipping execution {mode} in mutant {mutant_name}. "
                    f"Delete the file "
                    f"{testing_logs_statistics_file} or {testing_logs_statistics_file_without_mutant_name} to run "
                    f"the configurations again. ------"
                )
            return
        log.info(
            f"------ Executing {mode} in mutant {mutant_name}."
            f"Seed: {random_seed}, Num episodes: {num_configurations_run} ------"
        )
    else:
        if existing_condition:
            if not parallelize:
                log.info(
                    f"------ Skipping execution {mode} in original agent run: {num_run}. "
                    f"Delete the file "
                    f"{testing_logs_statistics_file} or {testing_logs_statistics_file_without_mutant_name} to run "
                    f"the configurations again. ------"
                )
            else:
                print(
                    f"------ Skipping execution {mode} in original agent run: {num_run}. "
                    f"Delete the file "
                    f"{testing_logs_statistics_file} or {testing_logs_statistics_file_without_mutant_name} to run "
                    f"the configurations again. ------"
                )
            return
        if not parallelize:
            log.info(
                f"------ Executing {mode} in original agent run: {num_run}. "
                f"Seed: {random_seed}, Num episodes: {num_configurations_run} ------"
            )
        else:
            print(
                f"------ Executing {mode} in original agent run: {num_run}. "
                f"Seed: {random_seed}, Num episodes: {num_configurations_run} ------"
            )

    test_agent.test_single_agent(
        run_folder=run_folder,
        test_configurator=test_configurator,
        threshold_failure=threshold_failure,
        reset_random_gen=True,
        parallelize=parallelize,
        env_configurations=env_configurations,
        mutant_name=mutant_name,
        num_run=num_run,
        encode_run_and_conf=encode_run_and_conf,
        log=log,
    )


if __name__ == "__main__":
    args = TestingArgs().parse()

    pltf = platform.system()
    if pltf.lower() == "windows" and args.parallelize:
        print("Disabling parallelization in Windows")
        args.parallelize = False

    logger = Log("test_agent", parallelize=args.parallelize)
    logger.info("Args: {}".format(args))

    args.training_type = TrainingType.parse(name_as_str=args.training_type)

    test_configurator_args = {
        "env_name": args.env_name,
        "render": args.render,
        "seed": args.seed,
        "device": args.device,
        "num_envs": 1,
        "folder": args.folder,
        "algo": args.algo,
        "exp_id": args.exp_id,
        "testing_policy_for_training_name": args.testing_policy_for_training_name,
        "testing_strategy_name": args.testing_strategy_name,
        "model_checkpoint": args.model_checkpoint,
        "training_progress_filter": args.training_progress_filter,
        "layers": args.layers,
        "num_episodes": args.num_episodes,
        "exp_name": "test_agent",
        "num_runs_each_env_config": args.num_runs_each_env_config,
        "budget": args.budget,
        "sampling_size": args.sampling_size,
        "register_environment": args.register_env,
        "training_type": args.training_type,
        "test_multiple_agents": args.test_multiple_agents,
        "parallelize": args.parallelize,
        **args.wrapper_kwargs,
    }

    if args.mode != "replay":
        assert (
            args.mutant_name is not None and args.mutant_name != ""
        ), f"Mutant name cannot be empty in {args.mode} mode"
        assert (
            args.num_episodes > 0
        ), f"Specify the number of episodes in {args.mode} mode"

    encode_run_and_conf = not args.do_not_encode_run_and_conf
    if args.mutant_configuration == "":
        args.mutant_configuration = None

    num_cpus = args.num_cpus

    if num_cpus == -1:
        num_cpus = multiprocessing.cpu_count()
    else:
        assert (
            num_cpus <= multiprocessing.cpu_count()
        ), f"Num cpus {num_cpus} cannot be > than the number of logical cores in the current machine {multiprocessing.cpu_count()}"

    # TODO: refactor in a function; there is a lot of duplicated code considering all the three testing modes
    # assuming that the original agent is already present; if not then raise error
    original_folder_name = (
        f"{args.folder}{os.path.sep}{args.algo}{os.path.sep}{args.env_name}_original"
    )
    assert os.path.exists(
        original_folder_name
    ), f"Original agent folder {original_folder_name} does not exist"

    if args.training_type == TrainingType.mutant:
        assert args.mutant_name is not None, "Specify the 'mutant-name' parameter"

        if args.mutant_configuration is not None:
            mutants_folders = [
                os.path.join(
                    f"{args.folder}{os.path.sep}{args.algo}",
                    f"{args.env_name}_mutant_{args.mutant_name}_{args.mutant_configuration}",
                )
            ]
            assert (
                os.path.exists(mutants_folders[0]) > 0
            ), f"Mutant configuration {args.mutant_configuration} of mutant {args.mutant_name} not found"
        else:
            mutants_folders = glob.glob(
                os.path.join(
                    f"{args.folder}{os.path.sep}{args.algo}",
                    f"{args.env_name}_mutant_{args.mutant_name}_*",
                )
            )

    original_runs_folders = sorted(
        glob.glob(os.path.join(original_folder_name, "run_*")),
        key=lambda filepath: int(filepath.split("_")[-1]),
    )

    if args.mode == "replay":
        if args.parallelize:
            print("****** Replay testing mode ******")
        else:
            logger.info("****** Replay testing mode ******")

        # replay training configurations of the original
        seeds = []
        nums_configurations = []

        if args.parallelize:
            logger.info(
                "****** Executing training configurations of original agent ******"
            )
        else:
            print("****** Executing training configurations of original agent ******")

        for i, original_run_folder in enumerate(original_runs_folders):
            num_configurations = len(
                np.load(
                    os.path.join(
                        original_run_folder,
                        LOG_PATH_CONFIGS_DIRECTORY,
                        "training_logs.npz",
                    ),
                    allow_pickle=True,
                )["training_logs"]
            )
            assert (
                num_configurations > 0
            ), f"Num configurations in run {os.path.join(original_run_folder, LOG_PATH_CONFIGS_DIRECTORY)} must be > 0"
            nums_configurations.append(num_configurations)
            with open(
                os.path.join(original_run_folder, "statistics.json"),
                "r+",
                encoding="utf-8",
            ) as f:
                seed = json.load(f)["seed"]
                seeds.append(seed)

        if args.parallelize:
            start_time = time.perf_counter()
            with Parallel(
                n_jobs=num_cpus, batch_size="auto", backend="multiprocessing"
            ) as parallel:
                test_configurator_instance = TestConfigurator(**test_configurator_args)
                test_agent_instance = TestAgent(
                    test_configurator=test_configurator_instance, all_args=args
                )
                res = parallel(
                    (
                        delayed(run_tests_on_agent)(
                            mode=args.mode,
                            num_run=i,
                            threshold_failure=args.threshold_failure,
                            run_folder=original_run_folder,
                            num_configurations_run=nums_configurations[i],
                            random_seed=seeds[i],
                            test_configurator=test_configurator_instance,
                            test_agent=test_agent_instance,
                            training_type=TrainingType.original,
                            mutant_name=args.mutant_name,
                            parallelize=True,
                            log=logger,
                        )
                        for i, original_run_folder in enumerate(original_runs_folders)
                    ),
                )
            print(f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s")
        else:
            start_time = time.perf_counter()

            if args.run_num > -1:
                assert (
                    0 <= args.run_num < len(original_runs_folders)
                ), f"Run number {args.run_num} cannot be >= number of runs {len(original_runs_folders)} or < 0"

            for i, original_run_folder in enumerate(original_runs_folders):

                if args.run_num > -1 and args.run_num != i:
                    continue

                test_configurator_instance = TestConfigurator(**test_configurator_args)
                test_agent_instance = TestAgent(
                    test_configurator=test_configurator_instance, all_args=args
                )
                run_tests_on_agent(
                    mode=args.mode,
                    num_run=i,
                    threshold_failure=args.threshold_failure,
                    run_folder=original_run_folder,
                    num_configurations_run=nums_configurations[i],
                    random_seed=seeds[i],
                    test_configurator=test_configurator_instance,
                    test_agent=test_agent_instance,
                    training_type=TrainingType.original,
                    mutant_name=args.mutant_name,
                    log=logger,
                )
            logger.info(
                f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
            )

        if args.parallelize:
            print(
                f"****** Executing training configurations of mutant {args.mutant_name} ******"
            )
        else:
            logger.info(
                f"****** Executing training configurations of mutant {args.mutant_name} ******"
            )

        if args.training_type == TrainingType.mutant:

            for mutant_folder in mutants_folders:
                if args.parallelize:
                    print(
                        f"****** Executing training configurations of mutant in folder {mutant_folder} ******"
                    )
                else:
                    logger.info(
                        f"****** Executing training configurations of mutant in folder {mutant_folder} ******"
                    )

                mutant_runs_folders = sorted(
                    glob.glob(os.path.join(mutant_folder, "run_*")),
                    key=lambda filepath: int(filepath.split("_")[-1]),
                )

                if args.parallelize:
                    start_time = time.perf_counter()
                    with Parallel(
                        n_jobs=num_cpus, batch_size="auto", backend="multiprocessing"
                    ) as parallel:
                        test_configurator_instance = TestConfigurator(
                            **test_configurator_args
                        )
                        test_agent_instance = TestAgent(
                            test_configurator=test_configurator_instance, all_args=args
                        )
                        res = parallel(
                            (
                                delayed(run_tests_on_agent)(
                                    mode=args.mode,
                                    num_run=i,
                                    threshold_failure=args.threshold_failure,
                                    run_folder=mutant_run_folder,
                                    num_configurations_run=nums_configurations[i],
                                    random_seed=seeds[i],
                                    test_configurator=test_configurator_instance,
                                    test_agent=test_agent_instance,
                                    training_type=TrainingType.mutant,
                                    mutant_name=args.mutant_name,
                                    parallelize=True,
                                    log=logger,
                                )
                                for i, mutant_run_folder in enumerate(
                                    mutant_runs_folders
                                )
                            ),
                        )
                    print(
                        f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                    )
                else:
                    start_time = time.perf_counter()

                    if args.run_num > -1:
                        assert (
                            0 <= args.run_num < len(original_runs_folders)
                        ), f"Run number {args.run_num} cannot be >= number of runs {len(mutant_runs_folders)} or < 0"

                    for i, mutant_run_folder in enumerate(mutant_runs_folders):

                        if args.run_num > -1 and args.run_num != i:
                            continue

                        test_configurator_instance = TestConfigurator(
                            **test_configurator_args
                        )
                        test_agent_instance = TestAgent(
                            test_configurator=test_configurator_instance, all_args=args
                        )
                        run_tests_on_agent(
                            mode=args.mode,
                            num_run=i,
                            threshold_failure=args.threshold_failure,
                            run_folder=mutant_run_folder,
                            num_configurations_run=nums_configurations[i],
                            random_seed=seeds[i],
                            test_configurator=test_configurator_instance,
                            test_agent=test_agent_instance,
                            training_type=TrainingType.mutant,
                            mutant_name=args.mutant_name,
                            log=logger,
                        )
                    logger.info(
                        f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                    )

            if len(mutants_folders) > 0 and args.delete_mutants_after_replay:
                subj_path, _ = create_results_folders(algo=args.algo, env=args.env_name)
                (killable_operators, killability_reports_conf, triviality_report) = (
                    check_killability_triviality(
                        subj_path=subj_path,
                        do_triviality=True,
                        algo=args.algo,
                        env=args.env_name,
                        num_runs=len(original_runs_folders),
                        threshold_failure=args.threshold_failure,
                        bootstrap=False
                    )
                )
                mutant_in_killable_operators = any(
                    filter(
                        lambda killable_operator: killable_operator["name"]
                        == args.mutant_name,
                        killable_operators,
                    )
                )
                if mutant_in_killable_operators:
                    if (Path(__file__).parent / "hyperparams").is_dir():
                        # Package version
                        yaml_filepath = Path(__file__).parent
                    else:
                        # Take the root folder
                        yaml_filepath = Path(__file__).parent.parent
                    hyperparams = MyExperimentManager.read_hyperparameters_static(
                        yaml_filepath=os.path.join(yaml_filepath, "hyperparams"),
                        env_name=args.env_name,
                        algo=args.algo,
                    )
                    original_mutant_value = hyperparams[args.mutant_name]
                    if isinstance(original_mutant_value, int):
                        cast_type = int
                    elif isinstance(original_mutant_value, float):
                        cast_type = float
                    else:
                        raise NotImplementedError(
                            f"Mutant value type {type(original_mutant_value)} not supported"
                        )

                    killable_mutant_values = list(
                        map(
                            lambda report: cast_type(report[1]),
                            filter(
                                lambda report: report[0] == args.mutant_name
                                and report[2],
                                killability_reports_conf[1:],
                            ),
                        )
                    )
                    trivial_mutant_values = list(
                        map(
                            lambda report: cast_type(report[1]),
                            filter(
                                lambda report: report[0] == args.mutant_name
                                and report[2],
                                triviality_report[1:],
                            ),
                        )
                    )

                    sorted_tuple_distance_killable_mutants = sorted(
                        map(
                            lambda mutant_value: (
                                abs(original_mutant_value - mutant_value),
                                mutant_value,
                            ),
                            killable_mutant_values,
                        )
                    )
                    closest_to_original_mutant_value = next(
                        filter(
                            lambda mutant_distance_value: mutant_distance_value[1]
                            not in trivial_mutant_values,
                            sorted_tuple_distance_killable_mutants,
                        ),
                        None,
                    )
                    if closest_to_original_mutant_value is not None:
                        closest_to_original_mutant_value = (
                            closest_to_original_mutant_value[1]
                        )

                    if args.parallelize:
                        if closest_to_original_mutant_value is None:
                            print(
                                f"****** Discarding all mutant folders as all killable configurations of {args.mutant_name} are trivial ******"
                            )
                        else:
                            print(
                                f"****** Keeping only mutant value {closest_to_original_mutant_value} of {args.mutant_name}, "
                                f"which is the closest to the original value {original_mutant_value} and killable. ******"
                            )
                    else:
                        if closest_to_original_mutant_value is None:
                            logger.info(
                                f"****** Discarding all mutant folders as all killable configurations of {args.mutant_name} are trivial ******"
                            )
                        else:
                            logger.info(
                                f"****** Keeping only mutant value {closest_to_original_mutant_value} of {args.mutant_name}, "
                                f"which is the closest to the original value {original_mutant_value} and killable. ******"
                            )

                    for mutant_folder in mutants_folders:
                        if closest_to_original_mutant_value != cast_type(
                            mutant_folder.split("_")[-1]
                        ):
                            if args.parallelize:
                                print(
                                    f"****** Deleting mutant folder {mutant_folder} ******"
                                )
                            else:
                                logger.info(
                                    f"****** Deleting mutant folder {mutant_folder} ******"
                                )
                            shutil.rmtree(mutant_folder)

                else:
                    logger.info(
                        f"****** Deleting all mutant folders as {args.mutant_name} is not killable ******"
                    )
                    for mutant_folder in mutants_folders:
                        logger.info(
                            f"****** Deleting mutant folder {mutant_folder} ******"
                        )
                        shutil.rmtree(mutant_folder)

    elif args.mode == "weak" or args.mode == "strong" or args.mode == "weaker":

        env_configurations = None
        if args.mode == "weaker":
            weaker_configs_file = os.path.join(
                original_folder_name, "weaker_configs.npz"
            )
            assert os.path.exists(
                weaker_configs_file
            ), f"Weaker config file {weaker_configs_file} does not exist"
            env_configurations = np.load(weaker_configs_file, allow_pickle=True)[
                "env_configs"
            ]
            assert (
                len(env_configurations) >= args.num_episodes
            ), f"The number of configurations for the weaker test generator {len(env_configurations)} cannot be less than the number of episodes {args.num_episodes}"

        if args.training_type == TrainingType.mutant:
            subj_path, _ = create_results_folders(algo=args.algo, env=args.env_name)
            killable_operators, _, triviality_report = check_killability_triviality(
                subj_path=subj_path,
                do_triviality=True,
                algo=args.algo,
                env=args.env_name,
                num_runs=len(original_runs_folders),
                threshold_failure=args.threshold_failure,
                bootstrap=False
            )

            mutant_in_killable_operators = any(
                filter(
                    lambda killable_operator: killable_operator["name"]
                    == args.mutant_name,
                    killable_operators,
                )
            )

            filter_trivial_mutants = filter(
                lambda item: item[0] == args.mutant_name and item[2],
                triviality_report,
            )
            trivial_confs = list(map(lambda item: item[1], filter_trivial_mutants))
            mutants_folders = list(
                filter(
                    lambda mutant_folder: mutant_folder.split("_")[-1]
                    not in trivial_confs,
                    mutants_folders,
                )
            )

            if len(killable_operators) > 0 and mutant_in_killable_operators:
                if args.parallelize:
                    print(
                        f"****** Testing original and mutant models using the {args.mode} test generator ******"
                    )
                    print("****** Testing the original agent ******")
                else:
                    logger.info(
                        f"****** Testing original and mutant models using the {args.mode} test generator ******"
                    )
                    logger.info("****** Testing the original agent ******")

                if args.parallelize:
                    # FIXME: with parallelization, environment configurations are generated at every execution. Assuming that generating configurations
                    # is not expensive
                    start_time = time.perf_counter()
                    with Parallel(
                        n_jobs=num_cpus, batch_size="auto", backend="multiprocessing"
                    ) as parallel:
                        test_configurator_instance = TestConfigurator(
                            **test_configurator_args
                        )
                        test_agent_instance = TestAgent(
                            test_configurator=test_configurator_instance, all_args=args
                        )
                        res = parallel(
                            (
                                delayed(run_tests_on_agent)(
                                    mode=args.mode,
                                    num_run=i,
                                    threshold_failure=args.threshold_failure,
                                    run_folder=original_run_folder,
                                    random_seed=args.seed,
                                    test_configurator=test_configurator_instance,
                                    test_agent=test_agent_instance,
                                    training_type=TrainingType.original,
                                    mutant_name=args.mutant_name,
                                    parallelize=True,
                                    encode_run_and_conf=encode_run_and_conf,
                                    env_configurations=env_configurations,
                                    log=logger,
                                )
                                for i, original_run_folder in enumerate(
                                    original_runs_folders
                                )
                            ),
                        )
                    print(
                        f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                    )
                else:
                    start_time = time.perf_counter()
                    for i, original_run_folder in enumerate(original_runs_folders):

                        test_configurator_instance = TestConfigurator(
                            **test_configurator_args
                        )
                        test_agent_instance = TestAgent(
                            test_configurator=test_configurator_instance, all_args=args
                        )
                        run_tests_on_agent(
                            mode=args.mode,
                            num_run=i,
                            threshold_failure=args.threshold_failure,
                            run_folder=original_run_folder,
                            random_seed=args.seed,
                            test_configurator=test_configurator_instance,
                            test_agent=test_agent_instance,
                            training_type=TrainingType.original,
                            mutant_name=args.mutant_name,
                            encode_run_and_conf=encode_run_and_conf,
                            env_configurations=env_configurations,
                            log=logger,
                        )

                    logger.info(
                        f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                    )

                for mutant_folder in mutants_folders:
                    if args.parallelize:
                        print(f"****** Testing mutant in folder {mutant_folder} ******")
                    else:
                        logger.info(
                            f"****** Testing mutant in folder {mutant_folder} ******"
                        )

                    mutant_runs_folders = sorted(
                        glob.glob(os.path.join(mutant_folder, "run_*")),
                        key=lambda filepath: int(filepath.split("_")[-1]),
                    )

                    if args.parallelize:
                        # FIXME: with parallelization, environment configurations are generated at every execution
                        start_time = time.perf_counter()
                        with Parallel(
                            n_jobs=num_cpus,
                            batch_size="auto",
                            backend="multiprocessing",
                        ) as parallel:
                            test_configurator_instance = TestConfigurator(
                                **test_configurator_args
                            )
                            test_agent_instance = TestAgent(
                                test_configurator=test_configurator_instance,
                                all_args=args,
                            )
                            res = parallel(
                                (
                                    delayed(run_tests_on_agent)(
                                        mode=args.mode,
                                        num_run=i,
                                        threshold_failure=args.threshold_failure,
                                        run_folder=mutant_run_folder,
                                        random_seed=args.seed,
                                        test_configurator=test_configurator_instance,
                                        test_agent=test_agent_instance,
                                        training_type=TrainingType.mutant,
                                        mutant_name=args.mutant_name,
                                        parallelize=True,
                                        encode_run_and_conf=encode_run_and_conf,
                                        env_configurations=env_configurations,
                                        log=logger,
                                    )
                                    for i, mutant_run_folder in enumerate(
                                        mutant_runs_folders
                                    )
                                ),
                            )
                        print(
                            f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                        )
                    else:
                        start_time = time.perf_counter()

                        for i, mutant_run_folder in enumerate(mutant_runs_folders):

                            test_configurator_instance = TestConfigurator(
                                **test_configurator_args
                            )
                            test_agent_instance = TestAgent(
                                test_configurator=test_configurator_instance,
                                all_args=args,
                            )
                            run_tests_on_agent(
                                mode=args.mode,
                                num_run=i,
                                threshold_failure=args.threshold_failure,
                                run_folder=mutant_run_folder,
                                random_seed=args.seed,
                                test_configurator=test_configurator_instance,
                                test_agent=test_agent_instance,
                                training_type=TrainingType.mutant,
                                mutant_name=args.mutant_name,
                                encode_run_and_conf=encode_run_and_conf,
                                env_configurations=env_configurations,
                                log=logger,
                            )
                        logger.info(
                            f"Time elapsed {args.mode}: {time.perf_counter() - start_time}s"
                        )
            else:
                logger.info(f"Mutant {args.mutant_name} is not killable")
