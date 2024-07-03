import argparse
import copy
import difflib
import glob
import importlib
import json
import logging
import multiprocessing
import os
import platform
import time
from typing import Dict, List, Tuple, cast

import gym

gym.logger.set_level(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pytype: disable=import-error

from envs import register_env
from joblib import Parallel, delayed
from log import Log
from my_experiment_manager import MyExperimentManager
from randomness_utils import set_random_seed
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS
from test_generation.env_wrapper import EnvWrapper
from test_generation.test_generator import TestGenerator
from training.monitor_utils import X_SUCCESS, ts2xy
from training.training_type import TrainingType
from training_args import TrainingArgs


def read_num_eval_episodes_from_eval_file(folder_name: str) -> int:
    assert os.path.exists(
        os.path.join(folder_name, "best_model_eval.json")
    ), "Filename {} not found".format(os.path.join(folder_name, "best_model_eval.json"))
    with open(
        os.path.join(folder_name, "best_model_eval.json"), "r+", encoding="utf-8"
    ) as f:
        return float(json.load(f)["num_episodes"])


def read_success_rate_from_eval_file(folder_name: str) -> float:
    assert os.path.exists(
        os.path.join(folder_name, "best_model_eval.json")
    ), "Filename {} not found".format(os.path.join(folder_name, "best_model_eval.json"))
    with open(
        os.path.join(folder_name, "best_model_eval.json"), "r+", encoding="utf-8"
    ) as f:
        return float(json.load(f)["success_rate"])


class TrainAgent:
    def __init__(self, training_type: TrainingType, args: argparse.Namespace):
        self.args = args
        self.import_modules = True
        self.training_type = training_type
        self.logger = Log("train_agent")

        if self.args.register_env and not self.args.parallelize:
            if self.args.custom_env_kwargs is None:
                all_kwargs = {}
            else:
                all_kwargs = self.args.custom_env_kwargs

            time_wrapper = False
            if self.args.wrapper_kwargs is not None:
                all_kwargs.update(self.args.wrapper_kwargs)
                if "timeout_steps" in all_kwargs.keys():
                    time_wrapper = True

            register_env(
                env_name=self.args.env,
                seed=self.args.seed,
                training=True,
                time_wrapper=time_wrapper,
                failure_predictor_path=my_exp_manager.save_path,
                test_generation=self.args.test_generation,
                parallelize=args.parallelize,
                **all_kwargs,
            )

        self.training_statistics = {
            "folders_names": [],
            "nums_episodes": [],
            "rewards": [],
            "episodes_lengths": [],
            "success_rates": [],
        }

    @staticmethod
    def compute_adjusted_rolling_average(
        metrics: np.ndarray, window_size: int = 100
    ) -> np.ndarray:
        averages = []
        for i in range(len(metrics) - window_size + 1):
            window = metrics[i : i + window_size]
            average = sum(window) / window_size
            averages.append(average)

        # FIXME, probably this part is wrong
        length_averages_list = len(averages)
        for i in range(len(metrics) - length_averages_list - 1):
            list_to_consider = metrics[-(len(metrics) - length_averages_list - i) : -1]
            assert (
                len(list_to_consider) > 0
            ), "Error when computing adjusted rolling average"
            averages.append(sum(list_to_consider) / len(list_to_consider))

        return np.asarray(averages)

    @staticmethod
    def plot_training_statistics(
        folder_name: str,
        num_episodes: np.ndarray,
        metrics_list: np.ndarray,
        metric_name: str,
    ) -> None:
        assert metric_name in [
            "reward",
            "episode_length",
            "success_rate",
        ], f"Unknown metric name {metric_name}"

        # Calculate the average and standard deviation of the metric at each step
        average_metric = np.mean(metrics_list, axis=0)
        std_dev_metric = np.std(metrics_list, axis=0)

        # Plot the average metric with associated standard deviation
        plt.figure(figsize=(10, 6))
        plt.plot(num_episodes, average_metric, label=f"Average {metric_name}")
        plt.fill_between(
            num_episodes,
            average_metric - std_dev_metric,
            average_metric + std_dev_metric,
            alpha=0.3,
        )
        plt.xlabel("Num episodes")
        plt.ylabel(f"Average {metric_name}")
        plt.title(f"Average {metric_name} with Standard Deviation")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder_name}/{metric_name}.png", format="png")
        plt.close()

    def training_run(
        self,
        exp_manager: MyExperimentManager,
        seeds: List[int],
        num_run: int,
        folder_name: str,
        tensorboard_folder_name: str,
        parallelize: bool = False,
        mock: bool = False,
        success_probability: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if seeds is not None:
            seed = seeds[num_run]
        else:
            # generate seed randomly
            try:
                # in some machines, depending on the architecture, 2^32 might overflow
                seed = np.random.randint(2**32 - 1, dtype="int64").item()
            except ValueError as _:
                seed = np.random.randint(2**30 - 1, dtype="int64").item()

        if not parallelize and not mock:
            self.logger.info(f"Run {num_run}")

        folder_name_run = f"{folder_name}{os.path.sep}run_{num_run}"
        start_time_single_run = time.perf_counter()

        x_rewards, y_rewards, y_ep_lengths, y_success_rates = self.train(
            exp_manager=exp_manager,
            seed=seed,
            folder_name=folder_name_run,
            tensorboard_folder_name=tensorboard_folder_name,
            args_training=self.args,
            parallelize=parallelize,
            mock=mock,
            success_probability=success_probability,
        )

        time_elapsed_s_single_run = time.perf_counter() - start_time_single_run

        if not parallelize and not mock:
            self.logger.info(
                f"Run {num_run} results. Avg reward: {np.mean(y_rewards)}, "
                f"Avg length episode: {np.mean(y_ep_lengths)}, "
                f"Avg success rate: {np.mean(y_success_rates)}, "
                f"Time elapsed (s): {time_elapsed_s_single_run}"
            )

        return x_rewards, y_rewards, y_ep_lengths, y_success_rates

    def train_multiple_times(
        self,
        exp_manager: MyExperimentManager,
        num: int,
        seeds: List[int] = None,
        num_cpus: int = -1,
        parallelize: bool = False,
        mock: bool = False,
        success_probability: float = 1.0,
        run_num: int = -1,
    ) -> str:
        if seeds is not None:
            assert num == len(seeds), (
                f"The number of times the agent should be trained {num} "
                f"should be the same as the number of seeds {len(seeds)}"
            )

        folder_name = (
            f"{exp_manager.log_folder}{os.path.sep}{exp_manager.algo}"
            f"{os.path.sep}{exp_manager.env_name}_{self.training_type.name}"
        )

        if exp_manager.mutant is not None:
            folder_name += f"_{exp_manager.mutant.operator_name}_{exp_manager.mutant.operator_value}"

        if self.args.tensorboard_log is not None and self.args.tensorboard_log != "":
            tensorboard_folder_name = (
                f"{self.args.tensorboard_log}"
                f"{os.path.sep}{exp_manager.algo}{os.path.sep}{exp_manager.env_name}_{self.training_type.name}"
            )
        else:
            tensorboard_folder_name = None

        if exp_manager.mutant is not None and tensorboard_folder_name is not None:
            tensorboard_folder_name += f"_{exp_manager.mutant.operator_name}_{exp_manager.mutant.operator_value}"

        if run_num > -1:
            os.makedirs(name=folder_name, exist_ok=True)
        else:
            os.makedirs(name=folder_name)

        if not parallelize:
            # apparently, nothing can be written in the 'folder_name' folder, otherwise, the second run of training
            # (i.e., when training mutants) gets stuck
            logging.basicConfig(
                filename=os.path.join(folder_name, "log.txt"),
                filemode="w",
                level=logging.DEBUG,
            )

        num_episodes_list = []
        rewards_list = []
        episode_lengths_list = []
        success_rates_list = []

        start_time = time.perf_counter()

        if not parallelize:

            if run_num > -1:

                assert (
                    0 <= run_num < num
                ), f"Run number {run_num} cannot be >= number of runs {num} or < 0"
                (
                    x_rewards,
                    y_rewards,
                    y_ep_lengths,
                    y_success_rates,
                ) = self.training_run(
                    exp_manager=exp_manager,
                    seeds=seeds,
                    num_run=run_num,
                    folder_name=folder_name,
                    tensorboard_folder_name=tensorboard_folder_name,
                    mock=mock,
                    success_probability=success_probability,
                )

            else:

                for num_run in range(num):
                    (
                        x_rewards,
                        y_rewards,
                        y_ep_lengths,
                        y_success_rates,
                    ) = self.training_run(
                        exp_manager=exp_manager,
                        seeds=seeds,
                        num_run=num_run,
                        folder_name=folder_name,
                        tensorboard_folder_name=tensorboard_folder_name,
                        mock=mock,
                        success_probability=success_probability,
                    )

                    num_episodes_list.append(x_rewards)
                    rewards_list.append(y_rewards)
                    episode_lengths_list.append(y_ep_lengths)
                    success_rates_list.append(y_success_rates)

        else:
            if seeds is None:
                random_seeds = []
                for num_run in range(num):
                    # generate seed randomly
                    try:
                        # in some machines, depending on the architecture, 2^32 might overflow
                        seed = np.random.randint(2**32 - 1, dtype="int64").item()
                    except ValueError as _:
                        seed = np.random.randint(2**30 - 1, dtype="int64").item()
                    random_seeds.append(seed)
            else:
                random_seeds = seeds

            if num_cpus == -1:
                num_cpus = multiprocessing.cpu_count()
            else:
                assert (
                    num_cpus <= multiprocessing.cpu_count()
                ), f"Num cpus {num_cpus} cannot be > than the number of logical cores in the current machine {multiprocessing.cpu_count()}"

            with Parallel(
                n_jobs=num_cpus, batch_size="auto", backend="loky"
            ) as parallel:
                res = parallel(
                    (
                        delayed(self.training_run)(
                            exp_manager=copy.deepcopy(exp_manager),
                            seeds=random_seeds,
                            num_run=num_run,
                            folder_name=folder_name,
                            tensorboard_folder_name=tensorboard_folder_name,
                            parallelize=True,
                        )
                        for num_run in range(num)
                    ),
                )

            for x_rewards, y_rewards, y_ep_lengths, y_success_rates in res:
                num_episodes_list.append(x_rewards)
                rewards_list.append(y_rewards)
                episode_lengths_list.append(y_ep_lengths)
                success_rates_list.append(y_success_rates)

        time_elapsed_s = round(time.perf_counter() - start_time, 2)
        if not mock:
            self.logger.info(f"Time elapsed (s): {time_elapsed_s}")

        num_episodes_list.append(x_rewards)
        rewards_list.append(y_rewards)
        episode_lengths_list.append(y_ep_lengths)
        success_rates_list.append(y_success_rates)

        max_num_episodes = max(
            [len(episode_list) for episode_list in num_episodes_list]
        )
        # padding each metric
        rewards_list_padded = []
        for rewards in rewards_list:
            rewards_list_padded.append(
                list(rewards)
                + [rewards[-1] for _ in range(max_num_episodes - len(rewards))]
            )
        episode_lengths_list_padded = []
        for episode_lengths in episode_lengths_list:
            episode_lengths_list_padded.append(
                list(episode_lengths)
                + [
                    episode_lengths[-1]
                    for _ in range(max_num_episodes - len(episode_lengths))
                ]
            )
        success_rates_list_padded = []
        for success_rates in success_rates_list:
            success_rates_list_padded.append(
                list(success_rates)
                + [
                    success_rates[-1]
                    for _ in range(max_num_episodes - len(success_rates))
                ]
            )

        rewards_list_padded = np.asarray(rewards_list_padded)
        episode_lengths_list_padded = np.asarray(episode_lengths_list_padded)
        success_rates_list_padded = np.asarray(success_rates_list_padded)

        num_episodes = np.arange(start=0, stop=max_num_episodes)

        if not parallelize:
            # apparently, nothing can be written in the 'folder_name' folder, otherwise, the second run of training
            # gets stuck
            self.plot_training_statistics(
                folder_name=folder_name,
                num_episodes=num_episodes,
                metrics_list=rewards_list_padded,
                metric_name="reward",
            )
            self.plot_training_statistics(
                folder_name=folder_name,
                num_episodes=num_episodes,
                metrics_list=episode_lengths_list_padded,
                metric_name="episode_length",
            )
            self.plot_training_statistics(
                folder_name=folder_name,
                num_episodes=num_episodes,
                metrics_list=success_rates_list_padded,
                metric_name="success_rate",
            )
        else:
            self.training_statistics["folders_names"].append(folder_name)
            self.training_statistics["nums_episodes"].append(num_episodes)
            self.training_statistics["rewards"].append(rewards_list_padded)
            self.training_statistics["episodes_lengths"].append(
                episode_lengths_list_padded
            )
            self.training_statistics["success_rates"].append(success_rates_list_padded)

        return folder_name

    def plot_all_training_statistics(self) -> None:
        if "folders_names" in self.training_statistics:
            for num_run in range(len(self.training_statistics["folders_names"])):
                folder_name = self.training_statistics["folders_names"][num_run]
                num_episodes = self.training_statistics["nums_episodes"][num_run]
                rewards = self.training_statistics["rewards"][num_run]
                episode_lengths = self.training_statistics["episodes_lengths"][num_run]
                success_rates = self.training_statistics["success_rates"][num_run]
                self.plot_training_statistics(
                    folder_name=folder_name,
                    num_episodes=num_episodes,
                    metrics_list=rewards,
                    metric_name="reward",
                )
                self.plot_training_statistics(
                    folder_name=folder_name,
                    num_episodes=num_episodes,
                    metrics_list=episode_lengths,
                    metric_name="episode_length",
                )
                self.plot_training_statistics(
                    folder_name=folder_name,
                    num_episodes=num_episodes,
                    metrics_list=success_rates,
                    metric_name="success_rate",
                )

    def clear_all_training_statistics(self) -> None:
        self.training_statistics["folders_names"].clear()
        self.training_statistics["nums_episodes"].clear()
        self.training_statistics["rewards"].clear()
        self.training_statistics["episodes_lengths"].clear()
        self.training_statistics["success_rates"].clear()

    @staticmethod
    def train(
        exp_manager: MyExperimentManager,
        seed: int,
        folder_name: str,
        tensorboard_folder_name: str,
        args_training: Dict,
        parallelize: bool = False,
        mock: bool = False,
        success_probability: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if mock:
            exp_manager.params_path = folder_name
            exp_manager.save_path = folder_name
            exp_manager.tensorboard_log = tensorboard_folder_name
            exp_manager.seed = seed
            exp_manager.learn_mock(success_probability=success_probability)

            json_string = json.dumps(
                {
                    "seed": seed,
                    "mock": True,
                    "mutant": (
                        None
                        if exp_manager.mutant is None
                        else {
                            "mutant_name": exp_manager.mutant.operator_name,
                            "value": exp_manager.mutant.operator_value,
                        }
                    ),
                },
                indent=4,
            )

            with open(
                os.path.join(folder_name, "statistics.json"), "w+", encoding="utf-8"
            ) as f:
                f.write(json_string)

            return np.ones((10,)), np.ones((10,)), np.ones((10,)), np.ones((10,))

        if parallelize and args_training.register_env:
            if args_training.custom_env_kwargs is None:
                all_kwargs = {}
            else:
                all_kwargs = args_training.custom_env_kwargs

            time_wrapper = False
            if args_training.wrapper_kwargs is not None:
                all_kwargs.update(args_training.wrapper_kwargs)
                if "timeout_steps" in all_kwargs.keys():
                    time_wrapper = True

            register_env(
                env_name=args_training.env,
                seed=args_training.seed,
                training=True,
                time_wrapper=time_wrapper,
                failure_predictor_path=exp_manager.save_path,
                test_generation=args_training.test_generation,
                parallelize=parallelize,
                **all_kwargs,
            )

        start_time = time.perf_counter()

        set_random_seed(seed=seed, reset_random_gen=True)

        # create the folder here before calling the setup_experiment method
        exp_manager.params_path = folder_name
        exp_manager.save_path = folder_name
        exp_manager.tensorboard_log = tensorboard_folder_name
        exp_manager.seed = seed

        # Prepare experiment and launch hyperparameter optimization if needed. Creates the environment
        results = exp_manager.setup_experiment(parallelize=parallelize)
        assert (
            results is not None
        ), "Training an agent from scratch (see setup_experiment())"

        assert exp_manager.env is not None, "Environment should be instantiated"

        for env_instance in exp_manager.env.unwrapped.envs:
            env_unwrapped = env_instance
            while not isinstance(env_unwrapped, EnvWrapper):
                env_unwrapped = env_unwrapped.unwrapped

            # trick to extract the test_generator object from the environment and set its log_path
            env_unwrapped = cast(EnvWrapper, env_unwrapped)
            test_generator = cast(TestGenerator, env_unwrapped.test_generator)
            test_generator.log_path = folder_name

        model, saved_hyperparams = results

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            if not exp_manager.eval_env and exp_manager.eval_freq < 0:
                exp_manager.save_trained_model(model)

        x_rewards, y_rewards = ts2xy(
            data_frame=load_results(path=folder_name), x_axis=X_TIMESTEPS
        )
        _, y_ep_lengths = ts2xy(
            data_frame=load_results(path=folder_name), x_axis=X_EPISODES
        )
        _, y_success_rates = ts2xy(
            data_frame=load_results(path=folder_name), x_axis=X_SUCCESS
        )

        avg_reward = float(np.mean(y_rewards))
        avg_episode_length = float(np.mean(y_ep_lengths))
        avg_success_rate = float(np.mean(y_success_rates))

        # TODO: refactor with class
        json_string = json.dumps(
            {
                "seed": seed,
                "avg_reward": avg_reward,
                "avg_episode_length": avg_episode_length,
                "avg_success_rate": avg_success_rate,
                "mutant": (
                    None
                    if exp_manager.mutant is None
                    else {
                        "mutant_name": exp_manager.mutant.operator_name,
                        "value": exp_manager.mutant.operator_value,
                    }
                ),
                "time_elapsed_s": round(time.perf_counter() - start_time, 2),
            },
            indent=4,
        )
        with open(
            os.path.join(folder_name, "statistics.json"), "w+", encoding="utf-8"
        ) as f:
            f.write(json_string)

        return x_rewards, y_rewards, y_ep_lengths, y_success_rates


if __name__ == "__main__":
    all_args = TrainingArgs().parse()

    pltf = platform.system()
    if pltf.lower() == "windows" and all_args.parallelize:
        print("Disabling parallelization in Windows")
        all_args.parallelize = False

    my_experiment_manager_params = {
        "args": all_args,
        "algo": all_args.algo,
        "env_id": all_args.env,
        "log_folder": all_args.log_folder,
        "log_success": all_args.log_success,
        "tensorboard_log": all_args.tensorboard_log,
        "n_timesteps": all_args.n_timesteps,
        "eval_freq": all_args.eval_freq,
        "n_eval_episodes": all_args.eval_episodes,
        "save_freq": all_args.save_freq,
        "hyperparams": all_args.hyperparams,
        "env_kwargs": all_args.env_kwargs,
        "trained_agent": all_args.trained_agent,
        "optimize_hyperparameters": all_args.optimize_hyperparameters,
        "storage": all_args.storage,
        "study_name": all_args.study_name,
        "n_trials": all_args.n_trials,
        "max_total_trials": all_args.max_total_trials,
        "n_jobs": all_args.n_jobs,
        "sampler": all_args.sampler,
        "pruner": all_args.pruner,
        "optimization_log_path": all_args.optimization_log_path,
        "n_startup_trials": all_args.n_startup_trials,
        "n_evaluations": all_args.n_evaluations,
        "truncate_last_trajectory": all_args.truncate_last_trajectory,
        "uuid_str": "",
        "seed": all_args.seed,
        "log_interval": all_args.log_interval,
        "save_replay_buffer": all_args.save_replay_buffer,
        "verbose": 0,
        "vec_env_type": all_args.vec_env,
        "n_eval_envs": all_args.n_eval_envs,
        "no_optim_plots": all_args.no_optim_plots,
        "device": all_args.device,
        "yaml_file": all_args.yaml_file,
        "show_progress": all_args.progress,
        "test_generation": all_args.test_generation,
        "eval_env": all_args.eval_env,
        "mutant_name": all_args.mutant_name,
    }

    if all_args.seed > -1:
        set_random_seed(seed=all_args.seed)

    mock = all_args.mock
    success_probability = 1.0

    my_exp_manager = MyExperimentManager(**my_experiment_manager_params)

    all_args.training_type = TrainingType.parse(name_as_str=all_args.training_type)

    train_agent = TrainAgent(training_type=all_args.training_type, args=all_args)
    logger = Log("train_agent_main")

    for env_module in all_args.gym_packages:
        importlib.import_module(env_module)

    # env_id = self.args.env
    registered_envs = set(
        gym.envs.registry.env_specs.keys()
    )  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if all_args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(
                all_args.gym_env, registered_envs, n=1
            )[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{all_args.gym_env} not found in gym registry, you maybe meant {closest_match}?"
        )

    if all_args.training_type == TrainingType.mutant:
        # assuming that the original agent is already present; if not then raise error
        original_folder_name = f"{my_exp_manager.log_folder}{os.path.sep}{my_exp_manager.algo}{os.path.sep}{my_exp_manager.env_name}_original"
        assert os.path.exists(
            original_folder_name
        ), f"Original agent folder {original_folder_name} does not exist"
        original_runs_files = sorted(
            glob.glob(os.path.join(original_folder_name, "run_*")),
            key=lambda filepath: int(filepath.split("_")[-1]),
        )

        num_runs = len(original_runs_files)
        assert num_runs > 0, f"No runs found in {original_folder_name}"

        stored_seeds = []
        for original_run_file in original_runs_files:
            with open(
                os.path.join(original_run_file, "statistics.json"),
                "r+",
                encoding="utf-8",
            ) as f:
                stored_seeds.append(json.load(f)["seed"])

        logger.info(
            f"Overriding the argument --num-runs with {num_runs} as we are training a mutant."
        )

        assert (
            all_args.search_budget > 0
        ), "Search budget for mutation cannot be negative"

        operator_values = [my_exp_manager.mutant.operator_value]

        for i in range(all_args.search_budget):
            max_iterations = 1000
            new_operator_value = my_exp_manager.mutant.mutate()

            while new_operator_value in operator_values and max_iterations > 0:
                new_operator_value = my_exp_manager.mutant.mutate()
                max_iterations -= 1

            if max_iterations == 0:
                logger.warn(
                    f"All values for mutant {my_exp_manager.mutant_name} have likely been sampled {operator_values[1:]}. Stopping the search at iteration {i}/{all_args.search_budget}."
                )
                break

            max_iterations = 1000
            operator_values.append(new_operator_value)

        change_direction = False
        end_condition = (
            min(len(operator_values) - 1, all_args.search_budget)
            if len(operator_values) > 1
            else all_args.search_budget
        )

        for i in range(end_condition):
            if all_args.search_iteration > -1:
                assert (
                    all_args.search_iteration < all_args.search_budget
                ), f"Search iteration {all_args.search_iteration} cannot be equal or greater than search budget {all_args.search_budget}"
                operator_value = operator_values[1:][all_args.search_iteration]
                logger.info(f"Selecting operator configuration {operator_value}")
            else:
                operator_value = operator_values[1:][i]

            my_exp_manager.mutant.operator_value = operator_value
            logger.info(
                f"Generating new value for operator "
                f"{my_exp_manager.mutant.operator_name}: "
                f"{my_exp_manager.mutant.operator_value}"
            )

            operator_values.append(operator_value)

            folder_name = train_agent.train_multiple_times(
                exp_manager=my_exp_manager,
                num=num_runs,
                seeds=stored_seeds,
                num_cpus=all_args.num_cpus,
                parallelize=all_args.parallelize,
                run_num=all_args.run_num,
                mock=mock,
                success_probability=success_probability,
            )

            if all_args.parallelize:
                train_agent.plot_all_training_statistics()
                train_agent.clear_all_training_statistics()

            if all_args.search_iteration > -1:
                assert (
                    all_args.search_iteration < all_args.search_budget
                ), f"Search iteration {all_args.search_iteration} cannot be bigger than search budget {all_args.search_budget}"
                logger.info(
                    f"Stopping random search after executing iteration {all_args.search_iteration} "
                    f"and operator configuration {operator_value}"
                )
                break

    else:
        _ = train_agent.train_multiple_times(
            exp_manager=my_exp_manager,
            num=all_args.num_runs,
            num_cpus=all_args.num_cpus,
            parallelize=all_args.parallelize,
            run_num=all_args.run_num,
            mock=mock,
        )

        if all_args.parallelize:
            train_agent.plot_all_training_statistics()
