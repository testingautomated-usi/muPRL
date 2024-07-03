"""
MIT License

Copyright (c) 2019 Antonin RAFFIN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import difflib
import importlib
import os
import time
import uuid

import gym
import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pytype: disable=import-error
from envs import register_env
from my_experiment_manager import MyExperimentManager
from randomness_utils import set_random_seed
from training_args import TrainingArgs


def train(exp_manager: MyExperimentManager, args: argparse.Namespace) -> None:
    # Going through custom gym packages to let them register in the global registry
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(
        gym.envs.registry.env_specs.keys()
    )  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{env_id} not found in gym registry, you maybe meant {closest_match}?"
        )

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    if args.track:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            )

        run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        run = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs{os.path.sep}{run_name}"

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            # we need to save the loaded hyperparameters
            args.saved_hyperparams = saved_hyperparams
            run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


if __name__ == "__main__":
    all_args = TrainingArgs().parse()

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if all_args.uuid else ""
    if all_args.seed < 0:
        # Seed but with a random one
        all_args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(seed=all_args.seed)

    exp_manager = MyExperimentManager(
        args=all_args,
        algo=all_args.algo,
        env_id=all_args.env,
        log_folder=all_args.log_folder,
        log_success=all_args.log_success,
        tensorboard_log=all_args.tensorboard_log,
        n_timesteps=all_args.n_timesteps,
        eval_freq=all_args.eval_freq,
        n_eval_episodes=all_args.eval_episodes,
        save_freq=all_args.save_freq,
        hyperparams=all_args.hyperparams,
        env_kwargs=all_args.env_kwargs,
        trained_agent=all_args.trained_agent,
        optimize_hyperparameters=all_args.optimize_hyperparameters,
        storage=all_args.storage,
        study_name=all_args.study_name,
        n_trials=all_args.n_trials,
        max_total_trials=all_args.max_total_trials,
        n_jobs=all_args.n_jobs,
        sampler=all_args.sampler,
        pruner=all_args.pruner,
        optimization_log_path=all_args.optimization_log_path,
        n_startup_trials=all_args.n_startup_trials,
        n_evaluations=all_args.n_evaluations,
        truncate_last_trajectory=all_args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=all_args.seed,
        log_interval=all_args.log_interval,
        save_replay_buffer=all_args.save_replay_buffer,
        verbose=all_args.verbose,
        vec_env_type=all_args.vec_env,
        n_eval_envs=all_args.n_eval_envs,
        no_optim_plots=all_args.no_optim_plots,
        device=all_args.device,
        yaml_file=all_args.yaml_file,
        show_progress=all_args.progress,
        test_generation=all_args.test_generation,
    )

    if all_args.register_env:
        if all_args.custom_env_kwargs is None:
            all_kwargs = {}
        else:
            all_kwargs = all_args.custom_env_kwargs

        time_wrapper = False
        if all_args.wrapper_kwargs is not None:
            all_kwargs.update(all_args.wrapper_kwargs)
            if "timeout_steps" in all_kwargs.keys():
                time_wrapper = True

        register_env(
            env_name=all_args.env,
            seed=all_args.seed,
            training=True,
            time_wrapper=time_wrapper,
            failure_predictor_path=exp_manager.save_path,
            test_generation=all_args.test_generation,
            **all_kwargs,
        )

    train(exp_manager=exp_manager, args=all_args)
