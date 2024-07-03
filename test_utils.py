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

import os
import sys
from typing import Optional, Tuple, Union

import torch as th
from envs import register_env
from huggingface_sb3 import EnvironmentName
from log import Log
from randomness_utils import set_random_seed
from rl_zoo3.utils import get_model_path
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from test_generation.utils.env_utils import ALGOS, create_test_env


def get_trained_model(
    exp_id: int,
    env_name: str,
    folder: str,
    algo: str,
    seed: int,
    device: Union[th.device, str] = "auto",
    test_generation: bool = False,
    load_best: bool = False,
    load_checkpoint: Optional[str] = None,
    load_last_checkpoint: bool = False,
    norm_reward: bool = False,
    render: bool = False,
    register_environment: bool = False,
    time_wrapper: bool = False,
    folder_with_agent: str = None,
    reset_random_gen: bool = False,
    parallelize: bool = False,
    **env_kwargs,
) -> Tuple[VecEnv, BaseAlgorithm, str]:
    logger = Log("get_trained_model")
    n_envs = 1

    env_name = EnvironmentName(gym_id=env_name)

    if folder_with_agent is not None:
        assert not load_checkpoint, "Load checkpoint not supported yet"
        assert not load_last_checkpoint, "Load last checkpoint not supported yet"

        log_path = folder_with_agent
        model_path = os.path.join(log_path, f"{env_name}.zip")

        if not os.path.exists(model_path):
            logger.warn(
                f"Model named {env_name}.zip does not exist in {log_path}. Trying with best_model.zip"
            )
            model_path = os.path.join(log_path, "best_model.zip")

        assert os.path.exists(model_path), f"{model_path} does not exist"

    else:
        _, model_path, log_path = get_model_path(
            exp_id=exp_id,
            folder=folder,
            algo=algo,
            env_name=env_name,
            load_best=load_best,
            load_checkpoint=load_checkpoint,
            load_last_checkpoint=load_last_checkpoint,
        )

    logger.info(f"Loading {model_path}")
    logger.info(f"Log path {log_path}")

    set_random_seed(seed=seed, reset_random_gen=reset_random_gen)

    if register_environment:
        register_env(
            env_name=env_name,
            seed=seed,
            training=False,
            time_wrapper=time_wrapper,
            failure_predictor_path=log_path,
            test_generation=True,
            parallelize=parallelize,
            **env_kwargs,
        )

    env = create_test_env(
        env_id=env_name.gym_id,
        test_generation=test_generation,
        n_envs=n_envs,
        stats_path=None,
        seed=seed,
        should_render=render,
        hyperparams={},
    )

    kwargs = dict(seed=seed)
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if os.path.exists(os.path.join(log_path, "vecnormalize.pkl")):
        logger.info("Loading running average")
        path_vec_normalize = os.path.join(log_path, "vecnormalize.pkl")
        env = VecNormalize.load(path_vec_normalize, env)
        # Deactivate training and reward normalization
        env.training = False
        env.norm_reward = False

    return (
        env,
        ALGOS[algo].load(
            model_path, device=device, env=env, custom_objects=custom_objects, **kwargs
        ),
        log_path,
    )
