"""
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


--------------------------- LICENSE FOR pix2pix --------------------------------
BSD License

For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

----------------------------- LICENSE FOR DCGAN --------------------------------
BSD License

For dcgan.torch software

Copyright (c) 2015, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name Facebook nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import importlib
from typing import Callable

import gym
from gym import register
from test_generation.test_generator import TestGenerator
from wrappers.time_wrapper import TimeWrapper


def find_env_using_name(env_name: str, test_generation: bool = False) -> gym.Env:
    """
    e.g., env_name = CartPole-v1 -> split by '-' and lower case
    Import the module "envs/[env_name]_env.py".
    It has to be a subclass of gym.Env.
    """

    env_name = env_name.split("-")[0]

    if env_name.lower() == "humanoid":
        env_filename = f"envs.{env_name.lower()}.{env_name.lower()}_env_wrapper"
    else:
        env_filename = f"envs.{env_name.lower()}.{env_name.lower()}_env"
        if test_generation:
            env_filename += "_wrapper"

    envlib = importlib.import_module(env_filename)

    if env_name.lower() == "humanoid":
        target_env_name = env_name.replace("_", "") + "EnvWrapper"
    else:
        target_env_name = env_name.replace("_", "") + "Env"
        if test_generation:
            target_env_name += "Wrapper"

    for name, cls in envlib.__dict__.items():
        if name.lower() == target_env_name.lower() and issubclass(cls, gym.Env):
            return cls

    raise RuntimeError(
        "In %s.py, there should be a subclass of gym.Env with class name that matches %s in lowercase."
        % (env_filename, target_env_name)
    )


def create_env(
    env_name: str,
    seed: int,
    training: bool,
    time_wrapper: bool = False,
    test_generation: bool = False,
    failure_predictor_path: str = None,
    parallelize: bool = False,
    **env_kwargs,
) -> Callable:
    """
    Create an environment given its name and its parameters.
    """

    env = find_env_using_name(env_name=env_name, test_generation=test_generation)

    def make_env():
        if time_wrapper:
            return TimeWrapper(env=env(**env_kwargs))
        return env(**env_kwargs)

    def make_env_with_test_generator(eval_env: bool = False):
        test_generator = TestGenerator(
            env_name=env_name,
            log_path=failure_predictor_path,
            storing_logs=training,
            seed=seed,
            evaluate_agent_during_training=eval_env,
            parallelize=parallelize,
        )

        env_kwargs["eval_env"] = eval_env

        return env(
            test_generator=test_generator, time_wrapper=time_wrapper, **env_kwargs
        )

    if test_generation:
        return make_env_with_test_generator

    return make_env


def register_env(
    env_name: str,
    seed: int,
    training: bool,
    time_wrapper: bool = False,
    test_generation: bool = False,
    failure_predictor_path: str = None,
    parallelize: bool = False,
    **env_kwargs,
) -> None:
    # overriding the env
    register(
        id=env_name,
        entry_point=create_env(
            env_name=env_name,
            seed=seed,
            training=training,
            time_wrapper=time_wrapper,
            test_generation=test_generation,
            failure_predictor_path=failure_predictor_path,
            parallelize=parallelize,
            **env_kwargs,
        ),
    )
