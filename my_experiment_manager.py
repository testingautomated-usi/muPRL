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
import copy
import os
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, cast

import gym
import json
import numpy as np
import yaml
from callbacks.my_eval_callback import MyEvalCallback
from mutants.mutant import Mutant
from log import Log
from mutants.utils import load_mutant
from rl_zoo3.callbacks import SaveVecNormalizeCallback
from rl_zoo3.exp_manager import ExperimentManager
from sb3_contrib.common.vec_env import AsyncEval

# For using HER with GoalEnv
from stable_baselines3 import HerReplayBuffer  # noqa: F401
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
    is_vecenv_wrapped,
)
from test_generation.utils.custom_dummy_vec_env import CustomDummyVecEnv
from test_generation.utils.env_utils import ALGOS
from rl_zoo3.utils import get_callback_list, get_wrapper_class

# For custom activation fn
import flax.linen as nn  # noqa: F401


class MyExperimentManager(ExperimentManager):
    def __init__(
        self,
        args: argparse.Namespace,
        algo: str,
        env_id: str,
        log_folder: str,
        log_success: bool = False,
        test_generation: bool = False,
        eval_env: bool = False,
        mutant_name: str = None,
        **kwargs,
    ):
        super().__init__(
            args=args, algo=algo, env_id=env_id, log_folder=log_folder, **kwargs
        )
        self.args = args
        self.kwargs = kwargs
        self.env_id = env_id
        self.log_success = log_success
        self.env = None
        self.eval_env = eval_env
        self.mutant_name = mutant_name
        self.mutant = None
        self.logger = Log("MyExperimentManager")
        self.test_generation = test_generation
        if test_generation:
            self.vec_env_class = CustomDummyVecEnv

        hyperparams, self.saved_hyperparams = self.read_hyperparameters(
            log_hyperparameters=False
        )
        if hyperparams.get("n_envs") is not None:
            assert hyperparams.get("n_envs") == 1, "n_envs > 1 not supported yet"

        if self.mutant_name is not None:
            # FIXME: if the following happens it might mean that
            #  1. the hyperparameter was not specified by the developer (in that case we should look for it by
            #  analyzing the list of parameters of the RL algo);
            #  2. it might have a different name (in that case we should create a mapping);
            #  In all the other cases the mutation operator is not implementable.
            assert (
                self.mutant_name in hyperparams
            ), f"Mutant name {self.mutant_name} not in the list of hyperparameters {self.yaml_file}. Check whether the hyperparameter is specific for {self.algo} and it is not listed or it does not apply to {self.algo}."
            operator_value = hyperparams[self.mutant_name]

            kwargs = {}

            # FIXME: refactor in relative_operators list; replace mutants names with global constants
            if self.mutant_name in [
                "target_update_interval",
                "n_timesteps",
                "learning_starts",
            ]:
                relative = True
            else:
                relative = False

            # FIXME: replace mutants names with global constant
            if self.mutant_name in [
                "learning_starts",
                "target_update_interval",
                "n_steps",
            ]:
                assert (
                    hyperparams.get("n_timesteps", None) is not None
                ), f"n_timesteps needs to be specified in {self.yaml_file} for {['learning_starts', 'target_update_interval_mutant']} mutants to work"
                kwargs = {"n_timesteps": int(hyperparams["n_timesteps"])}

            self.mutant = cast(
                Mutant,
                load_mutant(mutant_name=self.mutant_name)(
                    operator_value=operator_value, relative=relative, **kwargs
                ),
            )

    @staticmethod
    def read_hyperparameters_static(
        yaml_filepath: str,
        env_name: str,
        algo: str,
    ) -> Dict[str, Any]:
        yaml_file = os.path.join(yaml_filepath, f"{algo}.yml")
        assert os.path.exists(yaml_file), f"{yaml_file} does not exist"
        with open(yaml_file) as f:
            hyperparams_dict = yaml.safe_load(f)
            if env_name in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_name]
            else:
                raise ValueError(f"Hyperparameters not found for {algo}-{env_name}")

        return hyperparams

    def read_hyperparameters(
        self, log_hyperparameters: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load hyperparameters from yaml file
        if log_hyperparameters:
            print(f"Loading hyperparameters from: {self.yaml_file}")
        with open(self.yaml_file) as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_name.gym_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_name.gym_id]
            elif self._is_atari:
                hyperparams = hyperparams_dict["atari"]
            else:
                raise ValueError(
                    f"Hyperparameters not found for {self.algo}-{self.env_name.gym_id}"
                )

        if self.custom_hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict(
            [(key, hyperparams[key]) for key in sorted(hyperparams.keys())]
        )

        # Always print used hyperparameters
        if log_hyperparameters:
            print(
                "Default hyperparameters for environment (ones being tuned will be overridden):"
            )
            pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any], Optional[Callable], List[BaseCallback], Optional[Callable]
    ]:
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        # Convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # Pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # Should we overwrite the number of timesteps?
        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        # Derive n_evaluations from number of timesteps if needed
        if self.n_evaluations is None and self.optimize_hyperparameters:
            self.n_evaluations = max(1, self.n_timesteps // int(1e5))
            print(
                f"Doing {self.n_evaluations} intermediate evaluations for pruning based on the number of timesteps."
                " (1 evaluation every 100k timesteps)"
            )

        # Pre-process normalize config
        hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy/buffer keyword arguments
        # Convert to python object if needed
        for kwargs_key in {
            "policy_kwargs",
            "replay_buffer_class",
            "replay_buffer_kwargs",
        }:
            if kwargs_key in hyperparams.keys() and isinstance(
                hyperparams[kwargs_key], str
            ):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        # Delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        # Same for VecEnvWrapper
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        if "vec_env_wrapper" in hyperparams.keys():
            del hyperparams["vec_env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks, vec_env_wrapper

    def setup_experiment(
        self, parallelize: bool = False
    ) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters(
            log_hyperparameters=False
        )
        if hyperparams.get("n_envs") is not None:
            assert hyperparams.get("n_envs") == 1, "n_envs > 1 not supported yet"
        (
            hyperparams,
            self.env_wrapper,
            self.callbacks,
            self.vec_env_wrapper,
        ) = self._preprocess_hyperparams(hyperparams)

        # FIXME: assuming the same things as above
        if self.mutant is not None:
            # FIXME: replace mutant name with global constant
            if self.mutant_name == "n_timesteps":
                self.n_timesteps = int(self.mutant.operator_value)
            else:
                hyperparams[self.mutant_name] = self.mutant.operator_value

        self.create_log_folder()
        self.create_callbacks(parallelize=parallelize)

        # Create env to have access to action space for action noise
        n_envs = (
            1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        )
        self.env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(
            hyperparams, saved_hyperparams, self.env
        )

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, self.env)
        elif self.optimize_hyperparameters:
            self.env.close()
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def create_callbacks(self, parallelize: bool = False):
        if self.show_progress and not parallelize:
            self.callbacks.append(ProgressBarCallback())

        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if (self.eval_freq > 0 or self.eval_env) and not self.optimize_hyperparameters:
            if self.eval_freq > 0:
                # Account for the number of parallel environments
                self.eval_freq = max(self.eval_freq // self.n_envs, 1)
            elif self.eval_env:
                self.eval_freq = int(self.n_timesteps * 0.1)
                self.logger.info(
                    f"Evaluating the agent every "
                    f"{self.eval_freq} timesteps for {self.n_eval_episodes} episodes"
                )

            if self.verbose > 0:
                self.logger.info("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(
                save_freq=1, save_path=self.params_path
            )
            eval_callback = MyEvalCallback(
                eval_env=self.create_envs(n_envs=self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
                parallelize=parallelize,
            )

            self.callbacks.append(eval_callback)

    def create_envs(
        self, n_envs: int, eval_env: bool = False, no_log: bool = False
    ) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        monitor_kwargs = {}
        # Special case for GoalEnvs: log success rate too
        if (
            "Neck" in self.env_name.gym_id
            or self.is_robotics_env(self.env_name.gym_id)
            or "parking-v0" in self.env_name.gym_id
            or self.log_success
        ):
            monitor_kwargs = dict(info_keywords=("is_success",))

        env_kwargs = copy.deepcopy(self.env_kwargs)
        env_kwargs["eval_env"] = eval_env

        env_kwargs["disable_env_checker"] = True

        # On most env, SubprocVecEnv does not help and is quite memory hungry
        # therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_name.gym_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=monitor_kwargs,
        )

        if self.vec_env_wrapper is not None:
            env = self.vec_env_wrapper(env)

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space)
                        and not is_image_space_channels_first(space)
                    )
            else:
                wrap_with_vectranspose = is_image_space(
                    env.observation_space
                ) and not is_image_space_channels_first(env.observation_space)

            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def learn_mock(self, success_probability: float = 1.0) -> None:

        self.create_log_folder()

        n = 100
        success_rate = np.random.binomial(n=n, p=success_probability) / 100

        json_string = json.dumps(
            {
                "n_calls": self.n_timesteps,
                "n_timesteps": self.n_timesteps,
                "num_episodes": self.n_eval_episodes,
                "success_rate": success_rate,
                "success_probability": success_probability,
                "total_timesteps": self.n_timesteps,
                "mock": True,
            },
            indent=4,
        )

        with open(
            os.path.join(self.save_path, "best_model_eval.json"), "w+", encoding="utf-8"
        ) as f:
            f.write(json_string)

    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        # Special case for ARS
        if self.algo == "ars" and self.n_envs > 1:
            kwargs["async_eval"] = AsyncEval(
                [
                    lambda: self.create_envs(n_envs=1, no_log=True)
                    for _ in range(self.n_envs)
                ],
                model.policy,
            )

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Clean progress bar (add check that the first callback is the progress bar callback)
            if len(self.callbacks) > 0 and isinstance(
                self.callbacks[0], ProgressBarCallback
            ):
                self.callbacks[0].on_training_end()
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass

    def __deepcopy__(self, memodict: Dict) -> "MyExperimentManager":
        new_my_experiment_manager = MyExperimentManager(
            args=self.args,
            algo=self.algo,
            env_id=self.env_id,
            log_folder=self.log_folder,
            log_success=self.log_success,
            test_generation=self.test_generation,
            eval_env=self.eval_env,
            mutant_name=self.mutant_name,
            **self.kwargs,
        )
        if self.mutant is not None:
            new_my_experiment_manager.mutant.operator_value = self.mutant.operator_value
        return new_my_experiment_manager
