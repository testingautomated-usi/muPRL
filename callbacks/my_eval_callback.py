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

import json
import os
from typing import Optional, Union, cast

import gym
import numpy as np
from log import Log
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from test_generation.env_wrapper import EnvWrapper
from test_generation.test_generator import TestGenerator


class MyEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        parallelize: bool = False,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        # trick to extract the test_generator object from the environment and set its log_path

        eval_env_unwrapped = eval_env.unwrapped.envs[0]

        while not isinstance(eval_env_unwrapped, EnvWrapper):
            eval_env_unwrapped = eval_env_unwrapped.unwrapped

        env_unwrapped = cast(EnvWrapper, eval_env_unwrapped)
        test_generator = cast(TestGenerator, env_unwrapped.test_generator)
        test_generator.log_path = log_path

        self.parallelize = parallelize

        self.best_mean_reward = -np.inf
        self.best_success_rate = 0.0

        self.log = Log("MyEvalCallback")

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez_compressed(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1 and not self.parallelize:
                self.log.info(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                self.log.info(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                )
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1 and not self.parallelize:
                    self.log.info(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            mean_success_rate = np.mean(self._is_success_buffer)

            # the equal is to privilege the latest model given the same reward
            if mean_success_rate >= self.best_success_rate or (
                mean_success_rate == self.best_success_rate
                and mean_reward >= self.best_mean_reward
            ):
                self.best_success_rate = mean_success_rate
                self.best_mean_reward = mean_reward

                if self.verbose >= 1 and not self.parallelize:
                    self.log.info("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )

                    # just overwriting instead of updating if the file exists
                    # as I understand the difference between n_calls and num_timesteps is that n_calls refers to
                    # the current training, while num_timesteps is updated when the training of the model is
                    # resumed from a previous training run. In our case they should be equivalent.
                    json_string = json.dumps(
                        {
                            "n_calls": self.n_calls,
                            "num_timesteps": self.num_timesteps,
                            "num_episodes": self.n_eval_episodes,
                            "success_rate": self.best_success_rate,
                            "average_reward": self.best_mean_reward,
                            "average_length": mean_ep_length,
                            "total_timesteps": self.locals["total_timesteps"],
                        },
                        indent=4,
                    )

                    with open(
                        os.path.join(self.best_model_save_path, "best_model_eval.json"),
                        "w+",
                        encoding="utf-8",
                    ) as f:
                        f.write(json_string)

                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # this is to track the experiment when parallelize = True
            json_string = json.dumps(
                {
                    "n_calls": self.n_calls,
                    "num_timesteps": self.num_timesteps,
                    "total_timesteps": self.locals["total_timesteps"],
                },
                indent=4,
            )

            with open(
                os.path.join(self.best_model_save_path, "track_eval.json"),
                "w+",
                encoding="utf-8",
            ) as f:
                f.write(json_string)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def on_training_end(self) -> None:
        self.eval_env.close()
