"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
from gym import spaces
from sbx import PPO
from stable_baselines3.common.utils import explained_variance
from test_generation.env_wrapper import EnvWrapper
from test_generation.utils.custom_dummy_vec_env import CustomDummyVecEnv


class PPOWrapper(PPO):
    def __init__(self, **kwargs):
        super(PPOWrapper, self).__init__(**kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # train for n_epochs epochs
        for _ in range(self.n_epochs):
            # JIT only one update
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().numpy().astype(np.int32)
                else:
                    actions = rollout_data.actions.numpy()

                (self.policy.actor_state, self.policy.vf_state), (
                    value_loss,
                    pg_loss,
                ) = self._one_update(
                    self.actor,
                    self.vf,
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    observations=rollout_data.observations.numpy(),
                    actions=actions,
                    advantages=rollout_data.advantages.numpy(),
                    returns=rollout_data.returns.numpy(),
                    old_log_prob=rollout_data.old_log_prob.numpy(),
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        agent_state = dict()
        agent_state["training_progress"] = (
            self.num_timesteps / self._total_timesteps
        ) * 100
        # FIXME
        agent_state["ent_coef"] = value_loss.item()

        env_unwrapped = self.get_env().unwrapped
        while not isinstance(env_unwrapped, CustomDummyVecEnv):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped = env_unwrapped.envs[0].env
        while not isinstance(env_unwrapped, EnvWrapper):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped.send_agent_state(agent_state=agent_state)

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # TODO: use mean instead of one point
        self.logger.record("train/value_loss", value_loss.item())
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)
