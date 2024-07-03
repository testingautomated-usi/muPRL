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

import flax
import jax.numpy as jnp
from sbx.common.type_aliases import ReplayBufferSamplesNp
from test_generation.env_wrapper import EnvWrapper
from test_generation.utils.custom_dummy_vec_env import CustomDummyVecEnv

from algos.sac.sac import SAC


class SACWrapper(SAC):
    def __init__(self, **kwargs):
        super(SACWrapper, self).__init__(**kwargs)

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(
            batch_size * gradient_steps, env=self._vec_normalize_env
        )
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {
            i: True
            for i in range(gradient_steps)
            if ((self._n_updates + i + 1) % self.policy_delay) == 0
        }
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = jnp.concatenate(
                [data.observations[key].numpy() for key in keys], axis=1
            )
            next_obs = jnp.concatenate(
                [data.next_observations[key].numpy() for key in keys], axis=1
            )
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            (actor_loss_value, qf_loss_value, ent_coef_value),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
        )
        self._n_updates += gradient_steps

        agent_state = dict()
        agent_state["training_progress"] = (
            self.num_timesteps / self._total_timesteps
        ) * 100
        agent_state["ent_coef"] = ent_coef_value.item()

        env_unwrapped = self.get_env().unwrapped
        while not isinstance(env_unwrapped, CustomDummyVecEnv):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped = env_unwrapped.envs[0].env
        while not isinstance(env_unwrapped, EnvWrapper):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped.send_agent_state(agent_state=agent_state)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())
