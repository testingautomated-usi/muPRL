"""
The MIT License

Copyright (c) 2022 Antonin Raffin

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

import jax
import jax.numpy as jnp
from sbx.common.type_aliases import ReplayBufferSamplesNp
from test_generation.env_wrapper import EnvWrapper
from test_generation.utils.custom_dummy_vec_env import CustomDummyVecEnv

from algos.dqn.dqn import DQN


class DQNWrapper(DQN):
    def __init__(self, **kwargs):
        super(DQNWrapper, self).__init__(**kwargs)

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(
            batch_size * gradient_steps, env=self._vec_normalize_env
        )
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            # Convert to int64
            data.actions.long().numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )
        # Pre compute the slice indices
        # otherwise jax will complain
        indices = jnp.arange(len(data.dones)).reshape(gradient_steps, batch_size)

        update_carry = {
            "qf_state": self.policy.qf_state,
            "gamma": self.gamma,
            "data": data,
            "indices": indices,
            "info": {
                "critic_loss": jnp.array([0.0]),
                "qf_mean_value": jnp.array([0.0]),
            },
        }

        # jit the loop similar to https://github.com/Howuhh/sac-n-jax
        # we use scan to be able to play with unroll parameter
        update_carry, _ = jax.lax.scan(
            self._train,
            update_carry,
            indices,
            unroll=1,
        )

        self.policy.qf_state = update_carry["qf_state"]
        qf_loss_value = update_carry["info"]["critic_loss"]
        qf_mean_value = update_carry["info"]["qf_mean_value"] / gradient_steps

        self._n_updates += gradient_steps

        agent_state = dict()
        agent_state["training_progress"] = (
            self.num_timesteps / self._total_timesteps
        ) * 100
        agent_state["ent_coef"] = self.exploration_rate

        env_unwrapped = self.get_env().unwrapped
        while not isinstance(env_unwrapped, CustomDummyVecEnv):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped = env_unwrapped.envs[0].env
        while not isinstance(env_unwrapped, EnvWrapper):
            env_unwrapped = env_unwrapped.unwrapped

        env_unwrapped.send_agent_state(agent_state=agent_state)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/qf_mean_value", qf_mean_value.item())
