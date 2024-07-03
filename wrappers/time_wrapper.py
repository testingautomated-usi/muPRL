from typing import Union

import gym
from envs.env import Env
from test_generation.env_wrapper import EnvWrapper


class TimeWrapper(gym.Wrapper):
    def __init__(self, env: Union[gym.Env, Env]):
        super().__init__(env)
        self.steps = 0
        if isinstance(env, EnvWrapper):
            self.timeout_steps = env.unwrap().timeout_steps
            self.fail_on_timeout = env.unwrap().fail_on_timeout
        else:
            self.timeout_steps = env.timeout_steps
            self.fail_on_timeout = env.fail_on_timeout

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)
        if self.steps == self.timeout_steps:
            assert (
                info.get("is_success", None) is not None
            ), "The 'is_success' keyword must be present"
            if self.fail_on_timeout:
                info["is_success"] = 0
            else:
                info["is_success"] = 1
            done = True

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()
