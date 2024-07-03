import base64
from io import BytesIO
from typing import Dict, Optional, Tuple, cast

import numpy as np
from highway_env.envs import Action
from highway_env.envs.common.abstract import Observation
from PIL import Image
from test_generation.env_configuration import EnvConfiguration
from test_generation.env_wrapper import EnvWrapper
from test_generation.type_aliases import Observation
from wrappers.time_wrapper import TimeWrapper

from envs.lunarlander.lunarlander_env import LunarLanderEnv
from envs.lunarlander.lunarlander_env_configuration import LunarLanderEnvConfiguration
from envs.lunarlander.lunarlander_training_logs import LunarLanderTrainingLogs


class LunarLanderEnvWrapper(EnvWrapper):
    # no typing to avoid circular inputs when called from main
    def __init__(
        self,
        test_generator,
        time_wrapper: bool = False,
        headless: bool = True,
        **env_kwargs
    ):
        super(LunarLanderEnvWrapper, self).__init__(test_generator=test_generator)
        self.env: LunarLanderEnv = LunarLanderEnv(**env_kwargs)
        self.time_wrapper = time_wrapper
        if time_wrapper:
            self.env = TimeWrapper(env=self.env)

        self.configuration: EnvConfiguration = None
        self.agent_state = None
        self.first_frame_string = None
        self.actions = []
        self.rewards = []
        self.speeds = []
        self.fitness_values = []
        self.lander_trajectory = []
        self.headless = headless

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.seed = self.env.seed

    def unwrap(self):
        if self.time_wrapper:
            return self.env.env
        return self.env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action=action)

        if self.env.continuous:
            actions = list(map(lambda a: float(a), action))
        else:
            actions = int(action)

        self.actions.append(actions)
        self.rewards.append(reward)
        if info.get("fitness", None) is not None:
            self.fitness_values.append(info["fitness"])

        self.lander_trajectory.append(
            list(map(lambda a: float(a), list(info["lander_position"])))
        )
        if done:
            lundar_lander_training_logs = LunarLanderTrainingLogs(
                is_success=int(info["is_success"]),
                fitness_values=self.fitness_values,
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                lander_trajectory=self.lander_trajectory,
            )
            self.training_logs.append(lundar_lander_training_logs)
            self.lander_trajectory.clear()
            self.fitness_values.clear()

        return obs, reward, done, info

    def reset(self, end_of_episode: bool = False) -> Observation:
        if not end_of_episode:
            self.configuration: LunarLanderEnvConfiguration = cast(
                LunarLanderEnvConfiguration,
                self.test_generator.generate_env_configuration(),
            )
            self.configuration.update_implementation(
                height=self.configuration.height,
                apply_force_to_center=self.configuration.apply_force_to_center,
            )

            if not self.env.eval_env:
                self.env_configurations.append(self.configuration)
            else:
                self.eval_env_configurations.append(self.configuration)

            self.unwrap().height = self.configuration.height
            self.unwrap().apply_force_to_center = (
                self.configuration.apply_force_to_center
            )

        obs_reset = self.env.reset()
        if not self.headless:
            image = self.env.render(mode="rgb_array")
            buffered = BytesIO()
            pil_image = Image.fromarray(image)
            pil_image.save(buffered, optimize=True, format="PNG", quality=95)
            self.first_frame_string = base64.b64encode(buffered.getvalue()).decode(
                "utf-8"
            )

        return obs_reset

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()

    def send_agent_state(self, agent_state: Dict) -> None:
        self.agent_state = agent_state
