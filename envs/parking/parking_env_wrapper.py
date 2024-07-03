import base64
from io import BytesIO
from typing import Dict, Optional, Tuple, cast

import numpy as np
from highway_env.envs import Action
from highway_env.envs.common.abstract import Observation
from PIL import Image
from test_generation.env_configuration import EnvConfiguration
from test_generation.env_wrapper import EnvWrapper

from envs.parking.parking_env import ParkingEnv
from envs.parking.parking_env_configuration import ParkingEnvConfiguration
from envs.parking.parking_training_logs import ParkingTrainingLogs


class ParkingEnvWrapper(EnvWrapper):
    # no typing to avoid circular inputs when called from main
    def __init__(
        self,
        test_generator,
        time_wrapper: bool = False,
        headless: bool = True,
        **env_kwargs
    ):
        super(ParkingEnvWrapper, self).__init__(test_generator=test_generator)
        self.env: ParkingEnv = ParkingEnv(**env_kwargs)
        self.env.config["offscreen_rendering"] = "1" if headless else "0"

        self.configuration: EnvConfiguration = None
        self.agent_state = None
        self.first_frame_string = None
        self.actions = []
        self.rewards = []
        self.speeds = []
        self.fitness_values = []
        self.car_trajectory = []

        # FIXME: not used
        self.time_wrapper = time_wrapper
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

    # abstract method of ParkingEnv
    def _cost(self, action: Action) -> float:
        return self.env._cost(action=action)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action=action)
        actions = list(map(lambda a: float(a), action))
        self.actions.append(actions)
        self.rewards.append(reward)
        self.speeds.append(info["speed"])
        if info.get("fitness", None) is not None:
            self.fitness_values.append(info["fitness"])

        self.car_trajectory.append(
            list(map(lambda a: float(a), list(info["vehicle_position"])))
        )
        if done:
            assert self.first_frame_string is not None, "First frame not yet encoded"
            parking_training_logs = ParkingTrainingLogs(
                is_success=int(info["is_success"]),
                fitness_values=self.fitness_values,
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                car_trajectory=self.car_trajectory,
            )
            self.training_logs.append(parking_training_logs)
            self.car_trajectory.clear()
            self.fitness_values.clear()

        return obs, reward, done, info

    def reset(self, end_of_episode: bool = False) -> Observation:
        if not end_of_episode:
            self.configuration: ParkingEnvConfiguration = cast(
                ParkingEnvConfiguration,
                self.test_generator.generate_env_configuration(),
            )
            self.configuration.update_implementation(
                goal_lane_idx=self.configuration.goal_lane_idx,
                heading_ego=self.configuration.heading_ego,
                position_ego=(
                    self.configuration.position_ego[0],
                    self.configuration.position_ego[1],
                ),
            )

            if not self.env.eval_env:
                self.env_configurations.append(self.configuration)
            else:
                self.eval_env_configurations.append(self.configuration)

            self.unwrap().goal_lane_idx = self.configuration.goal_lane_idx
            self.unwrap().heading_ego = self.configuration.heading_ego
            self.unwrap().position_ego = self.configuration.position_ego

        obs_reset = self.env.reset()
        image = self.env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)
        self.first_frame_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return obs_reset

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        return self.env.compute_reward(
            achieved_goal=achieved_goal, desired_goal=desired_goal, info=info, p=p
        )

    def send_agent_state(self, agent_state: Dict) -> None:
        self.agent_state = agent_state
