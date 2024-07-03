import base64
from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from test_generation.env_configuration import EnvConfiguration
from test_generation.env_wrapper import EnvWrapper
from test_generation.type_aliases import Observation
from wrappers.time_wrapper import TimeWrapper

from envs.cartpole.cartpole_env import CartPoleEnv
from envs.cartpole.cartpole_training_logs import CartPoleTrainingLogs


class CartPoleEnvWrapper(EnvWrapper):
    # no typing to avoid circular inputs when called from main
    def __init__(
        self,
        test_generator,
        time_wrapper: bool = False,
        headless: bool = True,
        **env_kwargs
    ):
        super(CartPoleEnvWrapper, self).__init__(test_generator=test_generator)
        self.env: CartPoleEnv = CartPoleEnv(**env_kwargs)
        self.time_wrapper = time_wrapper
        if time_wrapper:
            self.env = TimeWrapper(env=self.env)
        self.configuration: EnvConfiguration = None
        self.agent_state = None
        self.first_frame_string = None
        self.actions = []
        self.rewards = []
        self.speeds = []
        self.cart_positions = []
        self.cart_angles = []
        self.fitness_values = []
        self.headless = headless

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self.seed = self.env.seed

    def unwrap(self):
        if self.time_wrapper:
            return self.env.env
        return self.env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action=action)
        if self.env.discrete_action_space:
            actions = int(action)
        else:
            actions = float(action)

        self.actions.append(actions)
        self.rewards.append(reward)

        if info.get("fitness", None) is not None:
            self.fitness_values.append(info["fitness"])

        self.cart_positions.append(self.env.state[0])
        self.cart_angles.append(self.env.state[2])
        if done:
            cartpole_training_logs = CartPoleTrainingLogs(
                is_success=int(info["is_success"]),
                agent_state=self.agent_state,
                first_frame_string=self.first_frame_string,
                config=self.configuration,
                cart_positions=self.cart_positions,
                cart_angles=self.cart_angles,
                fitness_values=self.fitness_values,
            )
            self.training_logs.append(cartpole_training_logs)
            self.actions.clear()
            self.rewards.clear()
            self.cart_positions.clear()
            self.cart_angles.clear()
            self.fitness_values.clear()

        return obs, reward, done, info

    def reset(self, end_of_episode: bool = False) -> Observation:
        if not end_of_episode:
            self.configuration = self.test_generator.generate_env_configuration()
            self.configuration.update_implementation(
                x=self.configuration.x,
                x_dot=self.configuration.x_dot,
                theta=self.configuration.theta,
                theta_dot=self.configuration.theta_dot,
            )

            if not self.env.eval_env:
                self.env_configurations.append(self.configuration)
            else:
                self.eval_env_configurations.append(self.configuration)

            self.unwrap().x = self.configuration.x
            self.unwrap().x_dot = self.configuration.x_dot
            self.unwrap().theta = self.configuration.theta
            self.unwrap().theta_dot = self.configuration.theta_dot

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
