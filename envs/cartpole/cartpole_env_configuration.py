import copy
import random
from io import BytesIO
from typing import Dict, Optional

import numpy as np
from config import PARAM_SEPARATOR
from PIL import Image
from randomness_utils import EvaluationRandomGenerator, RandomGenerator
from test_generation.env_configuration import EnvConfiguration, EnvMutations

from envs.cartpole.cartpole_env import CartPoleEnv


class CartPoleEnvConfiguration(EnvConfiguration):
    def __init__(
        self,
        x: float = 0.0,
        x_dot: float = 0.0,
        theta: float = 0.0,
        theta_dot: float = 0.0,
    ):
        super().__init__()
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

        self.low = -0.05
        self.high = 0.05

        self.generator = RandomGenerator.get_instance()
        self.evaluation_generator = EvaluationRandomGenerator.get_instance()

        self.key_names = ["x", "x_dot", "theta", "theta_dot"]
        self.update_implementation(
            x=self.x, x_dot=self.x_dot, theta=self.theta, theta_dot=self.theta_dot
        )

    def generate_configuration(
        self, evaluate_agent_during_training: bool = False
    ) -> "EnvConfiguration":
        gnt = (
            self.generator
            if not evaluate_agent_during_training
            else self.evaluation_generator
        )
        # print("Eval generator" if evaluate_agent_during_training else "Train generator")

        is_valid = False
        while not is_valid:
            self.x = gnt.rnd_state.uniform(low=self.low, high=self.high)
            self.x_dot = gnt.rnd_state.uniform(low=self.low, high=self.high)
            self.theta = gnt.rnd_state.uniform(low=self.low, high=self.high)
            self.theta_dot = gnt.rnd_state.uniform(low=self.low, high=self.high)
            is_valid = self._is_valid()

        self.update_implementation(
            x=self.x, x_dot=self.x_dot, theta=self.theta, theta_dot=self.theta_dot
        )
        return self

    def _is_valid(self) -> bool:
        if self.x < self.low or self.x > self.high:
            return False

        if self.x_dot < self.low or self.x_dot > self.high:
            return False

        if self.theta < self.low or self.theta > self.high:
            return False

        if self.theta_dot < self.low or self.theta_dot > self.high:
            return False

        return True

    def get_image(self) -> np.ndarray:
        # FIXME: maybe there is a better way of doing it, instead of instantiating the environment every time
        env = CartPoleEnv()
        env.x = self.x
        env.x_dot = self.x_dot
        env.theta = self.theta
        env.theta_dot = self.theta_dot

        _ = env.reset()
        image = env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)

        env.close()
        return np.asarray(pil_image)

    def get_str(self) -> str:
        return "{}{}{}{}{}{}{}".format(
            self.x,
            PARAM_SEPARATOR,
            self.x_dot,
            PARAM_SEPARATOR,
            self.theta,
            PARAM_SEPARATOR,
            self.theta_dot,
        )

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split = s.split(PARAM_SEPARATOR)
        self.x = float(split[0])
        self.x_dot = float(split[1])
        self.theta = float(split[2])
        self.theta_dot = float(split[3])

        self.update_implementation(
            x=self.x, x_dot=self.x_dot, theta=self.theta, theta_dot=self.theta_dot
        )
        return self

    def mutate_param(
        self,
        param_name: str,
        env_config: "CartPoleEnvConfiguration",
        env_mutation: EnvMutations = None,
        sign: str = "rnd",
    ) -> None:
        assert (
            param_name == "x"
            or param_name == "x_dot"
            or param_name == "theta"
            or param_name == "theta_dot"
        ), "Unknown param: {}".format(param_name)

        if env_mutation is not None or sign == "pos" or sign == "neg":
            if env_mutation == EnvMutations.LEFT or sign == "neg":
                if param_name == "x":
                    env_config.x = self.generator.rnd_state.uniform(
                        low=env_config.low, high=env_config.high
                    )
                elif param_name == "x_dot":
                    env_config.x_dot = self.generator.rnd_state.uniform(
                        low=env_config.low, high=env_config.high
                    )
                elif param_name == "theta":
                    env_config.theta = self.generator.rnd_state.uniform(
                        low=env_config.low, high=env_config.high
                    )
                elif param_name == "theta_dot":
                    env_config.theta_dot = self.generator.rnd_state.uniform(
                        low=env_config.low, high=env_config.high
                    )
                else:
                    raise RuntimeError("Unknown param: {}".format(param_name))

            elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                if param_name == "x":
                    env_config.x = self.generator.rnd_state.uniform(
                        low=self.x, high=env_config.high
                    )
                elif param_name == "x_dot":
                    env_config.x_dot = self.generator.rnd_state.uniform(
                        low=self.x_dot, high=env_config.high
                    )
                elif param_name == "theta":
                    env_config.theta = self.generator.rnd_state.uniform(
                        low=self.theta, high=env_config.high
                    )
                elif param_name == "theta_dot":
                    env_config.theta_dot = self.generator.rnd_state.uniform(
                        low=self.theta_dot, high=env_config.high
                    )
                else:
                    raise RuntimeError("Unknown param: {}".format(param_name))

            else:
                raise RuntimeError("Not possible to mutate masscart")
        else:
            if param_name == "x":
                env_config.x = self.generator.rnd_state.uniform(
                    low=env_config.low, high=self.x
                )
            elif param_name == "x_dot":
                env_config.x_dot = self.generator.rnd_state.uniform(
                    low=env_config.low, high=self.x_dot
                )
            elif param_name == "theta":
                env_config.theta = self.generator.rnd_state.uniform(
                    low=env_config.low, high=self.theta
                )
            elif param_name == "theta_dot":
                env_config.theta_dot = self.generator.rnd_state.uniform(
                    low=env_config.low, high=self.theta_dot
                )
            else:
                raise RuntimeError("Unknown param: {}".format(param_name))

    def mutate(self) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)

        r = self.generator.rnd_state.random()

        if r < 0.25:
            self.mutate_param(param_name="x", env_config=new_env_config)
        elif 0.25 <= r < 0.5:
            self.mutate_param(param_name="x_dot", env_config=new_env_config)
        elif 0.5 < r <= 0.75:
            self.mutate_param(param_name="theta", env_config=new_env_config)
        else:
            self.mutate_param(param_name="theta_dot", env_config=new_env_config)

        if new_env_config._is_valid():
            return new_env_config

        return None

    def mutate_hot(
        self, attributions: np.ndarray, mapping: Dict
    ) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)
        # get indices as if the array attributions was sorted and reverse it (::-1)
        # indices_sort = self.np.argsort(attributions)[::-1]
        idx_to_mutate = random.choices(
            population=list(range(0, len(attributions))),
            weights=self.np.abs(attributions),
            k=1,
        )[0]
        key_to_mutate = self.get_key_to_mutate(
            idx_to_mutate=idx_to_mutate, mapping=mapping
        )

        if self.np.all(attributions >= 0):
            sign = "rnd"
        elif attributions[idx_to_mutate] > 0:
            sign = "pos"
        elif attributions[idx_to_mutate] < 0:
            sign = "neg"
        else:
            sign = "rnd"

        if key_to_mutate == "x":
            self.mutate_param(param_name="x", env_config=new_env_config, sign=sign)
        elif key_to_mutate == "x_dot":
            self.mutate_param(param_name="x_dot", env_config=new_env_config, sign=sign)
        elif key_to_mutate == "theta":
            self.mutate_param(param_name="theta", env_config=new_env_config, sign=sign)
        elif key_to_mutate == "theta_dot":
            self.mutate_param(
                param_name="theta_dot", env_config=new_env_config, sign=sign
            )
        else:
            raise RuntimeError("Key not present {}".format(key_to_mutate))

        if new_env_config._is_valid():
            return new_env_config

        return None

    def crossover(
        self, other_env_config: "EnvConfiguration", pos1: int, pos2: int
    ) -> Optional["EnvConfiguration"]:
        new_env_config_impl = copy.deepcopy(self.impl)
        for i in range(pos1):
            new_env_config_impl[self.key_names[i]] = self.impl[self.key_names[i]]
        for i in range(pos2 + 1, self.get_length()):
            new_env_config_impl[self.key_names[i]] = other_env_config.impl[
                self.key_names[i]
            ]

        new_env_config = CartPoleEnvConfiguration(**new_env_config_impl)

        if new_env_config._is_valid():
            return new_env_config
        return None

    def __eq__(self, other: "CartPoleEnvConfiguration") -> bool:
        if self.x != other.x:
            return False

        if self.x_dot != other.x_dot:
            return False

        if self.theta != other.theta:
            return False

        if self.theta_dot != other.theta_dot:
            return False

        return True

    @staticmethod
    def normalize_feature(feature: float, min_value: float, max_value: float) -> float:
        return (feature - min_value) / (max_value - min_value)

    def compute_distance(self, other: "CartPoleEnvConfiguration") -> float:
        x_norm = self.normalize_feature(
            feature=self.x, min_value=self.low, max_value=self.high
        )
        x_dot_norm = self.normalize_feature(
            feature=self.x_dot, min_value=self.low, max_value=self.high
        )
        theta_norm = self.normalize_feature(
            feature=self.theta, min_value=self.low, max_value=self.high
        )
        theta_dot_norm = self.normalize_feature(
            feature=self.theta_dot, min_value=self.low, max_value=self.high
        )

        other_x_norm = self.normalize_feature(
            feature=other.x, min_value=self.low, max_value=self.high
        )
        other_x_dot_norm = self.normalize_feature(
            feature=other.x_dot, min_value=self.low, max_value=self.high
        )
        other_theta_norm = self.normalize_feature(
            feature=other.theta, min_value=self.low, max_value=self.high
        )
        other_theta_dot_norm = self.normalize_feature(
            feature=other.theta_dot, min_value=self.low, max_value=self.high
        )

        return float(
            np.mean(
                [
                    abs(x_norm - other_x_norm),
                    abs(x_dot_norm - other_x_dot_norm),
                    abs(theta_norm - other_theta_norm),
                    abs(theta_dot_norm - other_theta_dot_norm),
                ]
            )
        )
