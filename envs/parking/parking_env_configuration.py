import copy
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from config import PARAM_SEPARATOR
from PIL import Image
from randomness_utils import EvaluationRandomGenerator, RandomGenerator
from test_generation.env_configuration import EnvConfiguration, EnvMutations

from envs.parking.parking_env import ParkingEnv

MAX_NUM_LANES = 10


class ParkingEnvConfiguration(EnvConfiguration):
    def __init__(
        self,
        goal_lane_idx: int = -1,
        heading_ego: float = 0.0,
        position_ego: Tuple[float, float] = (20.0, 0.0),
    ):
        super().__init__()
        self.num_lanes = MAX_NUM_LANES
        self.goal_lane_idx = goal_lane_idx
        self.heading_ego = heading_ego
        self.position_ego = position_ego
        self.limit_x_position = 10
        self.limit_y_position = 5

        self.key_names = ["goal_lane_idx", "heading_ego", "position_ego"]

        self.generator = RandomGenerator.get_instance()
        self.evaluation_generator = EvaluationRandomGenerator.get_instance()

        self.update_implementation(
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

    def generate_configuration(
        self, evaluate_agent_during_training: bool = False
    ) -> "EnvConfiguration":
        gnt = (
            self.generator
            if not evaluate_agent_during_training
            else self.evaluation_generator
        )

        is_valid = False
        while not is_valid:
            self.goal_lane_idx = gnt.rnd_state.randint(
                low=0, high=2 * self.num_lanes - 1
            )
            self.heading_ego = round(gnt.rnd_state.rand(), 2)

            self.position_ego = (
                round(float(10 * gnt.rnd_state.rand()), 2),
                round(float(5 * gnt.rnd_state.rand()), 2),
            )
            is_valid = self._is_valid()

        self.update_implementation(
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

        return self

    def _is_valid(self) -> bool:
        if self.goal_lane_idx < 0 or self.goal_lane_idx > 2 * MAX_NUM_LANES - 1:
            return False

        if round(self.heading_ego, 2) < 0.00 or round(self.heading_ego, 2) > 1.00:
            return False

        if (
            round(self.position_ego[0], 2) < -self.limit_x_position
            or round(self.position_ego[0], 2) > self.limit_x_position
        ):
            return False

        if (
            round(self.position_ego[1], 2) < -self.limit_y_position
            or round(self.position_ego[1], 2) > self.limit_y_position
        ):
            return False

        return True

    def get_image(self) -> np.ndarray:
        # FIXME: maybe there is a better way of doing it, instead of instantiating the environment every time
        env = ParkingEnv()
        env.num_lanes = self.num_lanes
        env.goal_lane_idx = self.goal_lane_idx
        env.heading_ego = self.heading_ego
        env.parked_vehicles_lane_indices = self.parked_vehicles_lane_indices
        env.position_ego = self.position_ego

        _ = env.reset()
        image = env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)

        env.close()
        return np.asarray(pil_image)

    def get_str(self) -> str:
        return "{}{}{}{}{}".format(
            self.goal_lane_idx,
            PARAM_SEPARATOR,
            self.heading_ego,
            PARAM_SEPARATOR,
            (self.position_ego[0], self.position_ego[1]),
        )

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split = s.split(PARAM_SEPARATOR)
        self.goal_lane_idx = int(split[0])
        self.heading_ego = float(split[1])
        split_2 = (
            split[2].replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        )
        self.position_ego = (float(split_2.split(",")[0]), float(split_2.split(",")[1]))

        self.update_implementation(
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

        return self

    def mutate_goal_lane_idx(
        self,
        env_config: "ParkingEnvConfiguration",
        env_mutation: EnvMutations = None,
        sign: str = "rnd",
    ) -> None:
        if env_mutation is not None or sign == "pos" or sign == "neg":
            v = env_config.goal_lane_idx
            v_copy = copy.deepcopy(v)
            if (env_mutation == EnvMutations.LEFT or sign == "neg") and v > 1:
                v -= self.generator.rnd_state.randint(low=1, high=2 * self.num_lanes)
                while v < 0:
                    v = v_copy
                    v -= self.generator.rnd_state.randint(
                        low=1, high=2 * self.num_lanes
                    )
            elif (env_mutation == EnvMutations.RIGHT or sign == "pos") and v < (
                2 * self.num_lanes
            ) - 1:
                v += self.generator.rnd_state.randint(low=1, high=2 * self.num_lanes)
                while v > 2 * self.num_lanes - 1:
                    v = v_copy
                    v += self.generator.rnd_state.randint(
                        low=1, high=2 * self.num_lanes
                    )
            env_config.goal_lane_idx = v
        else:
            v = env_config.goal_lane_idx
            v_copy = copy.deepcopy(v)
            if self.generator.rnd_state.rand() <= 0.5 and v < (2 * self.num_lanes) - 1:
                v += self.generator.rnd_state.randint(low=1, high=2 * self.num_lanes)
                while v > 2 * self.num_lanes - 1:
                    v = v_copy
                    v += self.generator.rnd_state.randint(
                        low=1, high=2 * self.num_lanes
                    )
            elif v > 1:
                v -= self.generator.rnd_state.randint(low=1, high=2 * self.num_lanes)
                while v < 0:
                    v = v_copy
                    v -= self.generator.rnd_state.randint(
                        low=1, high=2 * self.num_lanes
                    )
            env_config.goal_lane_idx = v

    def mutate_heading_ego(
        self,
        env_config: "ParkingEnvConfiguration",
        env_mutation: EnvMutations = None,
        sign: str = "rnd",
    ) -> None:
        if env_mutation is not None or sign == "pos" or sign == "neg":
            if env_mutation == EnvMutations.LEFT or sign == "neg":
                env_config.heading_ego -= float(self.generator.rnd_state.rand())
            elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                env_config.heading_ego += float(self.generator.rnd_state.rand())
        else:
            if self.generator.rnd_state.rand() <= 0.5:
                env_config.heading_ego += float(self.generator.rnd_state.rand())
            else:
                env_config.heading_ego -= float(self.generator.rnd_state.rand())
        env_config.heading_ego = round(env_config.heading_ego, 2)

    def mutate_position_ego(
        self,
        env_config: "ParkingEnvConfiguration",
        idx_to_mutate: int = None,
        env_mutation: EnvMutations = None,
        sign: str = "rnd",
    ) -> None:
        if idx_to_mutate is not None:
            if idx_to_mutate == 1:
                if env_mutation is not None or sign == "pos" or sign == "neg":
                    if env_mutation == EnvMutations.LEFT or sign == "neg":
                        env_config.position_ego = (
                            round(env_config.position_ego[0], 2),
                            round(
                                env_config.position_ego[1]
                                - float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                        )
                    elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                        env_config.position_ego = (
                            round(env_config.position_ego[0], 2),
                            round(
                                env_config.position_ego[1]
                                + float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                        )
                else:
                    if self.generator.rnd_state.rand() <= 0.5:
                        env_config.position_ego = (
                            round(env_config.position_ego[0], 2),
                            round(
                                env_config.position_ego[1]
                                - float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                        )
                    else:
                        env_config.position_ego = (
                            round(env_config.position_ego[0], 2),
                            round(
                                env_config.position_ego[1]
                                + float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                        )
                return
            if idx_to_mutate == 0:
                if env_mutation is not None or sign == "pos" or sign == "neg":
                    if env_mutation == EnvMutations.LEFT or sign == "neg":
                        env_config.position_ego = (
                            round(
                                env_config.position_ego[0]
                                - float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                            round(env_config.position_ego[1], 2),
                        )
                    elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                        env_config.position_ego = (
                            round(
                                env_config.position_ego[0]
                                + float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                            round(env_config.position_ego[1], 2),
                        )
                else:
                    if self.generator.rnd_state.rand() <= 0.5:
                        env_config.position_ego = (
                            round(
                                env_config.position_ego[0]
                                - float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                            round(env_config.position_ego[1], 2),
                        )
                    else:
                        env_config.position_ego = (
                            round(
                                env_config.position_ego[0]
                                + float(self.generator.rnd_state.rand(low=0, high=1)),
                                2,
                            ),
                            round(env_config.position_ego[1], 2),
                        )
                return
            raise RuntimeError(
                "Index {} not present for position_ego".format(idx_to_mutate)
            )
        else:
            position_ego_0 = round(
                float(self.generator.rnd_state.rand(low=0, high=1)), 2
            )
            position_ego_1 = round(
                float(self.generator.rnd_state.rand(low=0, high=1)), 2
            )
            f = self.generator.rnd_state.rand()
            if f <= 0.75:
                env_config.position_ego = (
                    round(env_config.position_ego[0] + position_ego_0, 2),
                    round(env_config.position_ego[1] + position_ego_1, 2),
                )
            elif f <= 0.5:
                env_config.position_ego = (
                    round(env_config.position_ego[0] + position_ego_0, 2),
                    round(env_config.position_ego[1] - position_ego_1, 2),
                )
            elif f <= 0.25:
                env_config.position_ego = (
                    round(env_config.position_ego[0] - position_ego_0, 2),
                    round(env_config.position_ego[1] + position_ego_1, 2),
                )
            else:
                env_config.position_ego = (
                    round(env_config.position_ego[0] - position_ego_0, 2),
                    round(env_config.position_ego[1] - position_ego_1, 2),
                )

    def mutate(self) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)

        # FIXME: change only one parameter with equal probability
        self.mutate_position_ego(env_config=new_env_config)
        self.mutate_heading_ego(env_config=new_env_config)
        self.mutate_goal_lane_idx(env_config=new_env_config)

        if new_env_config._is_valid():
            return new_env_config

        return None

    def mutate_hot(
        self, attributions: np.ndarray, mapping: Dict, minimize: bool
    ) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)
        # get indices as if the array attributions was sorted and reverse it (::-1)
        # indices_sort = np.argsort(attributions)[::-1]

        if minimize:
            attributions *= -1

        idx_to_mutate = random.choices(
            population=list(range(0, len(attributions))),
            weights=np.abs(attributions),
            k=1,
        )[0]
        key_to_mutate = self.get_key_to_mutate(
            idx_to_mutate=idx_to_mutate, mapping=mapping
        )

        if np.all(attributions >= 0):
            sign = "rnd"
        elif attributions[idx_to_mutate] > 0:
            sign = "pos"
        elif attributions[idx_to_mutate] < 0:
            sign = "neg"
        else:
            sign = "rnd"

        if key_to_mutate == "goal_lane_idx":
            self.mutate_goal_lane_idx(env_config=new_env_config, sign=sign)
        elif key_to_mutate == "heading_ego":
            self.mutate_heading_ego(env_config=new_env_config, sign=sign)
        elif key_to_mutate == "position_ego":
            self.mutate_position_ego(
                env_config=new_env_config,
                idx_to_mutate=mapping[key_to_mutate].index(idx_to_mutate),
                sign=sign,
            )
        else:
            raise RuntimeError("Key not present {}".format(key_to_mutate))

        if new_env_config._is_valid():
            return new_env_config

        return None

    def crossover(
        self, other_env_config: "EnvConfiguration", pos1: int, pos2: int
    ) -> Optional["EnvConfiguration"]:
        # FIXME similar to test suite crossover: implement also test case crossover
        #  (e.g. env.position_ego[0] can be exchanged with other_env.position_ego[0])
        new_env_config_impl = copy.deepcopy(self.impl)
        for i in range(pos1):
            new_env_config_impl[self.key_names[i]] = self.impl[self.key_names[i]]
        for i in range(pos2 + 1, self.get_length()):
            new_env_config_impl[self.key_names[i]] = other_env_config.impl[
                self.key_names[i]
            ]

        new_env_config = ParkingEnvConfiguration(**new_env_config_impl)

        if new_env_config._is_valid():
            return new_env_config
        return None

    def __eq__(self, other: "ParkingEnvConfiguration") -> bool:
        if self.goal_lane_idx != other.goal_lane_idx:
            return False

        if self.heading_ego != other.heading_ego:
            return False

        if self.position_ego[0] != other.position_ego[0]:
            return False

        if self.position_ego[1] != other.position_ego[1]:
            return False

        return True

    def compute_distance(self, other: "ParkingEnvConfiguration") -> float:
        raise NotImplementedError("Not implemented yet")
