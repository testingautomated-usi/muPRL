from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
from config import PARAM_SEPARATOR
from PIL import Image
from randomness_utils import EvaluationRandomGenerator, RandomGenerator
from test_generation.env_configuration import EnvConfiguration, EnvMutations

from envs.lunarlander.lunarlander_env import CHUNKS, INITIAL_RANDOM, H, LunarLanderEnv


class LunarLanderEnvConfiguration(EnvConfiguration):
    def __init__(
        self,
        height: np.ndarray = None,
        apply_force_to_center: Tuple[float] = None,
    ):
        super().__init__()

        self.height = height
        self.apply_force_to_center = apply_force_to_center

        self.generator = RandomGenerator.get_instance()
        self.evaluation_generator = EvaluationRandomGenerator.get_instance()

        self.key_names = ["height", "apply_force_to_center"]
        self.update_implementation(
            height=self.height, apply_force_to_center=self.apply_force_to_center
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
            # 5 is the number of elements of the height vector that are set by the environment at every reset,
            # regardless of their values.
            self.height = gnt.rnd_state.uniform(0, H / 2, size=(CHUNKS - 5 + 1,))
            self.apply_force_to_center = (
                gnt.rnd_state.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                gnt.rnd_state.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            )
            is_valid = self._is_valid()

        self.update_implementation(
            height=self.height, apply_force_to_center=self.apply_force_to_center
        )
        return self

    def _is_valid(self) -> bool:
        if self.height is None:
            return False
        if self.apply_force_to_center is None:
            return False

        height_bool = np.all((self.height >= 0) & (self.height < H / 2))
        apply_force_to_center_np = np.asarray(self.apply_force_to_center)
        apply_force_to_center_bool = np.all(
            (apply_force_to_center_np >= -INITIAL_RANDOM)
            & (apply_force_to_center_np < INITIAL_RANDOM)
        )

        return height_bool and apply_force_to_center_bool

    def get_image(self) -> np.ndarray:
        # FIXME: maybe there is a better way of doing it, instead of instantiating the environment every time
        env = LunarLanderEnv()
        env.height = self.height
        env.apply_force_to_center = self.apply_force_to_center

        _ = env.reset()
        image = env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)

        env.close()
        return np.asarray(pil_image)

    def get_str(self) -> str:
        return "{}{}{}".format(
            list(self.height), PARAM_SEPARATOR, tuple(self.apply_force_to_center)
        )

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split = s.split(PARAM_SEPARATOR)
        split_1_0 = split[0].replace("[", "").replace("]", "").split(",")
        split_1_1 = split[1].replace("(", "").replace(")", "").split(",")
        self.height = np.asarray([float(num) for num in split_1_0])
        self.apply_force_to_center = np.asarray([float(num) for num in split_1_1])

        self.update_implementation(
            height=self.height, apply_force_to_center=self.apply_force_to_center
        )
        return self

    def mutate_param(
        self,
        param_name: str,
        env_config: "LunarLanderEnvConfiguration",
        env_mutation: EnvMutations = None,
        sign: str = "rnd",
    ) -> None:
        raise NotImplementedError("Not implemented yet")

    def mutate(self) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented yet")

    def mutate_hot(
        self, attributions: np.ndarray, mapping: Dict
    ) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented yet")

    def crossover(
        self, other_env_config: "EnvConfiguration", pos1: int, pos2: int
    ) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented yet")

    def __eq__(self, other: "LunarLanderEnvConfiguration") -> bool:
        height_bool = np.array_equal(self.height, other.height)
        if not height_bool:
            return False
        apply_force_to_center_bool = np.array_equal(
            np.asarray(self.apply_force_to_center),
            np.asarray(other.apply_force_to_center),
        )
        if not apply_force_to_center_bool:
            return False

        return True

    def compute_distance(self, other: "LunarLanderEnvConfiguration") -> float:
        raise NotImplementedError("Not implemented yet")
