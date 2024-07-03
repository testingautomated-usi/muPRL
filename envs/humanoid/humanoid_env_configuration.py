import copy
import random
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
from config import PARAM_SEPARATOR, C
from PIL import Image
from randomness_utils import EvaluationRandomGenerator, RandomGenerator
from test_generation.env_configuration import EnvConfiguration

from envs.humanoid.humanoid_env_wrapper import HumanoidEnvWrapper


class HumanoidEnvConfiguration(EnvConfiguration):
    def __init__(
        self,
        qpos: np.ndarray = None,
        qvel: np.ndarray = None,
    ):
        super().__init__()
        self.init_qpos = np.asarray(
            [
                0.0,
                0.0,
                1.4,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.init_qvel = np.zeros(shape=(23,))
        if isinstance(qpos, List):
            self.qpos = np.asarray(qpos)
        else:
            self.qpos = qpos
        if isinstance(qvel, List):
            self.qvel = np.asarray(qvel)
        else:
            self.qvel = qvel
        self.c = C
        self.qpos_c = np.zeros(shape=self.init_qpos.shape) + self.c
        self.qvel_c = np.zeros(shape=self.init_qvel.shape) + self.c

        self.generator = RandomGenerator.get_instance()
        self.evaluation_generator = EvaluationRandomGenerator.get_instance()

        self.key_names = ["qpos", "qvel"]

        self.update_implementation(qpos=self.qpos, qvel=self.qvel)

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
            self.qpos = self.init_qpos + gnt.rnd_state.uniform(
                low=-self.c, high=self.c, size=self.init_qpos.shape
            )
            self.qvel = self.init_qvel + gnt.rnd_state.uniform(
                low=-self.c, high=self.c, size=self.init_qvel.shape
            )
            is_valid = self._is_valid()

        self.update_implementation(
            qpos=self.qpos,
            qvel=self.qvel,
        )

        return self

    def _is_valid(self) -> bool:
        if self.qpos is None:
            return False
        if self.qvel is None:
            return False
        qpos_bool = np.alltrue(self.qpos_c > np.abs(self.init_qpos - self.qpos))
        qvel_bool = np.alltrue(self.qvel_c > np.abs(self.init_qvel - self.qvel))
        return qpos_bool and qvel_bool

    def get_image(self) -> np.ndarray:
        # FIXME: maybe there is a better way of doing it, instead of instantiating the environment every time
        env = HumanoidEnvWrapper()
        env.set_state(qpos=self.qpos, qvel=self.qvel)

        _ = env.reset()
        image = env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)

        env.close()
        return np.asarray(pil_image)

    def get_str(self) -> str:
        return "{}{}{}".format(
            list(self.qpos),
            PARAM_SEPARATOR,
            list(self.qvel),
        )

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split = s.split(PARAM_SEPARATOR)
        split_1_0 = split[0].replace("[", "").replace("]", "").split(",")
        split_1_1 = split[1].replace("[", "").replace("]", "").split(",")
        self.qpos = np.asarray([float(num) for num in split_1_0])
        self.qvel = np.asarray([float(num) for num in split_1_1])
        assert self.qpos.shape[0] == self.init_qpos.shape[0], "Length does not match"
        assert self.qvel.shape[0] == self.init_qvel.shape[0], "Length does not match"

        self.update_implementation(
            qpos=self.qpos,
            qvel=self.qvel,
        )

        return self

    def mutate(self) -> Optional["EnvConfiguration"]:
        # FIXME: change only one parameter (i.e. either qpos or qvel) with equal probability

        qpos_shape = self.generator.rnd_state.randint(
            low=1, high=self.init_qpos.shape[0]
        )
        qpos_indices = self.generator.rnd_state.random_sample(
            population=range(0, self.init_qpos.shape[0]), k=qpos_shape
        )
        qvel_shape = self.generator.rnd_state.randint(
            low=1, high=self.init_qvel.shape[0]
        )
        qvel_indices = self.generator.rnd_state.random_sample(
            population=range(0, self.init_qvel.shape[0]), k=qvel_shape
        )

        new_env_config = copy.deepcopy(self)
        for idx in range(len(qpos_indices)):
            num = np.random.uniform(low=-self.c, high=self.c, size=1)
            if self.generator.rnd_state.rand() <= 0.5:
                new_env_config.qpos[idx] += num
            else:
                new_env_config.qpos[idx] -= num

        for idx in range(len(qvel_indices)):
            num = np.random.uniform(low=-self.c, high=self.c, size=1)
            if self.generator.rnd_state.rand() <= 0.5:
                new_env_config.qvel[idx] += num
            else:
                new_env_config.qvel[idx] -= num

        if new_env_config._is_valid():
            return new_env_config

        return None

    def mutate_idx_vector(
        self, idx_to_mutate: int, v: List[int], sign: str = "rnd"
    ) -> None:
        num = np.random.uniform(low=-self.c, high=self.c, size=1)
        if sign == "pos":
            v[idx_to_mutate] += num
        elif sign == "neg":
            v[idx_to_mutate] -= num
        else:
            if self.generator.rnd_state.rand() <= 0.5:
                v[idx_to_mutate] += num
            else:
                v[idx_to_mutate] -= num

    def mutate_vector(
        self,
        idx_to_mutate: int,
        key_to_mutate: str,
        env_config: EnvConfiguration,
        sign: str = "rnd",
    ) -> None:
        self.mutate_idx_vector(
            idx_to_mutate=idx_to_mutate, v=env_config.impl[key_to_mutate], sign=sign
        )

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
        keys_to_mutate = list(
            filter(lambda key: idx_to_mutate in mapping[key], mapping.keys())
        )
        assert (
            len(keys_to_mutate) == 1
        ), "There must be only one key where the attribution is max ({}). Found: {}".format(
            idx_to_mutate, len(keys_to_mutate)
        )
        key_to_mutate = keys_to_mutate[0]
        assert (
            key_to_mutate in self.key_names
        ), "Key to mutate not present in the implementation of env configuration: {} not in {}".format(
            key_to_mutate, self.key_names
        )

        if np.all(attributions >= 0):
            sign = "rnd"
        elif attributions[idx_to_mutate] > 0:
            sign = "pos"
        elif attributions[idx_to_mutate] < 0:
            sign = "neg"
        else:
            sign = "rnd"

        # print(idx_to_mutate, key_to_mutate, mapping)
        if key_to_mutate == "qpos":
            self.mutate_vector(
                idx_to_mutate=idx_to_mutate,
                key_to_mutate=key_to_mutate,
                env_config=new_env_config,
                sign=sign,
            )
        elif key_to_mutate == "qvel":
            self.mutate_vector(
                idx_to_mutate=idx_to_mutate - len(new_env_config.qpos),
                key_to_mutate=key_to_mutate,
                env_config=new_env_config,
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
        #  (e.g. env.qpos[0] can be exchanged with other_env.qpos[0])
        new_env_config_impl = copy.deepcopy(self.impl)
        for i in range(pos1):
            new_env_config_impl[self.key_names[i]] = self.impl[self.key_names[i]]
        for i in range(pos2 + 1, self.get_length()):
            new_env_config_impl[self.key_names[i]] = other_env_config.impl[
                self.key_names[i]
            ]

        new_env_config = HumanoidEnvConfiguration(**new_env_config_impl)

        if new_env_config._is_valid():
            return new_env_config
        return None

    def __eq__(self, other: "HumanoidEnvConfiguration") -> bool:
        qpos_bool = np.array_equal(self.qpos, other.qpos)
        if not qpos_bool:
            return False
        qvel_bool = np.array_equal(self.qvel, other.qvel)
        if not qvel_bool:
            return False

        return True

    def compute_distance(self, other: "HumanoidEnvConfiguration") -> float:
        # TODO: not clear how to normalize the two arrays to sum the two distances
        raise NotImplementedError("Not implemented for now")


if __name__ == "__main__":
    env_config = HumanoidEnvConfiguration()
    env_config = env_config.generate_configuration()
    print(env_config.get_str())
