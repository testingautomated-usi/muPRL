import random

import numpy as np
import torch as th
from numpy.random import RandomState


def set_random_seed(
    seed: int, using_cuda: bool = False, reset_random_gen: bool = False
) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    :param reset_random_gen:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    _ = RandomGenerator.get_instance(seed=seed, reset_random_gen=reset_random_gen)
    # I don't know if the +1 is necessary
    _ = EvaluationRandomGenerator.get_instance(
        seed=seed + 1, reset_random_gen=reset_random_gen
    )

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


class RandomGenerator:
    __instance: "RandomGenerator" = None

    @staticmethod
    def get_instance(
        seed: int = 0, reset_random_gen: bool = False
    ) -> "RandomGenerator":
        if RandomGenerator.__instance is None:
            RandomGenerator(seed=seed)
        elif reset_random_gen:
            RandomGenerator.__instance = None
            RandomGenerator(seed=seed)
        return RandomGenerator.__instance

    def __init__(self, seed: int):
        if RandomGenerator.__instance is not None:
            raise Exception("This class is a singleton!")

        self.rnd_state: RandomState = np.random.RandomState(seed=seed)
        RandomGenerator.__instance = self


class EvaluationRandomGenerator:
    __instance: "EvaluationRandomGenerator" = None

    @staticmethod
    def get_instance(
        seed: int = 0, reset_random_gen: bool = False
    ) -> "EvaluationRandomGenerator":
        if EvaluationRandomGenerator.__instance is None:
            EvaluationRandomGenerator(seed=seed)
        elif reset_random_gen:
            EvaluationRandomGenerator.__instance = None
            EvaluationRandomGenerator(seed=seed)
        return EvaluationRandomGenerator.__instance

    def __init__(self, seed: int):
        if EvaluationRandomGenerator.__instance is not None:
            raise Exception("This class is a singleton!")

        self.rnd_state: RandomState = np.random.RandomState(seed=seed)
        EvaluationRandomGenerator.__instance = self
