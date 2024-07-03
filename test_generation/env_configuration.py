from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

import numpy as np


class EnvMutations(Enum):
    ADD = 0
    REMOVE = 1
    LEFT = 2
    RIGHT = 3


class EnvConfiguration(ABC):
    def __init__(self):
        self.impl: Dict = dict()
        self.key_names = []

    def update_implementation(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.impl[k] = v

    def get_length(self) -> int:
        return len(self.impl.keys())

    @abstractmethod
    def generate_configuration(
        self, evaluate_agent_during_training: bool = False
    ) -> "EnvConfiguration":
        raise NotImplementedError("Not implemented error")

    # FIXME: there is a bug in gym==0.25.0 that prevents the import of rendering module (ImportError: cannot import name 'rendering' from 'gym.envs.classic_control')
    # Downgrading to 0.21.0 makes it work -> https://stackoverflow.com/questions/71973392/importerror-cannot-import-rendering-from-gym-envs-classic-control
    # for now leaving 0.25.0 version, as it is required by the framework to work
    @abstractmethod
    def get_image(self) -> np.ndarray:
        raise NotImplementedError("Not implemented error")

    @abstractmethod
    def get_str(self) -> str:
        raise NotImplementedError("Not implemented error")

    @abstractmethod
    def str_to_config(self, s: str) -> "EnvConfiguration":
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def mutate(self) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented error")

    @abstractmethod
    def mutate_hot(
        self, attributions: np.ndarray, mapping: Dict
    ) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented error")

    @abstractmethod
    def crossover(
        self, other_env_config: "EnvConfiguration", pos1: int, pos2: int
    ) -> Optional["EnvConfiguration"]:
        raise NotImplementedError("Not implemented error")

    @abstractmethod
    def __eq__(self, other: "EnvConfiguration") -> bool:
        raise NotImplementedError("Not implemented error")

    def __hash__(self) -> int:
        return hash(self.get_str())

    @abstractmethod
    def compute_distance(self, other: "EnvConfiguration") -> float:
        raise NotImplementedError("Not implemented error")

    def get_key_to_mutate(self, idx_to_mutate: int, mapping: Dict) -> str:
        keys_to_mutate = list(
            filter(lambda key: idx_to_mutate in mapping[key], mapping.keys())
        )
        # print(idx_to_mutate, keys_to_mutate, attributions, attributions[idx_to_mutate])
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
        return key_to_mutate
