from abc import abstractmethod
from typing import Dict, List, Optional

import gym
import numpy as np

from test_generation.env_configuration import EnvConfiguration
from test_generation.training_logs import TrainingLogs


class EnvWrapper(gym.Env):
    # no typing to avoid circular inputs when called from main
    def __init__(self, test_generator):
        self.test_generator = test_generator
        self.agent_state = None
        self.env_configurations: List[EnvConfiguration] = []
        self.eval_env_configurations: List[EnvConfiguration] = []
        self.training_logs: TrainingLogs = []

    @abstractmethod
    def unwrap(self):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def step(self, action: np.ndarray):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def reset(self, end_of_episode: bool = False):
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def send_agent_state(self, agent_state: Dict) -> None:
        assert NotImplementedError("Not implemented error")

    @abstractmethod
    def close(self) -> None:
        assert NotImplementedError("Not implemented error")
