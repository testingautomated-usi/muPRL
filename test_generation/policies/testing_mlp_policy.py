import torch.nn as nn
from torch import Tensor

from test_generation.policies.testing_policy import TestingPolicy
from test_generation.utils.torch_utils import DEVICE


class TestingMlpPolicy(TestingPolicy):
    def __init__(
        self, input_size: int, layers: int = 4, learning_rate: float = 3e-4
    ) -> None:
        super(TestingMlpPolicy, self).__init__(
            loss_type="classification",
            learning_rate=learning_rate,
            input_size=input_size,
            layers=layers,
            policy="mlp",
        )
        self.model = self.get_mlp_architecture(input_size=input_size, layers=layers).to(
            DEVICE
        )

    def get_model(self) -> nn.Module:
        return self.model

    def generate(self, data: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError("Generate not implemented")

    def get_mlp_architecture(self, input_size: int, layers: int = 4) -> nn.Module:
        return self.get_model_architecture(
            input_size=input_size, layers=layers, avf_policy="mlp"
        )()
