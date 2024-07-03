import abc
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_generation.dataset import Dataset
from test_generation.utils.torch_utils import DEVICE, to_numpy


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction="none")
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_focal_mse_loss(
    inputs, targets, activate="sigmoid", beta=0.2, gamma=1, weights=None
):
    loss = (inputs - targets) ** 2
    loss *= (
        (th.tanh(beta * th.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * th.sigmoid(beta * th.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_focal_l1_loss(
    inputs, targets, activate="sigmoid", beta=0.2, gamma=1, weights=None
):
    loss = F.l1_loss(inputs, targets, reduction="none")
    loss *= (
        (th.tanh(beta * th.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * th.sigmoid(beta * th.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1.0, weights=None):
    l1_loss = th.abs(inputs - targets)
    cond = l1_loss < beta
    loss = th.where(cond, 0.5 * l1_loss**2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = th.mean(loss)
    return loss


class TestingPolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        loss_type: str,
        input_size: int,
        layers: int,
        policy: str,
        learning_rate: float = 3e-4,
    ):
        super(TestingPolicy, self).__init__()
        self.loss_type = loss_type
        self.num_evaluation_predictions = 0
        self.current_evaluation_predictions = 0
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = layers
        self.policy = policy

        assert (
            self.loss_type == "classification" or self.loss_type == "anomaly_detection"
        ), "Loss type {} not supported".format(self.loss_type)

    @staticmethod
    def get_model_architecture(
        input_size: int, layers: int, avf_policy: str
    ) -> Callable[[], nn.Module]:
        dropout_percentage = 0.1
        num_features = 32

        def __init_model() -> nn.Module:
            if avf_policy == "mlp":
                models = [
                    # 1 layer nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Linear(in_features=num_features, out_features=2),
                    ),
                    # 2 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=2),
                    ),
                    # 3 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=2),
                    ),
                    # 4 layers nn
                    nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=num_features),
                        nn.LayerNorm(normalized_shape=num_features),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_percentage),
                        nn.Linear(in_features=num_features, out_features=2),
                    ),
                ]
                assert 0 <= layers - 1 < len(models)
                return models[layers - 1]
            raise NotImplementedError("avf_policy {} not supported".format(avf_policy))

        return __init_model

    @staticmethod
    def loss_function(input: Tensor, target: Tensor, weights: Tensor = None) -> Tensor:
        if weights is not None:
            weights = weights[0].view(len(weights[0]))
        return F.cross_entropy(input=input, target=target, weight=weights)

    def save(self, filepath: str) -> None:
        th.save(self.state_dict(), filepath)

    def load(self, filepath: str, load_on_device: bool = False) -> None:
        if load_on_device:
            self.load_state_dict(th.load(filepath, map_location=th.device(DEVICE)))
        else:
            self.load_state_dict(th.load(filepath, map_location=th.device("cpu")))
        self.eval()

    @abc.abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError("Get model not implemented")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, data: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError("Generate not implemented")

    @staticmethod
    def compute_score(logits: Tensor) -> Tensor:
        # select failure class
        return F.softmax(logits, dim=1).squeeze()[:, 1]

    def forward(self, data):
        return self.get_model().forward(data)

    def _enable_dropout(self):
        self.eval()
        dropout_present_during_training = False
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith("Dropout"):
                dropout_present_during_training = True
                each_module.train()
        assert (
            dropout_present_during_training
        ), "Dropout layer not present during training. Not possible to evaluate model uncertainty"

    def _disable_dropout(self):
        dropout_present_during_training = False
        for each_module in self.modules():
            if each_module.__class__.__name__.startswith("Dropout"):
                dropout_present_during_training = True
                each_module.eval()
        assert (
            dropout_present_during_training
        ), "Dropout layer not present during training. Not possible to evaluate model uncertainty"

    @staticmethod
    def predictive_entropy(predictions: np.ndarray) -> float:
        epsilon = sys.float_info.min
        predictive_entropy = -np.sum(
            np.mean(predictions, axis=0)
            * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1,
        )
        return predictive_entropy

    @torch.no_grad()
    def get_failure_class_prediction_dropout(
        self,
        env_config_transformed: np.ndarray,
        dataset: Dataset,
        mc_dropout_num: int,
        count_num_evaluation: bool = True
        # ) -> Tuple[float, float]:
    ) -> float:
        assert mc_dropout_num > 0, "Number of times the model should be ran must be > 0"
        # predict stochastic dropout model T times
        self._enable_dropout()
        outputs = []
        for i in range(mc_dropout_num):
            prediction = self.get_failure_class_prediction(
                env_config_transformed=env_config_transformed,
                dataset=dataset,
                count_num_evaluation=count_num_evaluation,
            )
            if i == 0:
                outputs = prediction
            else:
                outputs = np.vstack((outputs, prediction))

        self._disable_dropout()
        return self.predictive_entropy(predictions=np.asarray(outputs))
        # return float(np.mean(outputs)), float(np.std(outputs))

    def get_activations_and_children(
        self, data: Tensor
    ) -> Tuple[Dict, List[nn.Module]]:
        self.eval()
        activations = dict()
        children = list(self.get_model().children())
        with torch.no_grad():
            layer_index = 0
            data = data.view(data.size(0), -1)
            # We need to manually loop through the layers to save all activations
            for layer_index, layer in enumerate(children[:-1]):
                data = layer(data)
                activations[layer_index] = data.view(-1).cpu().numpy()
        return activations, children

    # from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/02-activation-functions.html
    @torch.no_grad()
    def measure_number_dead_neurons(
        self, dataloader: DataLoader
    ) -> Tuple[List[Tensor], int]:
        """Function to measure the number of dead neurons in a trained neural network.

        For each neuron, we create a boolean variable initially set to 1. If it has an activation unequals 0 at any time, we
        set this variable to 0. After running through the whole training set, only dead neurons will have a 1.
        """
        device = "cpu"
        children = list(self.get_model().children())
        neurons_dead = [
            torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
            for layer in children[:-1]
            if isinstance(layer, nn.Linear)
        ]

        self.eval()
        with tqdm(dataloader, unit="batch") as epoch:
            for data, target, _ in epoch:
                layer_index = 0
                data = data.view(data.size(0), -1)
                for layer in children[:-1]:
                    data = layer(data)
                    # FIXME: not very accurate I guess, but activation functions do not have a type
                    # type of activation functions torch.nn.modules.activation
                    if "activation" in str(type(layer)):
                        # Are all activations == 0 in the batch, and we did not record the opposite in the last batches?
                        neurons_dead[layer_index] = torch.logical_and(
                            neurons_dead[layer_index], (data == 0).all(dim=0)
                        )
                        layer_index += 1

        number_neurons_dead = [t.sum().item() for t in neurons_dead]
        return neurons_dead, number_neurons_dead

    def forward_and_loss(
        self,
        data: Tensor,
        target: Tensor,
        training: bool = True,
        weights: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.loss_type == "classification":
            output = self.forward(data)
            predictions = (
                F.softmax(output, dim=1).detach().argmax(dim=1, keepdim=True).squeeze()
            )
            if training:
                return (
                    self.loss_function(input=output, target=target, weights=weights),
                    predictions,
                )
            return output.detach(), predictions
        else:
            raise NotImplementedError(
                "Loss type {} is not implemented".format(self.loss_type)
            )

    def get_failure_class_prediction(
        self,
        env_config_transformed: np.ndarray,
        dataset: Dataset,
        count_num_evaluation: bool = True,
        do_not_squeeze: bool = False,
    ) -> Union[np.ndarray, float]:
        if count_num_evaluation:
            self.num_evaluation_predictions += 1
            self.current_evaluation_predictions += 1

        output = self.forward(
            th.tensor(env_config_transformed, dtype=th.float32).view(1, -1)
        )

        output = to_numpy(F.softmax(output, dim=1))
        if dataset.output_scaler is not None:
            output = dataset.output_scaler.transform(X=output)

        if do_not_squeeze:
            return output

        return output.squeeze()[1]

    def get_num_evaluation_predictions(self) -> int:
        return self.num_evaluation_predictions

    def get_current_num_evaluation_predictions(self) -> int:
        return self.current_evaluation_predictions

    def reset_current_num_evaluation_predictions(self) -> None:
        self.current_evaluation_predictions = 0
