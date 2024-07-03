from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from test_generation.utils.file_utils import (
    get_one_hot_for_mutant_coef,
    get_run_number,
    get_coef_number,
)
from log import Log
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset as ThDataset

from test_generation.env_configuration import EnvConfiguration
from test_generation.training_logs import TrainingLogs
from test_generation.type_aliases import Scaler
from test_generation.utils.torch_utils import DEVICE


class Data(ABC):
    def __init__(self, filename: str = None, index: int = -1):
        self.index = index
        self.filename = filename

    def get_filename(self) -> str:
        return self.filename

    def get_episode_index(self) -> int:
        return self.index

    @abstractmethod
    def get_label(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_config(self) -> EnvConfiguration:
        raise NotImplementedError()

    @abstractmethod
    def get_regression_value(self) -> Optional[float]:
        raise NotImplementedError()


class TrainingData(Data):
    def __init__(self, index: int, training_logs: TrainingLogs):
        super().__init__(index=index)
        self.training_logs = training_logs
        self.exploration_coefficient = self.training_logs.get_exploration_coefficient()

    def __lt__(self, other: "TrainingData"):
        # return np.max(self.reconstruction_losses) < np.max(other.reconstruction_losses)
        return self.get_training_progress() < other.get_training_progress()

    def get_exploration_coefficient(self) -> float:
        return self.exploration_coefficient

    def get_regression_value(self) -> Optional[float]:
        if self.training_logs.is_regression_value_set():
            return self.training_logs.get_regression_value()
        return None

    def get_training_progress(self) -> float:
        return self.training_logs.get_training_progress()

    def get_label(self) -> int:
        return self.training_logs.get_label()

    def get_config(self) -> EnvConfiguration:
        return self.training_logs.get_config()


class TestingData(Data):
    # TODO: refactor with index
    def __init__(
        self, filename: str, testing_configuration: Tuple[EnvConfiguration, float]
    ):
        super().__init__(filename=filename)
        self.testing_configuration = testing_configuration

    def get_regression_value(self) -> Optional[float]:
        # TODO: not supported for now
        return None

    def get_label(self) -> int:
        # takes care of deterministic and non-deterministic environments (0.5 is the threshold for a failure)
        return int(self.testing_configuration[1] > 0.5)

    def get_config(self) -> EnvConfiguration:
        return self.testing_configuration[0]


# defining the Dataset class
class TorchDataset(ThDataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weight_loss: bool = False,
    ):
        self.data = data
        self.labels = labels
        self.weight_loss = weight_loss
        self.weights = None
        if len(self.labels) > 0:
            if self.weight_loss:
                hist = dict(Counter(self.labels))
                n_classes = len(hist)
                self.weights = list(
                    compute_class_weight(
                        class_weight="balanced",
                        classes=np.arange(n_classes),
                        y=self.labels,
                    )
                )
            else:
                hist = dict(Counter(self.labels))
                self.weights = [np.float32(1.0) for _ in range(len(hist.keys()))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
            th.tensor(self.labels[index], dtype=th.long).to(DEVICE),
            th.tensor(self.weights, dtype=th.float32).to(DEVICE),
        )


class Dataset(ABC):
    def __init__(self, policy: str = None):
        self.dataset: List[Data] = []
        self.input_scaler: Scaler = None
        self.output_scaler: Scaler = None
        self.policy = policy

    def add(self, data: Data) -> None:
        self.dataset.append(data)

    def get(self) -> List[Data]:
        return self.dataset

    def get_num_failures(self) -> int:
        assert (
            len(self.dataset) != 0
        ), "Cannot compute num failures since the dataset is empty"
        return sum([1 if data_item.get_label() == 1 else 0 for data_item in self.get()])

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        raise NotImplementedError("Not implemented")

    def get_num_features(
        self,
        encode_run_and_conf: bool = False,
        mutant_name: str = None,
        log_path: str = None,
    ) -> int:
        assert (
            len(self.dataset) > 0
        ), "Not possible to infer num features since there is no data point"
        assert self.policy is not None, "Policy not instantiated"
        if self.policy == "mlp":
            data_item = self.dataset[0]
            length_mlp = len(
                self.transform_mlp(env_configuration=data_item.get_config())
            )
            if encode_run_and_conf:
                run_number = get_run_number(
                    filepath=data_item.filename, mutant_name=mutant_name
                )
                coef_number = get_coef_number(
                    filepath=data_item.filename, mutant_name=mutant_name
                )
                if coef_number is not None:
                    coef_one_hot = get_one_hot_for_mutant_coef(
                        coef_number=coef_number,
                        mutant_name=mutant_name,
                        log_path=log_path,
                    )
                else:
                    coef_one_hot = coef_number

                if run_number is not None and coef_one_hot is not None:
                    return length_mlp + 1 + len(coef_one_hot)
                if run_number is not None:
                    return length_mlp + 1
            return length_mlp
        # TODO: add cnn, i.e. number of channels
        raise NotImplementedError("Unknown policy: {}".format(self.policy))

    def transform_data_item(
        self,
        data_item: Data,
        encode_run_and_conf: bool = False,
        mutant_name: str = None,
        log_path: str = None,
    ) -> np.ndarray:
        run_number = get_run_number(
            filepath=data_item.filename, mutant_name=mutant_name
        )
        coef_number = get_coef_number(
            filepath=data_item.filename, mutant_name=mutant_name
        )
        if coef_number is not None:
            coef_one_hot = get_one_hot_for_mutant_coef(
                coef_number=coef_number, mutant_name=mutant_name, log_path=log_path
            )
        else:
            coef_one_hot = coef_number
        return self.transform_env_configuration(
            env_configuration=data_item.get_config(),
            policy_name=self.policy,
            encode_run_and_conf=encode_run_and_conf,
            run_number=run_number,
            coef_one_hot=coef_one_hot,
        )

    def transform_env_configuration(
        self,
        env_configuration: EnvConfiguration,
        policy_name: str,
        encode_run_and_conf: bool = False,
        run_number: Optional[int] = None,
        coef_one_hot: Optional[int] = None,
    ) -> np.ndarray:
        assert self.policy is not None, "Policy not instantiated"
        if policy_name == "mlp":
            transformed = self.transform_mlp(env_configuration=env_configuration)
            if self.input_scaler is not None:
                transformed = self.input_scaler.transform(
                    X=transformed.reshape(1, -1)
                ).squeeze()
            if encode_run_and_conf:
                to_concatenate = []
                if run_number is not None and coef_one_hot is not None:
                    to_concatenate.append(run_number)
                    to_concatenate.extend(coef_one_hot)
                elif run_number is not None:
                    to_concatenate.append(run_number)
                return np.concatenate([to_concatenate, transformed])
            return transformed
        raise NotImplementedError("Unknown policy: {}".format(policy_name))

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        raise NotImplementedError("Transform mlp not implemented")

    @abstractmethod
    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        raise NotImplementedError("Get mapping transformed not implemented")

    @abstractmethod
    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        raise NotImplementedError("Get original env configuration not implemented")

    @staticmethod
    def get_scalers_for_data(
        data: np.ndarray, labels: np.ndarray, regression: bool
    ) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        if regression:
            # input_scaler = MinMaxScaler()
            # input_scaler.fit(X=data)
            output_scaler = MinMaxScaler()
            output_scaler.fit(X=labels)
            return None, output_scaler

        input_scaler = StandardScaler()
        input_scaler.fit(X=data)
        # return input_scaler, None
        return None, None

    def compute_distance(
        self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration
    ) -> float:
        env_config_1_transformed = self.transform_env_configuration(
            env_configuration=env_config_1, policy=self.policy
        )
        env_config_2_transformed = self.transform_env_configuration(
            env_configuration=env_config_2, policy=self.policy
        )
        return euclidean_distances(
            u=env_config_1_transformed, v=env_config_2_transformed
        )

    @staticmethod
    def sampling(
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        under: bool = False,
        sampling_percentage: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger = Log("sampling")

        if sampling_percentage > 0.0:
            if under:
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_percentage, random_state=seed
                )
            else:
                sampler = RandomOverSampler(
                    sampling_strategy=sampling_percentage, random_state=seed
                )

            logger.info("Label proportions before sampling: {}".format(labels.mean()))
            sampled_data, sampled_labels = sampler.fit_resample(X=data, y=labels)
            logger.info(
                "Label proportions after sampling: {}".format(sampled_labels.mean())
            )

            return sampled_data, sampled_labels

        return data, labels

    @staticmethod
    def split_train_test(
        test_split: float,
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if test_split > 0.0:
            train_data, test_data, train_labels, test_labels = train_test_split(
                data,
                labels,
                test_size=test_split,
                shuffle=shuffle,
                stratify=labels,
                random_state=seed,
            )
        else:
            train_data, train_labels, test_data, test_labels = (
                data,
                labels,
                np.asarray([]),
                np.asarray([]),
            )
            if shuffle:
                np.random.shuffle(train_data)
                np.random.shuffle(train_labels)

        return train_data, train_labels.reshape(-1), test_data, test_labels.reshape(-1)

    def preprocess_test_data(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(
                    X=test_labels.reshape(len(test_labels), 1)
                ).reshape(len(test_labels))
        return test_data, test_labels

    # also assigns input and output scalers
    def preprocess_train_and_test_data(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger = Log("preprocess_train_and_test_data")
        # The statistics required for the transformation (e.g., the mean) are estimated
        # from the training set and are applied to all data sets (e.g., the test set or new samples)
        self.input_scaler, self.output_scaler = self.get_scalers_for_data(
            data=train_data, labels=train_labels.reshape(len(train_labels), 1)
        )
        if self.input_scaler is not None:
            logger.info("Preprocessing input data")
            train_data = self.input_scaler.transform(X=train_data)
        if self.output_scaler is not None:
            logger.info("Preprocessing output data")
            train_labels = self.output_scaler.transform(
                X=train_labels.reshape(len(train_labels), 1)
            ).reshape(len(train_labels))
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(
                    X=test_labels.reshape(len(test_labels), 1)
                ).reshape(len(test_labels))
        return train_data, train_labels, test_data, test_labels

    def transform_data(
        self,
        seed: int,
        test_split: float = 0.2,
        shuffle: bool = True,
        undersample_majority_class: float = 0.0,
        encode_run_and_conf: bool = False,
        mutant_name: str = None,
        log_path: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_features = self.get_num_features(
            encode_run_and_conf=encode_run_and_conf,
            mutant_name=mutant_name,
            log_path=log_path,
        )

        data = np.zeros(shape=(len(self.dataset), num_features))
        labels = np.zeros(shape=(len(self.dataset), 1))

        for idx in range(len(self.dataset)):
            data_item = self.dataset[idx]
            data[idx] = self.transform_data_item(
                data_item=data_item,
                encode_run_and_conf=encode_run_and_conf,
                mutant_name=mutant_name,
                log_path=log_path,
            )
            labels[idx] = data_item.get_label()

        if undersample_majority_class > 0.0:
            assert (
                0.0 < undersample_majority_class <= 0.8
            ), f"Undersampling percentage cannot be > 0.8. Found {undersample_majority_class}"
            # if failure class samples < 30%
            # 0.16 for 0.2, 0.23 for 0.3, 0.28 for 0.4, 0.33 for 0.5, 0.37 for 0.6, 0.41 for 0.7, 0.44 for 0.8
            if np.mean(labels) < 0.3:
                # resample such that the failure class samples are {undersample_majority_class} of the total data points
                data, labels = Dataset.sampling(
                    data=data,
                    labels=labels,
                    seed=seed,
                    under=True,
                    sampling_percentage=undersample_majority_class,
                )
            else:
                print("Undersampling is not needed as the failure class is > 30%")

        if (
            np.mean(labels) <= 0.1
            and len(list(filter(lambda label: label == 1, labels))) < 20
        ):
            raise RuntimeError(
                f"Number of failures too low {np.mean(labels)}, i.e., {sum(labels)}/{len(labels)} or number of failures is low {len(list(filter(lambda l: l == 1, labels)))}"
            )

        # TODO: the split should be contextual if encode_run_and_conf == True
        # create artificial labels for run_number and coef_number

        train_data, train_labels, test_data, test_labels = self.split_train_test(
            test_split=test_split,
            data=data,
            labels=labels,
            seed=seed,
            shuffle=shuffle,
        )

        return train_data, train_labels, test_data, test_labels
