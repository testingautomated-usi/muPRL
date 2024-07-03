import math
import os
from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from config import ENV_NAMES
from log import Log
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_generation.dataset import TorchDataset
from test_generation.policies.testing_policy import TestingPolicy
from test_generation.policy_factory import get_testing_policy
from test_generation.test_generation_config import DNN_POLICIES
from test_generation.utils.torch_utils import to_numpy


class FailurePredictor:
    def __init__(
        self,
        env_name: str,
        log_path: str,
    ):
        self.logger = Log("test_generator")
        self.env_name = env_name
        assert self.env_name in ENV_NAMES, "Env not supported: {}".format(env_name)
        self.log_path = log_path

    def train_dnn(
        self,
        input_size: int,
        train_dataset: TorchDataset,
        validation_dataset: TorchDataset,
        test_dataset: TorchDataset,
        testing_policy_for_training_name: str,
        mutant_name: str = None,
        mutant_configuration: str = None,
        n_epochs: int = 20,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 20,
        batch_size: int = 64,
        training_progress_filter: int = 0,
        layers: int = 4,
        weight_loss: bool = False,
    ) -> Tuple[float, float, float, float, float, float]:
        powers_of_two = [2, 4, 8, 16, 32, 64, 128]

        self.logger.info(f"Train data labels proportion: {train_dataset.labels.mean()}")
        self.logger.info(
            f"Validation data labels proportion: {validation_dataset.labels.mean()}"
        )
        if test_dataset is not None:
            self.logger.info(
                f"Test data labels proportion: {test_dataset.labels.mean()}"
            )

        failures_train_dataset = reduce(
            lambda a, b: a + b, filter(lambda label: label == 1, train_dataset.labels)
        )
        failures_validation_dataset = reduce(
            lambda a, b: a + b,
            filter(lambda label: label == 1, validation_dataset.labels),
        )
        if test_dataset is not None:
            failures_test_dataset = reduce(
                lambda a, b: a + b,
                filter(lambda label: label == 1, test_dataset.labels),
            )

        non_failures_train_dataset = len(train_dataset.labels) - failures_train_dataset
        non_failures_validation_dataset = (
            len(validation_dataset.labels) - failures_validation_dataset
        )
        if test_dataset is not None:
            non_failures_test_dataset = len(test_dataset.labels) - failures_test_dataset

        self.logger.info(
            f"Failures train dataset: {failures_train_dataset}/{non_failures_train_dataset} non failures"
        )
        self.logger.info(
            f"Failures validation dataset: {failures_validation_dataset}/{non_failures_validation_dataset} non failures"
        )
        if test_dataset is not None:
            self.logger.info(
                f"Failures test dataset: {failures_test_dataset}/{non_failures_test_dataset} non failures"
            )

        if test_dataset is not None:
            # otherwise impossible to compute precision and recall on the failure class
            assert (
                failures_test_dataset > 0
            ), "Failures in test dataset must be > 0. Found: {}".format(
                failures_test_dataset
            )
        else:
            # otherwise impossible to compute precision and recall on the failure class
            assert (
                failures_validation_dataset > 0
            ), "Failures in test dataset must be > 0. Found: {}".format(
                failures_validation_dataset
            )

        if len(train_dataset.data) < batch_size:
            # find first power of two smaller than validation_set length
            batch_size = list(
                filter(lambda n: n < len(train_dataset.data), powers_of_two)
            )[-1]
            self.logger.info(f"New batch size for training dataset: {batch_size}")

        if len(validation_dataset.data) < batch_size:
            # find first power of two smaller than validation_set length
            batch_size = list(
                filter(lambda n: n < len(validation_dataset.data), powers_of_two)
            )[-1]
            self.logger.info(f"New batch size for validation dataset: {batch_size}")

        if test_dataset is not None:
            self.logger.info(
                f"Train dataset size: {len(train_dataset.data)}, "
                f"Validation dataset size: {len(validation_dataset.data)}, "
                f"Test dataset size: {len(test_dataset.data)}"
            )
        else:
            self.logger.info(
                f"Train dataset size: {len(train_dataset.data)}, "
                f"Validation dataset size: {len(validation_dataset.data)}"
            )

        learning_rate = learning_rate
        patience = patience
        n_val_epochs_no_improve = 0

        save_path = self.get_policy_save_path(
            log_path=self.log_path,
            training_progress_filter=training_progress_filter,
            testing_policy_for_training_name=testing_policy_for_training_name,
            layers=layers,
            mutant_name=mutant_name,
            mutant_configuration=mutant_configuration,
        )

        testing_policy = get_testing_policy(
            policy_name=testing_policy_for_training_name,
            input_size=input_size,
            layers=layers,
            learning_rate=learning_rate,
        )
        self.logger.info(f"Model architecture: {testing_policy.get_model()}")
        self.logger.info(f"Model parameters: {testing_policy.get_num_params()}")

        optimizer = optim.Adam(
            params=testing_policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        best_val_precision = -1.0
        best_val_loss = np.inf
        test_accuracy: float = None
        best_epochs: float = -1

        for epoch in range(n_epochs):
            if epoch > 0:  # test untrained net first
                testing_policy.train()
                train_accuracy = 0
                train_loss = 0
                train_batches = 0
                with tqdm(train_dataloader, unit="batch") as train_epoch:
                    for train_data, train_target, weights in train_epoch:
                        # ===================forward=====================
                        loss, predictions = testing_policy.forward_and_loss(
                            data=train_data, target=train_target, weights=weights
                        )

                        if predictions is not None:
                            correct = (predictions == train_target).sum().item()
                            accuracy = correct / len(train_target)
                            train_accuracy += accuracy

                        train_loss += loss.item()
                        train_batches += 1
                        # ===================backward====================
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_epoch.set_postfix(epoch=epoch)

                train_loss /= train_batches
                train_accuracy /= train_batches
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                self.logger.info(
                    "Train loss: {:.2f}, Train accuracy: {:.2f}, Epoch: {}".format(
                        train_loss, train_accuracy, epoch
                    )
                )

            if len(validation_dataset.data) > 0:
                # calculate accuracy on validation set
                val_loss = 0
                val_accuracy = 0

                total_tp = 0
                total_fp = 0

                total_fn = 0

                with torch.no_grad():
                    # switch model to evaluation mode
                    testing_policy.eval()
                    with tqdm(validation_dataloader, unit="batch") as validation_epoch:
                        validation_batches = 0
                        for (
                            validation_data,
                            validation_target,
                            weights,
                        ) in validation_epoch:
                            # ===================forward=====================
                            loss, predictions = testing_policy.forward_and_loss(
                                data=validation_data,
                                target=validation_target,
                                weights=weights,
                            )

                            if predictions is not None:
                                correct = (
                                    (predictions == validation_target).sum().item()
                                )
                                accuracy = correct / len(validation_target)
                                val_accuracy += accuracy

                            validation_epoch.set_postfix(epoch=epoch)
                            val_loss += loss.item()
                            validation_batches += 1

                            # Compute TP, FP, TN, FN for the batch
                            tp = (
                                (
                                    (to_numpy(predictions) == 1)
                                    & (to_numpy(validation_target) == 1)
                                )
                                .sum()
                                .item()
                            )
                            fp = (
                                (
                                    (to_numpy(predictions) == 1)
                                    & (to_numpy(validation_target) == 0)
                                )
                                .sum()
                                .item()
                            )
                            fn = (
                                (
                                    (to_numpy(predictions) == 0)
                                    & (to_numpy(validation_target) == 1)
                                )
                                .sum()
                                .item()
                            )

                            # Accumulate TP and FP counts across all batches
                            total_tp += tp
                            total_fp += fp
                            total_fn += fn

                val_loss /= validation_batches
                val_accuracy /= validation_batches

                if total_tp + total_fn > 0:
                    recall_batch = total_tp / (total_tp + total_fn)
                else:
                    recall_batch = 0.0

                if total_tp + total_fp > 0:
                    precision_batch = total_tp / (total_tp + total_fp)
                else:
                    precision_batch = 0.0

                self.logger.info(
                    "Validation loss: {:.2f}, Validation accuracy: {:.2f}, "
                    "Precision failure class: {:.2f}, Recall failure class: {:.2f}, Epoch: {}".format(
                        val_loss, val_accuracy, precision_batch, recall_batch, epoch
                    )
                )

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_precision = precision_batch
                    best_epochs = epoch
                    self.logger.info(
                        "New best validation loss: {}. Saving model to path: {}".format(
                            best_val_loss, save_path
                        )
                    )
                    self.logger.info(
                        "Corresponding precision: {}".format(best_val_precision)
                    )
                    testing_policy.save(filepath=save_path)
                    n_val_epochs_no_improve = 0
                else:
                    n_val_epochs_no_improve += 1

                # if weight_loss:
                #     if precision_batch > best_val_precision:
                #         best_val_loss = val_loss
                #         best_val_precision = precision_batch
                #         best_epochs = epoch
                #         self.logger.info(
                #             "New best validation precision: {}. Saving model to path: {}".format(
                #                 best_val_precision, save_path
                #             )
                #         )
                #         self.logger.info(
                #             "Corresponding validation loss: {}".format(best_val_loss)
                #         )
                #         testing_policy.save(filepath=save_path)
                #         n_val_epochs_no_improve = 0
                #     else:
                #         n_val_epochs_no_improve += 1
                # else:
                #     if val_loss < best_val_loss:
                #         best_val_loss = val_loss
                #         best_val_precision = precision_batch
                #         best_epochs = epoch
                #         self.logger.info(
                #             "New best validation loss: {}. Saving model to path: {}".format(
                #                 best_val_loss, save_path
                #             )
                #         )
                #         self.logger.info(
                #             "Corresponding precision: {}".format(best_val_precision)
                #         )
                #         testing_policy.save(filepath=save_path)
                #         n_val_epochs_no_improve = 0
                #     else:
                #         n_val_epochs_no_improve += 1

            else:
                if len(val_accuracies) > 0:
                    val_accuracies.append(val_accuracies[-1])
                    val_losses.append(val_losses[-1])
                else:
                    val_accuracies.append(0)
                    val_losses.append(train_losses[0])

            if n_val_epochs_no_improve == patience:
                self.logger.info(
                    "Early stopping! No improvement in validation loss for {} epochs".format(
                        n_val_epochs_no_improve
                    )
                )
                break

        if len(validation_dataset.data) == 0:
            self.logger.info("Saving model to path: {}".format(save_path))
            testing_policy.save(filepath=save_path)

        # loading best model and evaluate it on test set if it exists, else evaluate on the validation set
        with torch.no_grad():
            # loading the best policy
            testing_policy.load(filepath=save_path)

            if test_dataset is not None:
                self.logger.info("Evaluating on test dataset")
                test_dataloader = DataLoader(
                    dataset=test_dataset,
                    batch_size=len(test_dataset.data),
                    shuffle=False,
                )
            else:
                self.logger.info("Evaluating on validation dataset")
                test_dataloader = DataLoader(
                    dataset=validation_dataset,
                    batch_size=len(validation_dataset.data),
                    shuffle=False,
                )

            test_data, test_target, weights = next(iter(test_dataloader))

            test_loss, _ = testing_policy.forward_and_loss(
                data=test_data, target=test_target, weights=weights, training=True
            )
            test_loss = test_loss.item()

            logits, predictions = testing_policy.forward_and_loss(
                data=test_data, target=test_target, training=False
            )
            scores = testing_policy.compute_score(logits=logits)

            correct = (predictions == test_target).sum().item()
            self.logger.info("Accuracy test set: {}".format(correct / len(test_target)))

            auc = roc_auc_score(y_true=to_numpy(test_target), y_score=to_numpy(scores))
            precision = precision_score(
                y_true=to_numpy(test_target), y_pred=to_numpy(predictions), pos_label=1
            )
            recall = recall_score(
                y_true=to_numpy(test_target), y_pred=to_numpy(predictions), pos_label=1
            )

            self.logger.info("Target: {}".format(to_numpy(test_target)))
            self.logger.info("Predictions: {}".format(to_numpy(predictions)))
            self.logger.info("Precision: {:.2f}".format(precision))
            self.logger.info("Recall: {:.2f}".format(recall))
            self.logger.info(
                "F-measure: {:.2f}".format(
                    2 * (precision * recall) / (precision + recall)
                    if precision + recall > 0.0
                    else 0.0
                )
            )
            self.logger.info("AUROC: {:.2f}".format(auc))

            test_precision = precision
            test_recall = recall
            auc_roc = auc
            # no cross-validation so plot results
            fpr, tpr, thresholds = roc_curve(
                y_true=to_numpy(test_target), y_score=to_numpy(scores)
            )
            self.plot_roc_curve(fpr=fpr, tpr=tpr)
            if training_progress_filter is not None:
                auc_roc_plot_filename = f"policy-{testing_policy_for_training_name}-{training_progress_filter}-{layers}-roc-auc"
            else:
                auc_roc_plot_filename = (
                    f"policy-{testing_policy_for_training_name}-{layers}-roc-auc"
                )

            if mutant_name is not None:
                if mutant_configuration is not None:
                    auc_roc_plot_filename += f"-{mutant_name}-{mutant_configuration}"
                else:
                    auc_roc_plot_filename += f"-{mutant_name}"

            plt.savefig(
                os.path.join(self.log_path, f"{auc_roc_plot_filename}.png"),
                format="png",
            )

            activations, children = testing_policy.get_activations_and_children(
                data=test_data
            )

            if layers > 1:
                self.visualize_activation_distributions(
                    log_path=self.log_path,
                    plot_filename=f"{auc_roc_plot_filename.replace('-roc-auc', '')}-activations.png",
                    activations=activations,
                    children=children,
                )

            # recreate the dataloader
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

            (
                neurons_dead,
                number_neurons_dead,
            ) = testing_policy.measure_number_dead_neurons(dataloader=train_dataloader)
            self.logger.info(f"Number of dead neurons: {number_neurons_dead}")
            percentage_dead_neurons = [
                f"{(100.0 * num_dead / tens.shape[0]):4.2f}%"
                for tens, num_dead in zip(neurons_dead, number_neurons_dead)
            ]
            self.logger.info(f"In percentage: {', '.join(percentage_dead_neurons)}")

            assert sum(predictions) != len(
                predictions
            ), "Failure predictor always predicts failure; there must be something wrong with the data distribution or the training has gone wrong."
            assert (
                sum(predictions) != 0.0
            ), "Failure predictor always predicts success; there must be something wrong with the data distribution or the training has gone wrong."

            assert (
                recall > 0.05
            ), "Recall is very low, consider training another model and change the hyperparameters (or preprocess the dataset in a different way)"
            assert (
                precision > 0.05
            ), "Precision is very low, consider training another model and change the hyperparameters (or preprocess the dataset in a different way)"

        plt.figure()
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(val_accuracies, label="Validation accuracy")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.figure()
        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Validation loss")
        plt.xlabel("# Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if training_progress_filter is not None:
            loss_plot_filename = f"policy-{testing_policy_for_training_name}-{training_progress_filter}-{layers}-loss.png"
        else:
            loss_plot_filename = (
                f"policy-{testing_policy_for_training_name}-{layers}-loss.png"
            )

        if mutant_name is not None:
            if mutant_configuration is not None:
                loss_plot_filename += f"-{mutant_name}-{mutant_configuration}.png"
            else:
                loss_plot_filename += f"-{mutant_name}.png"

        plt.savefig(os.path.join(self.log_path, loss_plot_filename), format="png")

        return (
            test_loss,
            test_precision,
            test_recall,
            best_epochs,
            auc_roc,
            test_accuracy,
        )

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray):
        plt.figure()
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()

    @staticmethod
    def get_policy_save_path(
        log_path: str,
        training_progress_filter: int,
        testing_policy_for_training_name: str,
        layers: int = 4,
        mutant_name: str = None,
        mutant_configuration: str = None,
    ) -> str:
        if testing_policy_for_training_name in DNN_POLICIES:
            if training_progress_filter is not None:
                save_path = os.path.join(
                    log_path,
                    "best-policy-{}-{}-{}.pkl".format(
                        testing_policy_for_training_name,
                        training_progress_filter,
                        layers,
                    ),
                )
            else:
                save_path = os.path.join(
                    log_path,
                    "best-policy-{}-{}.pkl".format(
                        testing_policy_for_training_name, layers
                    ),
                )

        else:
            raise NotImplementedError(
                "Train policy: {} not supported".format(
                    testing_policy_for_training_name
                )
            )

        if mutant_name is not None:
            if mutant_configuration is not None:
                save_path = f"{save_path.split('.pkl')[0]}-{mutant_name}-{mutant_configuration}.pkl"
            else:
                save_path = f"{save_path.split('.pkl')[0]}-{mutant_name}.pkl"

        return save_path

    # from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/02-activation-functions.html
    @staticmethod
    def visualize_activation_distributions(
        log_path: str,
        plot_filename: str,
        activations: Dict,
        children: List[nn.Module],
        color: str = "C0",
    ) -> None:
        columns = 4
        rows = math.ceil(len(activations) / columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns * 2.7, rows * 2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index // columns][fig_index % columns]
            sns.histplot(
                data=activations[key],
                bins=50,
                ax=key_ax,
                color=color,
                kde=True,
                stat="density",
            )
            key_ax.set_title(f"Layer {key} - {children[key].__class__.__name__}")
            fig_index += 1
        fig.suptitle("Activation distribution", fontsize=14)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(os.path.join(log_path, plot_filename), format="png")
        plt.close()

    @staticmethod
    def load_testing_policy(
        test_policy_for_training_name: str,
        input_size: int,
        load_path: str,
        layers: int = 4,
    ) -> TestingPolicy:
        logger = Log("load_testing_policy")

        if test_policy_for_training_name in DNN_POLICIES:
            testing_policy = get_testing_policy(
                policy_name=test_policy_for_training_name,
                input_size=input_size,
                layers=layers,
            )
            logger.info("Loading test generation model: {}".format(load_path))
            testing_policy.load(filepath=load_path)
            logger.info("Model architecture: {}".format(testing_policy.get_model()))
            return testing_policy

        raise NotImplementedError(
            "Test policy for training {} not supported".format(
                test_policy_for_training_name
            )
        )
