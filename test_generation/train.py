import argparse
import copy
import glob
import logging
import os
import time

import numpy as np
from config import CARTPOLE_ENV_NAME, ENV_NAMES
from log import Log
from mutants.utils import find_all_mutation_operators
from randomness_utils import set_random_seed
from stable_baselines3.common.utils import get_latest_run_id
from training.training_type import TrainingType

from test_generation.dataset import Dataset, TorchDataset
from test_generation.failure_predictor import FailurePredictor
from test_generation.preprocessor import preprocess_data
from test_generation.test_generation_config import (
    CLASSIFIER_LAYERS,
    DNN_POLICIES,
    TESTING_POLICIES_FOR_TRAINING,
)
from test_generation.utils.env_utils import ALGOS

algos_list = list(ALGOS.keys())

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=algos_list,
)
parser.add_argument(
    "--env-name",
    type=str,
    default=CARTPOLE_ENV_NAME,
    choices=ENV_NAMES,
    help="environment ID",
)
parser.add_argument("--exp-id", help="Experiment ID (0: latest)", default=0, type=int)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument(
    "--test-split",
    help="Percentage of data reserved for testing",
    type=float,
    default=0.2,
)
parser.add_argument(
    "--testing-policy-for-training-name",
    help="Testing policy for training",
    type=str,
    choices=TESTING_POLICIES_FOR_TRAINING,
)
parser.add_argument(
    "--training-progress-filter",
    help="Percentage of training to filter",
    type=int,
    default=None,
)
parser.add_argument(
    "--n-epochs", help="Number of epochs to train DNN policy", type=int, default=20
)
parser.add_argument(
    "--learning-rate",
    help="Learning rate to train DNN policy",
    type=float,
    default=3e-4,
)
parser.add_argument(
    "--batch-size", help="Batch size to train DNN policy", type=int, default=64
)
parser.add_argument(
    "--weight-decay",
    help="Weight decay for optimizer when training DNN policy",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--weight-loss",
    help="Whether to use a simple weight loss scheme",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--train-from-multiple-runs",
    help="Whether to train a predictor using data from multiple training runs of the same agent",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--mutant-name",
    type=str,
    choices=find_all_mutation_operators(),
    help=f"Mutation operator name (see package 'mutants'; each mutant is named <name>_mutant.py). "
    f"This flag is only considered when 'train-from-multiple-runs' is active. It trains the "
    f"predictor using the data from all the mutants and the data from the original agent",
    default=None,
)
parser.add_argument(
    "--mutant-configuration",
    type=str,
    help="Configuration of the mutant, if any",
    default=None,
)
parser.add_argument(
    "--use-test-set",
    help="Whether to use the test set",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--undersample",
    help="Undersample majority class. 0.5 means that the two classes will be balanced "
    "after undersampling; by default there is no undersampling",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--patience",
    help="Early stopping patience (# of epochs of no improvement) when training AVF DNN",
    type=int,
    default=20,
)
parser.add_argument(
    "--layers",
    help="Num layers architecture",
    type=int,
    choices=CLASSIFIER_LAYERS,
    default=1,
)
parser.add_argument(
    "--do-not-encode-run-and-conf",
    help="Do not encode the run number and mutant value as input of the failure predictor",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--percentage-failure-discard",
    type=float,
    help="Percentage of failures in testing mode above which the configurations are discarded",
    default=0.9,
)
# FIXME: duplicated with evaluate and testing_args
parser.add_argument(
    "--threshold-failure",
    type=float,
    help="Threshold of failure probability after which the episode is considered a failure (this holds for deterministic and non-deterministic environments)",
    default=0.5,
)
args = parser.parse_args()

if __name__ == "__main__":
    algo = args.algo
    folder = args.folder

    encode_run_and_conf = not args.do_not_encode_run_and_conf

    uuid_str = ""

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(args.seed)

    log_path = f"{folder}/{algo}/"

    save_paths = []

    if args.train_from_multiple_runs:
        save_path = f"{log_path}{args.env_name}_{TrainingType.original.name}"
        original_save_path = save_path
        save_paths.append(save_path)
        if args.mutant_name is not None:
            if args.mutant_configuration is not None:
                mutants_folders = [
                    os.path.join(
                        log_path,
                        f"{args.env_name}_mutant_{args.mutant_name}_{args.mutant_configuration}",
                    )
                ]
                assert os.path.exists(
                    mutants_folders[0]
                ), f"Mutant folder {mutants_folders[0]} does not exist in {log_path}"
            else:
                mutants_folders = glob.glob(
                    os.path.join(
                        log_path, f"{args.env_name}_mutant_{args.mutant_name}_*"
                    )
                )
                assert (
                    len(mutants_folders) > 0
                ), f"Mutants folder of mutant {args.mutant_name} do not exist in {log_path}"

            save_paths = copy.deepcopy(mutants_folders)
    else:
        exp_id = (
            get_latest_run_id(log_path, args.env_name) + 1
            if args.exp_id == 0
            else args.exp_id
        )
        save_path = os.path.join(log_path, f"{args.env_name}_{exp_id}{uuid_str}")
        original_save_path = save_path
        save_paths.append(save_path)

    # save failure predictor on original agent folder
    failure_predictor = FailurePredictor(
        env_name=args.env_name, log_path=original_save_path
    )

    logger = Log("failure_predictor_train")
    log_filename = (
        f"failure-predictor-{args.testing_policy_for_training_name}-seed-{args.seed}"
    )

    if args.testing_policy_for_training_name in DNN_POLICIES:
        if args.training_progress_filter is not None:
            log_filename += "-{}-{}".format(args.training_progress_filter, args.layers)
        else:
            log_filename += "-{}".format(args.layers)

    if args.mutant_name is not None:
        log_filename += f"-{args.mutant_name}"
        if args.mutant_configuration is not None:
            log_filename += f"-{args.mutant_configuration}"

    # save logs on original agent folder
    logging.basicConfig(
        filename=os.path.join(original_save_path, "{}.txt".format(log_filename)),
        filemode="w",
        level=logging.DEBUG,
    )

    logger.info("Args: {}".format(args))
    start_time = time.time()

    dataset = preprocess_data(
        env_name=args.env_name,
        log_paths=save_paths,
        training_progress_filter=args.training_progress_filter,
        policy_name=args.testing_policy_for_training_name,
        train_from_multiple_runs=args.train_from_multiple_runs,
        threshold_failure=args.threshold_failure,
        percentage_failure_discard=args.percentage_failure_discard,
    )

    (
        train_data,
        train_labels,
        test_validation_data,
        test_validation_labels,
    ) = dataset.transform_data(
        test_split=args.test_split,
        seed=args.seed,
        shuffle=True,
        undersample_majority_class=args.undersample,
        encode_run_and_conf=encode_run_and_conf,
        mutant_name=args.mutant_name,
        log_path=log_path,
    )
    if args.use_test_set:
        (
            validation_data,
            validation_labels,
            test_data,
            test_labels,
        ) = Dataset.split_train_test(
            test_split=0.5,
            data=test_validation_data,
            labels=test_validation_labels,
            seed=args.seed,
            shuffle=True,
        )

    if args.testing_policy_for_training_name in DNN_POLICIES:
        train_dataset = TorchDataset(
            data=train_data, labels=train_labels, weight_loss=args.weight_loss
        )
        if args.use_test_set:
            validation_dataset = TorchDataset(
                data=validation_data,
                labels=validation_labels,
                weight_loss=args.weight_loss,
            )
            test_dataset = TorchDataset(
                data=test_data, labels=test_labels, weight_loss=args.weight_loss
            )
        else:
            validation_dataset = TorchDataset(
                data=test_validation_data,
                labels=test_validation_labels,
                weight_loss=args.weight_loss,
            )
            test_dataset = None

        (
            test_loss,
            test_precision,
            test_recall,
            best_epochs,
            auc_roc,
            test_accuracy,
        ) = failure_predictor.train_dnn(
            input_size=dataset.get_num_features(
                encode_run_and_conf=encode_run_and_conf,
                mutant_name=args.mutant_name,
                log_path=log_path,
            ),
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            testing_policy_for_training_name=args.testing_policy_for_training_name,
            n_epochs=args.n_epochs,
            layers=args.layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            batch_size=args.batch_size,
            training_progress_filter=args.training_progress_filter,
            weight_loss=args.weight_loss,
            mutant_name=args.mutant_name,
            mutant_configuration=args.mutant_configuration,
        )
    else:
        raise NotImplementedError(
            "testing_policy_for_training_name {} not supported".format(
                args.test_policy_for_training_name
            )
        )

    logger.info("Time elapsed: {}s".format(time.time() - start_time))
