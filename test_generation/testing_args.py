"""
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


--------------------------- LICENSE FOR pix2pix --------------------------------
BSD License

For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

----------------------------- LICENSE FOR DCGAN --------------------------------
BSD License

For dcgan.torch software

Copyright (c) 2015, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name Facebook nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
from typing import Union

from config import ENV_NAMES
from mutants.utils import find_all_mutation_operators
from rl_zoo3 import ALGOS
from rl_zoo3.utils import StoreDict
from training.training_type import TrainingType

from test_generation.test_generation_config import (
    CLASSIFIER_LAYERS,
    SAMPLING_SIZE,
    TESTING_POLICIES_FOR_TRAINING,
    TESTING_STRATEGIES,
)

algos_list = list(ALGOS.keys())


class TestingArgs:
    """
    This class defines options used by rl_zoo3.

    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "-f", "--folder", help="Log folder", type=str, default="logs"
        )
        parser.add_argument(
            "--algo",
            help="RL Algorithm",
            default="sac",
            type=str,
            required=False,
            choices=algos_list,
        )
        parser.add_argument(
            "--seed", help="Random generator seed", type=int, default=-1
        )
        parser.add_argument(
            "--exp-id", help="Experiment ID (0: latest)", default=0, type=int
        )
        parser.add_argument(
            "--env-name",
            help="Env name",
            type=str,
            choices=ENV_NAMES,
            default="CartPole-v1",
        )

        parser.add_argument(
            "--num-episodes",
            help="Num episodes (i.e. num of env configurations) to run",
            type=int,
            default=-1,
        )
        parser.add_argument(
            "--num-runs-each-env-config",
            help="Num runs for each env configuration (valid when failure probability dist = True)",
            type=int,
            default=30,
        )

        # Policy params
        parser.add_argument(
            "--testing-strategy-name",
            help="Test strategy",
            type=str,
            choices=TESTING_STRATEGIES,
            default="random",
        )
        parser.add_argument(
            "--testing-policy-for-training-name",
            help="Testing policy for training",
            type=str,
            choices=TESTING_POLICIES_FOR_TRAINING,
            default="mlp",
        )
        parser.add_argument(
            "--budget",
            help="Timeout in seconds for the failure search technique",
            type=int,
            default=-1,
        )
        parser.add_argument(
            "--training-progress-filter",
            help="Percentage of training to filter",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--layers",
            help="Num layers architecture",
            type=int,
            choices=CLASSIFIER_LAYERS,
            default=1,
        )
        parser.add_argument(
            "--model-checkpoint",
            help="Model checkpoint to load (valid when estimate failure probability = True)",
            type=int,
            default=-1,
        )
        parser.add_argument(
            "--sampling-size",
            help="Sampling size when testing-strategy-name == 'nn'. It is disabled if budget != -1.",
            type=int,
            default=SAMPLING_SIZE,
        )

        parser.add_argument(
            "--render",
            action="store_true",
            default=False,
            help="Render the environment",
        )
        parser.add_argument(
            "--num-runs-experiments",
            help="Number of times to run the experiments",
            type=int,
            default=1,
        )

        # env_kwargs
        parser.add_argument(
            "--register-env",
            action="store_true",
            default=False,
            help="Register env with its args and override existing one",
        )
        parser.add_argument(
            "--wrapper-kwargs",
            type=str,
            nargs="*",
            action=StoreDict,
            help="Wrapper keyword arguments",
            default=None,
        )

        parser.add_argument(
            "--test-multiple-agents",
            action="store_true",
            help="Test multiple agents on the same environment "
            "in a common directory obtained by training an agent multiple times",
            default=False,
        )

        parser.add_argument(
            "--training-type",
            type=str,
            choices=[training_type.name for training_type in TrainingType],
            help=f"Type of training, i.e., {[training_type.name for training_type in TrainingType]}",
            default=TrainingType.original.name,
        )
        parser.add_argument(
            "--mutant-name",
            type=str,
            choices=find_all_mutation_operators() + [None, ""],
            help="Mutation operator name (see package 'mutants'; each mutant is named <name>_mutant.py). "
            "This flag is only considered when 'training_type' == 'mutant'",
            default=None,
        )
        parser.add_argument(
            "--mutant-configuration",
            type=str,
            help="Configuration of the mutant, if any",
            default=None,
        )

        parser.add_argument(
            "--mode",
            type=str,
            choices=["replay", "weak", "strong", "weaker"],
            help="Type of testing mode when 'mutant-name' is not None",
            default="replay",
        )
        parser.add_argument(
            "--parallelize",
            action="store_true",
            help="Parallelize testing runs",
            default=False,
        )
        parser.add_argument(
            "--delete-mutants-after-replay",
            action="store_true",
            help="If True, it deletes unpromising mutants after replay.",
            default=False,
        )

        # FIXME: duplicated with TrainingArgs
        parser.add_argument(
            "--run-num",
            type=int,
            help="Only works with parallelization disabled. It is meant to parallelize "
            "the execution of different runs (of replay) on multiple machines. It should be < num_runs (starts from 0)",
            default=-1,
        )

        # FIXME: duplicated with evaluate and train.py (test_generation)
        parser.add_argument(
            "--threshold-failure",
            type=float,
            help="Threshold of failure probability after which the episode is considered a failure (this holds for deterministic and non-deterministic environments)",
            default=0.5,
        )
        # FIXME: duplicated with TrainingArgs
        parser.add_argument(
            "--num-cpus",
            type=int,
            help="Number of cpus to be used for parallelization (default = -1, i.e., the number of logical cores in the current machine)",
            default=-1,
        )
        # FIXME: duplicated with train.py
        parser.add_argument(
            "--do-not-encode-run-and-conf",
            help="Do not encode the run number and mutant value as input of the failure predictor",
            action="store_true",
            default=False,
        )
        # FIXME: duplicated with BaseArgs
        parser.add_argument(
            "--device",
            help="PyTorch device to be use (ex: cpu, cuda...)",
            default="cpu",
            type=str,
        )

        self.initialized = True
        return parser

    def gather_args(self) -> argparse.Namespace:
        """
        Initialize our parser with basic args (only once).
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_args(self, args: argparse.Namespace) -> None:
        """
        Print and save args

        It will print both current args and default values (if different).
        """
        message = ""

        message += "-" * 40 + " Args " + "-" * 40 + "\n"
        for k, v in sorted(vars(args).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "-" * 40 + " End " + "-" * 40 + "\n"
        print(message)

    def parse(self) -> argparse.Namespace:
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        args = self.gather_args()

        self.print_args(args)

        self.args = args
        return self.args
