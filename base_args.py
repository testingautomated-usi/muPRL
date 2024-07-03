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

from config import CARTPOLE_ENV_NAME, ENV_NAMES
from rl_zoo3 import ALGOS
from rl_zoo3.utils import StoreDict

algos_list = list(ALGOS.keys())


class BaseArgs:
    """
    This class defines options used by rl_zoo3.

    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--algo",
            help="RL Algorithm",
            default="ppo",
            type=str,
            required=False,
            choices=algos_list,
        )
        parser.add_argument(
            "--env",
            type=str,
            default=CARTPOLE_ENV_NAME,
            choices=ENV_NAMES,
            help="environment ID",
        )
        parser.add_argument(
            "-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str
        )
        parser.add_argument(
            "-i",
            "--trained-agent",
            help="Path to a pretrained agent to continue training",
            default="",
            type=str,
        )
        parser.add_argument(
            "--truncate-last-trajectory",
            help="When using HER with online sampling the last trajectory "
            "in the replay buffer will be truncated after reloading the replay buffer.",
            default=True,
            type=bool,
        )
        parser.add_argument(
            "-n",
            "--n-timesteps",
            help="Overwrite the number of timesteps",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--num-threads",
            help="Number of threads for PyTorch (-1 to use default)",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--log-interval",
            help="Override log interval (default: -1, no change)",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--eval-freq",
            help="Evaluate the agent every n steps (if negative, no evaluation). "
            "During hyperparameter optimization n-evaluations is used instead",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--optimization-log-path",
            help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
            "Disabled if no argument is passed.",
            type=str,
        )
        parser.add_argument(
            "--eval-episodes",
            help="Number of episodes to use for evaluation",
            default=5,
            type=int,
        )
        parser.add_argument(
            "--n-eval-envs",
            help="Number of environments for evaluation",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--save-freq",
            help="Save the model every n steps (if negative, no checkpoint)",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--save-replay-buffer",
            help="Save the replay buffer too (when applicable)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-f", "--log-folder", help="Log folder", type=str, default="logs"
        )
        parser.add_argument(
            "--seed", help="Random generator seed", type=int, default=-1
        )
        parser.add_argument(
            "--vec-env",
            help="VecEnv type",
            type=str,
            default="dummy",
            choices=["dummy", "subproc"],
        )
        parser.add_argument(
            "--device",
            help="PyTorch device to be use (ex: cpu, cuda...)",
            default="cpu",
            type=str,
        )
        parser.add_argument(
            "--n-trials",
            help="Number of trials for optimizing hyperparameters. "
            "This applies to each optimization runner, not the entire optimization process.",
            type=int,
            default=500,
        )
        parser.add_argument(
            "--max-total-trials",
            help="Number of (potentially pruned) trials for optimizing hyperparameters. "
            "This applies to the entire optimization process and takes precedence over --n-trials if set.",
            type=int,
            default=None,
        )
        parser.add_argument(
            "-optimize",
            "--optimize-hyperparameters",
            action="store_true",
            default=False,
            help="Run hyperparameters search",
        )
        parser.add_argument(
            "--no-optim-plots",
            action="store_true",
            default=False,
            help="Disable hyperparameter optimization plots",
        )
        parser.add_argument(
            "--n-jobs",
            help="Number of parallel jobs when optimizing hyperparameters",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--sampler",
            help="Sampler to use when optimizing hyperparameters",
            type=str,
            default="tpe",
            choices=["random", "tpe", "skopt"],
        )
        parser.add_argument(
            "--pruner",
            help="Pruner to use when optimizing hyperparameters",
            type=str,
            default="median",
            choices=["halving", "median", "none"],
        )
        parser.add_argument(
            "--n-startup-trials",
            help="Number of trials before using optuna sampler",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--n-evaluations",
            help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization."
            "Default is 1 evaluation per 100k timesteps.",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--storage",
            help="Database storage path if distributed optimization should be used",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--study-name",
            help="Study name for distributed optimization",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--verbose",
            help="Verbose mode (0: no output, 1: INFO)",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--gym-packages",
            type=str,
            nargs="+",
            default=[],
            help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
        )
        parser.add_argument(
            "--env-kwargs",
            type=str,
            nargs="+",
            action=StoreDict,
            help="Optional keyword argument to pass to the env constructor",
        )
        parser.add_argument(
            "-params",
            "--hyperparams",
            type=str,
            nargs="+",
            action=StoreDict,
            help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
        )
        parser.add_argument(
            "-yaml",
            "--yaml-file",
            type=str,
            default=None,
            help="Custom yaml file from which the hyperparameters will be loaded",
        )
        parser.add_argument(
            "-uuid",
            "--uuid",
            action="store_true",
            default=False,
            help="Ensure that the run has a unique ID",
        )
        parser.add_argument(
            "--track",
            action="store_true",
            default=False,
            help="if toggled, this experiment will be tracked with Weights and Biases",
        )
        parser.add_argument(
            "--wandb-project-name",
            type=str,
            default="sb3",
            help="the wandb's project name",
        )
        parser.add_argument(
            "--wandb-entity",
            type=str,
            default=None,
            help="the entity (team) of wandb's project",
        )
        parser.add_argument(
            "-P",
            "--progress",
            action="store_true",
            default=False,
            help="if toggled, display a progress bar using tqdm and rich",
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
