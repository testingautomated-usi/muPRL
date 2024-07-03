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

from base_args import BaseArgs
from mutants.utils import find_all_mutation_operators
from rl_zoo3.utils import StoreDict
from training.training_type import TrainingType


class TrainingArgs(BaseArgs):
    """
    This class includes custom options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = BaseArgs.initialize(self, parser=parser)
        parser.add_argument(
            "--register-env",
            action="store_true",
            default=False,
            help="Register env with its args and override existing one",
        )
        parser.add_argument(
            "--log-success",
            action="store_true",
            default=False,
            help="Log success in the monitor file during training",
        )
        parser.add_argument(
            "--test-generation",
            action="store_true",
            default=False,
            help="Enable test generation (enables logging of initial configurations during training)",
        )
        parser.add_argument(
            "--eval-env",
            action="store_true",
            default=False,
            help='Enable creation of evaluation environment. The difference with "--eval-freq" is that '
            "in this case the frequency is decided automatically, i.e., 10% of the total "
            "training timesteps.",
        )
        parser.add_argument(
            "--custom-env-kwargs",
            type=str,
            nargs="+",
            action=StoreDict,
            help="Custom keyword argument to pass to the env constructor",
            default=None,
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
            "--num-runs", type=int, help="Number of runs for the agent", default=1
        )
        parser.add_argument(
            "--num-cpus",
            type=int,
            help="Number of cpus to be used for parallelization (default = -1, i.e., the number of logical cores in the current machine)",
            default=-1,
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
            choices=find_all_mutation_operators(),
            help="Mutation operator name (see package 'mutants'; each mutant is named <name>_mutant.py). "
            "This flag is only considered when 'training_type' == 'mutant'",
            default=None,
        )
        parser.add_argument(
            "--search-budget",
            type=int,
            help="Search budget for the mutation in terms of number of mutations",
            default=1,
        )
        parser.add_argument(
            "--search-iteration",
            type=int,
            help="It is meant to parallelize "
            "the execution of different mutants on multiple machines. It should be < search-budget (starts from 0)",
            default=-1,
        )
        parser.add_argument(
            "--run-num",
            type=int,
            help="Only works with parallelization disabled. It is meant to parallelize "
            "the execution of different runs on multiple machines. It should be < num_runs (starts from 0)",
            default=-1,
        )
        parser.add_argument(
            "--mock",
            action="store_true",
            help="Mock the execution (only works parallelization disabled)",
            default=False,
        )
        parser.add_argument(
            "--parallelize",
            action="store_true",
            help="Parallelize training runs",
            default=False,
        )
        return parser
