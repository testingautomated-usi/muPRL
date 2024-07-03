from typing import Union

import numpy as np

from mutants.mutant import Mutant


class NStepsMutant(Mutant):
    # n_steps must be greater than 1 because of advantage normalization (PPO doc)
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="n_steps",
            operator_value=operator_value,
            operator_range_values=(1, 5024),
            relative=relative,
            **kwargs,
        )

        assert (
            self.kwargs.get("n_timesteps", None) is not None
        ), "The value of the n_timestep parameter needs to be passed in the kwargs dictionary"
        n_timesteps = int(self.kwargs["n_timesteps"])

        self.operator_range_values = (self.operator_range_values[0], n_timesteps)
        self.possible_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072]

    def mutate(
        self,
    ) -> int:
        return int(np.random.choice(a=self.possible_values, size=1)[0])
