from typing import Union

import numpy as np

from mutants.mutant import Mutant


class LearningStartsMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="learning_starts",
            operator_value=operator_value,
            operator_range_values=(
                0.1,
                100,
            ),  # 0 means no learning_starts, 50 means learning_starts = 50% * n_timesteps
            relative=relative,
            **kwargs,
        )

        assert (
            self.kwargs.get("n_timesteps", None) is not None
        ), "The value of the n_timestep parameter needs to be passed in the kwargs dictionary"
        self.n_timesteps = int(self.kwargs["n_timesteps"])

        self.possible_values = [0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        assert self.original_operator_value <= int(
            self.operator_range_values[1] * self.n_timesteps / 100
        ), f"Max operator range {self.operator_range_values[1]} too small w.r.t. original learning_starts value {self.original_operator_value} and n_timesteps {self.n_timesteps}"

        self.multiplier = 2

    def mutate(self) -> int:
        percentage_learning_starts = np.random.choice(a=self.possible_values, size=1)[0]
        return int(
            self.linear_map(
                x=percentage_learning_starts,
                x_min=self.operator_range_values[0],
                x_max=self.operator_range_values[1],
                new_min=int(self.operator_range_values[0] * self.n_timesteps / 100),
                new_max=int(self.operator_range_values[1] * self.n_timesteps / 100),
            )
        )
