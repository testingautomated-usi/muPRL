from typing import Union

import numpy as np

from mutants.mutant import Mutant


class NTimestepsMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="n_timesteps",
            operator_value=int(operator_value),
            operator_range_values=(
                1,
                100,
            ),  # 10% of the initial n_timesteps to 100% initial n_timesteps
            relative=relative,
            **kwargs,
        )

        self.possible_values = [
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
        ]

        self.lower_bound = (
            self.original_operator_value * self.operator_range_values[0] / 100
        )

    def mutate(self) -> float:
        percentage_n_timesteps = int(
            np.random.choice(a=self.possible_values, size=1)[0]
        )
        return self.linear_map(
            x=percentage_n_timesteps,
            x_min=self.operator_range_values[0],
            x_max=self.operator_range_values[1],
            new_min=int(
                self.original_operator_value * self.operator_range_values[0] / 100
            ),
            new_max=self.original_operator_value,
        )
