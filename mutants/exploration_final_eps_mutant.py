from typing import Union

import numpy as np

from mutants.mutant import Mutant


class ExplorationFinalEpsMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="exploration_final_eps",
            operator_value=operator_value,
            operator_range_values=(0.0, 1.0),
            relative=relative,
            **kwargs,
        )

        self.clear_challenging_direction = True

        self.multiplier = 2.0

    def mutate(self) -> float:
        return round(
            float(
                np.random.uniform(
                    low=self.operator_range_values[0],
                    high=self.operator_range_values[1],
                )
            ),
            2,
        )
