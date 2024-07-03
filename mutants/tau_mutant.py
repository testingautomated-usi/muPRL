from typing import Union

import numpy as np

from mutants.mutant import Mutant


class TauMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="tau",
            operator_value=operator_value,
            operator_range_values=(0, 1),
            relative=relative,
            **kwargs,
        )

        self.possible_values = [
            0.0005,
            0.0008,
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
            0.08,
            0.1,
            0.12,
        ]

    def mutate(self) -> float:
        return float(np.random.choice(a=self.possible_values, size=1)[0])
