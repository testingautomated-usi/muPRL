from typing import Union

import numpy as np

from mutants.mutant import Mutant


class GammaMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="gamma",
            operator_value=operator_value,
            operator_range_values=(0.1, 0.9999),
            relative=relative,
            **kwargs,
        )

        self.possible_values = [
            0.45,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            0.9999,
        ]

    def mutate(self) -> float:
        return float(np.random.choice(a=self.possible_values, size=1)[0])
