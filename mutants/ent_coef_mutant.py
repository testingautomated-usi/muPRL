from typing import Union

import numpy as np

from mutants.mutant import Mutant


class EntCoefMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="ent_coef",
            operator_value=operator_value,
            operator_range_values=(
                0.0,
                20.0,
            ),  # original maximum is 0.1
            relative=relative,
            **kwargs,
        )

        self.lower_bound_if_zero = 0.01
        self.low = (
            np.log(self.operator_range_values[0])
            if self.operator_range_values[0] > 0.0
            else np.log(self.lower_bound_if_zero)
        )
        self.high = np.log(0.2)

    def mutate(self) -> float:
        ent_coef_sample = round(np.exp(np.random.uniform(self.low, self.high)), 2)
        while ent_coef_sample == 0.0:
            ent_coef_sample = round(np.exp(np.random.uniform(self.low, self.high)), 2)
        return ent_coef_sample
