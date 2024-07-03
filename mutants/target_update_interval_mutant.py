from typing import Union

import numpy as np

from mutants.mutant import Mutant


class TargetUpdateIntervalMutant(Mutant):
    def __init__(self, operator_value: Union[int, float], relative: bool, **kwargs):
        super().__init__(
            operator_name="target_update_interval",
            operator_value=operator_value,
            operator_range_values=(0, 100),  # 100 means never update
            relative=relative,
            **kwargs,
        )

        # FIXME: replace mutant name with global constant
        assert (
            self.kwargs.get("n_timesteps", None) is not None
        ), "The value of the n_timestep parameter needs to be passed in the kwargs dictionary"
        self.n_timesteps = int(self.kwargs["n_timesteps"])

        self.possible_values = [0.5, 1, 5, 10, 15, 20, 25, 30]

        assert self.original_operator_value <= int(
            self.operator_range_values[1] * self.n_timesteps / 100
        ), f"Max operator range {self.operator_range_values[1]} too small w.r.t. original target_updat_interval value {self.original_operator_value} and n_timesteps {self.n_timesteps}"

    def mutate(self) -> int:
        percentage_target_interval = int(
            np.random.choice(a=self.possible_values, size=1)[0]
        )
        return int(
            self.linear_map(
                x=percentage_target_interval,
                x_min=self.operator_range_values[0],
                x_max=self.operator_range_values[1],
                new_min=1,
                new_max=int(self.operator_range_values[1] * self.n_timesteps / 100),
            )
        )
