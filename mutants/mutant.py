from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class Mutant(ABC):
    """
    The kwargs parameters are useful in case an operator needs to rely on other operators' values
    """

    def __init__(
        self,
        operator_name: str,
        operator_value: Union[int, float],
        operator_range_values: Union[Tuple[int, int], Tuple[float, float]],
        relative: bool,
        **kwargs,
    ):
        self.operator_name = operator_name
        self.operator_value = operator_value
        self.original_operator_value = operator_value
        self.operator_range_values = operator_range_values
        self.kwargs = kwargs
        if not relative:
            # We do not check the range for relative operators
            assert (
                self.operator_range_values[0] < self.operator_range_values[1]
            ), f"Not valid range: {self.operator_range_values}. Lower bound is >= than upper bound."
            assert (
                self.operator_range_values[0]
                <= self.operator_value
                <= self.operator_range_values[1]
            ), f"Operator value {self.operator_value} not in range {self.operator_range_values}"

    @staticmethod
    def linear_map(x: int, x_min: int, x_max: int, new_min: int, new_max: int) -> float:
        # Perform the linear mapping
        mapped_value = (x - x_min) * (new_max - new_min) / (x_max - x_min) + new_min
        return mapped_value

    @abstractmethod
    def mutate(self) -> List[Union[int, float]]:
        pass
