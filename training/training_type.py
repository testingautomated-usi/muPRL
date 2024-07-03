from enum import Enum


class TrainingType(Enum):
    original = 1
    mutant = 2
    predictor = 3

    @staticmethod
    def parse(name_as_str: str) -> "TrainingType":
        if name_as_str == "original":
            return TrainingType.original
        if name_as_str == "mutant":
            return TrainingType.mutant
        if name_as_str == "predictor":
            return TrainingType.predictor

        raise NotImplementedError(f"{name_as_str} not supported")
