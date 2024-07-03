import glob
import importlib
import os
from typing import List

from env_variables import PROJECT_ROOT

from mutants.mutant import Mutant


def find_all_mutation_operators() -> List[str]:
    all_files = glob.glob(os.path.join(PROJECT_ROOT, "mutants", "*_mutant.py"))
    return list(
        map(
            lambda filepath: filepath[
                filepath.rindex(os.sep) + 1 : filepath.rindex("_")
            ],
            all_files,
        )
    )


def load_mutant(mutant_name: str) -> Mutant:
    filepath = f"mutants.{mutant_name}_mutant"

    if "_" in mutant_name:
        mutant_name = "".join(
            split_name.capitalize() for split_name in mutant_name.split("_")
        )
    else:
        mutant_name = mutant_name.capitalize()

    class_name = f"{mutant_name}Mutant"

    module = importlib.import_module(filepath)

    class_obj = getattr(module, class_name)
    return class_obj
