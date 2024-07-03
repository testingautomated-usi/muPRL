import os

import mutators
from properties import mutation_operators

for operator in mutation_operators:
    mutationClass = getattr(mutators, operator["class"])
    # mutationClass.mutate(save_path_prepared, save_path_mutated)
    operator_class = mutationClass()
    print(operator_class)
