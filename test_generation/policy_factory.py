from test_generation.policies.testing_mlp_policy import TestingMlpPolicy
from test_generation.policies.testing_policy import TestingPolicy


def get_testing_policy(
    policy_name: str, input_size: int, layers: int = 4, learning_rate: float = 3e-4
) -> TestingPolicy:
    if policy_name == "mlp":
        return TestingMlpPolicy(
            input_size=input_size, layers=layers, learning_rate=learning_rate
        )
    raise NotImplementedError("Unknown policy: {}".format(policy_name))
