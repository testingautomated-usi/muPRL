import time
import unittest

import numpy as np
from config import CARTPOLE_ENV_NAME, ENV_NAMES
from envs.cartpole.cartpole_env_configuration import CartPoleEnvConfiguration
from joblib import Parallel, delayed
from randomness_utils import set_random_seed


def generate_random_configs(
    env_name: str, seed: int, num_configs: int = 100, parallelize: bool = False
) -> np.ndarray:
    assert env_name in ENV_NAMES, f"Env name {env_name} should be in [{ENV_NAMES}]"

    set_random_seed(seed=seed, reset_random_gen=True)

    if parallelize:
        # simulate execution
        time.sleep(2)

    if env_name == CARTPOLE_ENV_NAME:
        return np.asarray(
            [
                CartPoleEnvConfiguration().generate_configuration()
                for _ in range(num_configs)
            ]
        )

    raise NotImplementedError(f"Env name {env_name} not supported")


class TestSeedConfigurations(unittest.TestCase):
    def test_cartpole_configurations(self):
        configs_1 = generate_random_configs(env_name=CARTPOLE_ENV_NAME, seed=0)
        configs_2 = generate_random_configs(env_name=CARTPOLE_ENV_NAME, seed=0)
        assert np.alltrue(configs_1 == configs_2)

    def test_parallel_cartpole_configurations(self):
        with Parallel(
            n_jobs=-1, batch_size="auto", backend="multiprocessing"
        ) as parallel:
            configs_1, configs_2 = parallel(
                (
                    delayed(generate_random_configs)(
                        env_name=CARTPOLE_ENV_NAME, seed=0, parallelize=True
                    )
                    for _ in range(2)
                ),
            )

        assert np.alltrue(configs_1 == configs_2)


if __name__ == "__main__":
    unittest.main()
