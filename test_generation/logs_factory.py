from typing import Dict, List, Union

from config import CARTPOLE_ENV_NAME
from envs.cartpole.cartpole_training_logs import CartPoleTrainingLogs

from test_generation.env_configuration_factory import make_env_configuration
from test_generation.testing_logs import TestingLogs
from test_generation.training_logs import TrainingLogs


def make_log(
    env_name: str,
    log_type: str,
    env_config: Union[str, Dict],
    data: Dict,
    dynamic_info: List = None,
) -> Union[TrainingLogs, TestingLogs]:
    assert log_type in ["training", "testing"], f"Unknown log_type: {log_type}"
    if env_name == CARTPOLE_ENV_NAME:
        if log_type == "training":
            logs = CartPoleTrainingLogs(
                config=make_env_configuration(env_name=env_name, env_config=env_config),
                **data,
            )
        elif log_type == "testing":
            assert dynamic_info is not None, "Dynamic info must not be None"
            logs = TestingLogs(
                config=make_env_configuration(env_name=env_name, env_config=env_config),
                dynamic_info=dynamic_info,
            )
        else:
            raise RuntimeError(f"Unknown log_type: {log_type}")
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))

    return logs
