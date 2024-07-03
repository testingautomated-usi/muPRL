import importlib
from typing import Dict, Union

from test_generation.env_configuration import EnvConfiguration


# FIXME: duplication with load_generator method in test_generator class
def make_env_configuration(
    env_name: str, env_config: Union[Dict, str]
) -> EnvConfiguration:
    env_name = env_name.split("-")[0]

    env_filename = f"envs.{env_name.lower()}.{env_name.lower()}_env_configuration"

    envlib = importlib.import_module(env_filename)

    target_env_name = env_name.replace("_", "") + "EnvConfiguration"

    for name, cls in envlib.__dict__.items():
        if name.lower() == target_env_name.lower() and issubclass(
            cls, EnvConfiguration
        ):
            if isinstance(env_config, Dict):
                env_configuration = cls(**env_config)
            elif isinstance(env_config, str):
                env_configuration = cls().str_to_config(s=env_config)
            else:
                raise RuntimeError(f"Unknown type {env_config}")
            return env_configuration

    raise RuntimeError(
        "In %s.py, there should be a subclass of EnvConfiguration with class name that matches %s in lowercase."
        % (env_filename, target_env_name)
    )
