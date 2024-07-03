from typing import Dict, List, cast

import numpy as np
from test_generation.dataset import Dataset
from test_generation.env_configuration import EnvConfiguration

from envs.lunarlander.lunarlander_env import INITIAL_RANDOM, H
from envs.lunarlander.lunarlander_env_configuration import LunarLanderEnvConfiguration


class LunarLanderDataset(Dataset):
    def __init__(self, policy: str):
        super(LunarLanderDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        lunar_lander_env_configuration = cast(
            LunarLanderEnvConfiguration, env_configuration
        )
        mapping["height"] = list(range(0, len(lunar_lander_env_configuration.height)))
        mapping["apply_force_to_center"] = list(
            range(
                len(lunar_lander_env_configuration.height),
                len(lunar_lander_env_configuration.height)
                + len(lunar_lander_env_configuration.apply_force_to_center),
            )
        )
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        lunar_lander_env_configuration = cast(
            LunarLanderEnvConfiguration, env_configuration
        )
        height = lunar_lander_env_configuration.height / (H / 2)
        apply_force_to_center_np = np.asarray(
            lunar_lander_env_configuration.apply_force_to_center
        )
        apply_force_to_center = (apply_force_to_center_np + INITIAL_RANDOM) / (
            2 * INITIAL_RANDOM
        )
        return np.concatenate((height, apply_force_to_center), axis=0)

    def get_feature_names(self) -> List[str]:
        # bound to transform_mlp method, in particular the order
        data = self.dataset[0]
        env_config = cast(LunarLanderDataset, data.training_logs.get_config())
        res = ["height_{}".format(i) for i in range(len(env_config.height))]
        res.extend(
            [
                "apply_force_to_center_{}".format(i)
                for i in range(len(env_config.apply_force_to_center))
            ]
        )
        return res

    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        # FIXME: it is normalized, it should be unnormalized
        lunar_lander_env_configuration = LunarLanderEnvConfiguration()
        lunar_lander_env_configuration.height = env_config_transformed[
            : len(lunar_lander_env_configuration.height)
        ]
        lunar_lander_env_configuration.apply_force_to_center = env_config_transformed[
            len(lunar_lander_env_configuration.height) : len(
                lunar_lander_env_configuration.height
            )
            + len(lunar_lander_env_configuration.apply_force_to_center)
        ]
        return lunar_lander_env_configuration
