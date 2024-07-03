from typing import Dict, List, cast

import numpy as np
from test_generation.dataset import Dataset
from test_generation.env_configuration import EnvConfiguration

from envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration


class HumanoidDataset(Dataset):
    def __init__(self, policy: str):
        super(HumanoidDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        humanoid_env_configuration = cast(HumanoidEnvConfiguration, env_configuration)
        mapping["qpos"] = list(range(0, len(humanoid_env_configuration.qpos)))
        mapping["qvel"] = list(
            range(
                len(humanoid_env_configuration.qpos),
                len(humanoid_env_configuration.qpos)
                + len(humanoid_env_configuration.qvel),
            )
        )
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        humanoid_env_configuration = cast(HumanoidEnvConfiguration, env_configuration)
        qpos = (
            (humanoid_env_configuration.qpos - humanoid_env_configuration.init_qpos)
            + humanoid_env_configuration.c
        ) / (2 * humanoid_env_configuration.c)
        qvel = (
            (humanoid_env_configuration.qvel - humanoid_env_configuration.init_qvel)
            + humanoid_env_configuration.c
        ) / (2 * humanoid_env_configuration.c)
        return np.concatenate((qpos, qvel), axis=0)

    def get_feature_names(self) -> List[str]:
        # bound to transform_mlp method, in particular the order
        data = self.dataset[0]
        env_config = cast(HumanoidEnvConfiguration, data.training_logs.get_config())
        res = ["qpos_{}".format(i) for i in range(len(env_config.qpos))]
        res.extend(["qvel_{}".format(i) for i in range(len(env_config.qvel))])
        return res

    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        # FIXME: it is normalized, it should be unnormalized
        humanoid_env_configuration = HumanoidEnvConfiguration()
        humanoid_env_configuration.qpos = env_config_transformed[
            : len(humanoid_env_configuration.init_qpos)
        ]
        humanoid_env_configuration.qvel = env_config_transformed[
            len(humanoid_env_configuration.init_qpos) : len(
                humanoid_env_configuration.init_qpos
            )
            + len(humanoid_env_configuration.init_qvel)
        ]
        return humanoid_env_configuration
