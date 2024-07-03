from typing import Dict, List, cast

import numpy as np
from test_generation.dataset import Dataset
from test_generation.env_configuration import EnvConfiguration

from envs.parking.parking_env_configuration import ParkingEnvConfiguration


class ParkingDataset(Dataset):
    def __init__(self, policy: str):
        super(ParkingDataset, self).__init__(policy=policy)

    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        mapping = dict()
        mapping = dict()
        mapping["goal_lane_idx"] = [0]
        mapping["heading_ego"] = [1]
        mapping["position_ego"] = [2, 3]
        return mapping

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        parking_env_configuration = cast(ParkingEnvConfiguration, env_configuration)

        transformed = np.asarray(
            [
                float(
                    parking_env_configuration.goal_lane_idx
                    / parking_env_configuration.num_lanes
                ),
                parking_env_configuration.heading_ego,
                (
                    parking_env_configuration.position_ego[0]
                    + parking_env_configuration.limit_x_position
                )
                / (2 * parking_env_configuration.limit_x_position),
                (
                    parking_env_configuration.position_ego[1]
                    + parking_env_configuration.limit_y_position
                )
                / (2 * parking_env_configuration.limit_y_position),
            ]
        )
        return transformed

    def get_feature_names(self) -> List[str]:
        res = ["goal_lane", "h", "pos_x", "pos_y"]
        return res

    def get_original_env_configuration(
        self, env_config_transformed: np.ndarray
    ) -> EnvConfiguration:
        # FIXME: it is normalized, it should be unnormalized
        parking_env_configuration = ParkingEnvConfiguration()
        parking_env_configuration.position_ego = (
            env_config_transformed[-2],
            env_config_transformed[-1],
        )
        parking_env_configuration.heading_ego = env_config_transformed[-3]
        parking_env_configuration.goal_lane_idx = env_config_transformed[-4]
        return parking_env_configuration
