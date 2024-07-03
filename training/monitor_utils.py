from typing import Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME

X_SUCCESS = "success_rate"


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to xs ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes', X_WALLTIME='walltime_hrs' or X_SUCCESS='success_rate')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.l.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    elif x_axis == X_SUCCESS:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.is_success.values
    else:
        raise NotImplementedError

    return x_var, y_var
