from typing import Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

Scaler = Union[MinMaxScaler, StandardScaler]

Observation = np.ndarray
