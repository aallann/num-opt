import abc
import numpy as np
import pandas as pd

from typing import Union


class DictDataset(object):
    """Custom data structure / handler for supervised learning"""

    def __init__(
        self, x: Union[pd.Series, pd.DataFrame], y: Union[pd.Series, pd.DataFrame]
    ):
        assert x.shape[0] == y.shape[0], "x and y must have the same first dimension"
        self.x: np.array = np.array(x)
        self.y: np.array = np.array(y).squeeze()

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        return {"x": x, "y": y}

    def __len__(self):
        return self.x.shape[0]


# DEPRECATED  - use numpy instead, easier handling
class PandasDictDataset(object):
    """Custom data structure / handler for supervised learning"""

    def __init__(self, data: pd.DataFrame):
        assert x.shape[0] == y.shape[0], "x and y must have the same first dimension"
        self.x: pd.Series = data.iloc[:, :-1].values
        self.y: pd.Series = data.iloc[:, -1].values

    def __getitem__(self, idx: int):
        x = self.x.iloc[idx]
        y = self.y[idx]

        return {"x": x, "y": y}

    def __len__(self):
        return self.x.shape[0]


class Callback(object):
    """Callback for storing updates in optimization method"""

    def __init__(self):
        self.state_dict = pd.DataFrame(
            columns=[
                "active_set_size",
                "objective",
                "gradient_norm",
                "w",
                "iter",
            ]
        )

    def update_state_dict(self, **kwargs):
        for key in self.state_dict.columns:
            if key in kwargs:
                self.state_dict.at[self.state_dict.shape[0], key] = kwargs[key]
            else:
                self.state_dict.at[self.state_dict.shape[0], key] = None
