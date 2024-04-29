import abc
import numpy as np
import pandas as pd

from typing import List, Union


class Transform(abc.ABC):
    """Transform base class"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def transform(self, data_frame: pd.DataFrame):
        """Forward transform"""
        pass

    @abc.abstractmethod
    def inverse_transform(self, data_frame: pd.DataFrame):
        """Inverse transform"""
        pass


class Composite(Transform):
    """Composite transform for multiprocess data transforms

    Input is a list of transformations, must be instantiations
    of the Transform class.

    Args
    ----
        :param transforms: transformations pipeline
    """

    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transfroms"""
        for tform in self.transforms:
            data_frame = tform.transform(data_frame)

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms"""
        for tform in reversed(self.transforms):
            data_frame = tform.inverse_transform(data_frame)

        return data_frame


class UnitGaussianNormalizer(Transform):
    """Normalizes data to unit Gaussian

    Args
    ----
        :param dims: dimensional discriminator; since dataframe
            dims is a list of strings corresponding to columns
    """

    def __init__(self, dims: list = None):
        super().__init__()
        self.dims = dims
        self.mean: dict = {}
        self.std: dict = {}

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform"""
        if not self.dims:
            self.dims = data_frame.columns
        for dim in self.dims:
            values: np.array = np.array(data_frame[dim].values)
            self.mean[dim] = np.mean(values)
            self.std[dim] = np.std(values)
            data_frame[dim] = (values - self.mean[dim]) / self.std[dim]
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform"""
        for dim in self.dims:
            values: np.array = np.array(data_frame[dim].values)
            data_frame[dim] = values * self.std[dim] + self.mean[dim]
        return data_frame
