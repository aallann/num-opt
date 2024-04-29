import abc
import pandas as pd

from transforms import Transform


class AbstractDataProcessor(abc.ABC):
    """Data processing abstract base class for pre-
    and post-processing data for training/inference"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def preprocess(self, data_frame: pd.DataFrame):
        """Preprocess data"""
        return NotImplementedError

    @abc.abstractmethod
    def postprocess(self, data_frame: pd.DataFrame):
        """Postprocess data"""
        return NotImplementedError


class DataProcessor(AbstractDataProcessor):
    """Data processor for training/inference data

    Args
    ----
        :param input_encoder: input encoder
        :param output_encoder: output encoder
        :param n_features: number of input features
    """

    def __init__(
        self,
        input_encoder: Transform = None,
        output_encoder: Transform = None,
        n_features: int = None,
    ):

        super().__init__()
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.n_features = n_features

    def preprocess(self, data_frame: pd.DataFrame):
        if self.input_encoder:
            data_frame = self.input_encoder.transform(data_frame)

        return data_frame

    def postprocess(self, data_frame: pd.DataFrame):
        if self.output_encoder:
            data_frame = self.output_encoder.inverse_transform(data_frame)

        return data_frame
