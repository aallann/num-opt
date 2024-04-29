from pathlib import Path
import numpy as np
import pandas as pd

from data_processor import DataProcessor
from transforms import UnitGaussianNormalizer
from utils import DictDataset


def load_breast_cancer_data(
    input_encoder: bool = False,
    output_encoder: bool = False,
    data_path: str = "data",
    saver: bool = False,
    verbose: bool = False,
):
    """loads pre- / post-processed breast cancer data

    Args
    ----
        :param input_encoder: input encoder
        :param output_encoder: output encoder
        :param data_path: path to data
        :param saver: save proecessed data, False by default
        :param verbose: verbose flag

    Returns
    -------
        :return: train_loader, test_loader, data_processor
    """

    names: list = [
        "id",
        "clump_thickness",
        "uniformity_cell_size",
        "uniformity_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
        "class",
    ]

    DIR: Path = Path(__file__).parent

    data: pd.DataFrame = pd.read_csv(
        DIR.joinpath(data_path, "raw/breast_cancer_wisconsin.data"),
        names=names,
        low_memory=False,
    )

    x: pd.DataFrame = data.drop(["id", "class"], axis=1)
    y: pd.DataFrame = data["class"].map({2: -1, 4: 1}).to_frame()
    del data

    if not output_encoder:
        input_encoder = UnitGaussianNormalizer()
        data_processor = DataProcessor(
            input_encoder=input_encoder, output_encoder=output_encoder
        )
        x = data_processor.preprocess(x)

    if input_encoder and output_encoder:
        input_encoder = UnitGaussianNormalizer()
        output_encoder = UnitGaussianNormalizer()
        data_processor = DataProcessor(
            input_encoder=input_encoder, output_encoder=output_encoder
        )
        x = data_processor.preprocess(x)
        y = data_processor.preprocess(y)

    if verbose:
        print("Data loaded and processed")
        print(f"X shape: {x.shape}")
        print(f"Y shape: {y.shape}")
        print(f"X head: {x.head()}")
        print(f"Y head: {y.head()}")

    split: int = int(0.8 * x.shape[0])

    # train samples
    x_train: pd.DataFrame = x.iloc[:split]
    y_train: pd.DataFrame = y.iloc[:split]

    # test samples
    x_test: pd.DataFrame = x.iloc[split:]
    y_test: pd.DataFrame = y.iloc[split:]
    del x, y

    if saver:
        x_train.to_csv(DIR.joinpath(data_path, "clean/train_features.csv"), index=False)
        y_train.to_csv(DIR.joinpath(data_path, "clean/train_labels.csv"), index=False)
        x_test.to_csv(DIR.joinpath(data_path, "clean/test_features.csv"), index=False)
        y_test.to_csv(DIR.joinpath(data_path, "clean/test_labels.csv"), index=False)

    train_loader = DictDataset(x_train.values, y_train.values)
    test_loader = DictDataset(x_test.values, y_test.values)

    return train_loader, test_loader, data_processor
