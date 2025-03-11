from pathlib import Path
from collections import defaultdict
import re
from typing import Callable, List, Optional, Sequence

import numpy as np
from omegaconf import ListConfig
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from rich import print
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from .utils import workers_handler


class CustomDataset(Dataset):
    """Pytorch Data Module"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.X.shape[0] != self.y.shape[0]:
            raise IndexError("The length of the inputs and labels do not match.")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        features: str | List[int | str],
        labels: str | List[int | str],
        batch_size: int = 1,
        time_steps: int = 1,
        overlap: bool = True,
        scaler: Callable = StandardScaler(),
        data_limit: Optional[float] = None,
        split_size: Sequence[float] = (0.75, 0.15, 0.1),
        num_workers: int = 0,
        pin_memory: bool = torch.cuda.is_available(),
    ) -> None:
        """
        Custom Data Module for PyTorch Lightning

        Args:
            data_path (str): Path to the dataset.
            features (str | List[int | str]): List of feature indices or column names to be used as input features.
            labels (str | List[int | str]): List of label indices or column names to be used as target variables.
            batch_size (int, optional): Batch size for data loading. Default: 1
            time_steps (int, optional): Number of time steps to include in each input sequence. Default: 1
            overlap (bool, optional): Whether overlapping windows should be used when generating sequences. Default: True
            scaler (Callable, optional): Scaling function to apply to the data. Default: StandardScaler()
            data_limit (float, optional): Limit for the size of the dataset as a fraction (0 -> 1.0).
                                          If None, use the full dataset. Default: None
            split_size (Sequence[float], optional): Proportions for train, validation, and test splits.
                                                    The values should sum to 1.0. Default: (0.75, 0.15, 0.1)
            num_workers (int, optional): Number of workers for data loading in parallel. Default: 0
            pin_memory (bool, optional): Whether to pin memory for faster data transfer to GPU. Default: True
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.dataframe = self._load_data(self.data_path)
        self.features = self._check_features(features)
        self.labels = self._check_features(labels)
        self.time_steps = time_steps
        self.overlap = overlap
        self.scaler = scaler
        self.encoder = defaultdict(lambda: scaler)
        self.data_limit = self._check_limit(data_limit)
        self.split_size = split_size
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
            "shuffle": False,
            "drop_last": True,
        }

    @staticmethod
    def _load_data(path: Path) -> pd.DataFrame:
        "Load data into pandas Dataframe."
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        match path.suffix:
            case ".csv":
                data = pd.read_csv(path, na_filter=True, skip_blank_lines=True)
            case ".xlsx":
                data = pd.read_excel(path, na_filter=True, skip_blank_lines=True)
            case _:
                raise ValueError("Only csv or xlsx file are supported.")

        # Correct header
        data.columns = data.columns.str.strip().str.lower()

        return data.drop_duplicates()

    def _check_features(self, indices: List[int | str]) -> List[str]:
        "Check input value for features and labels indices."
        if isinstance(indices, ListConfig):
            indices = list(indices)

        # Check if indices is string
        if isinstance(indices, str):
            match = re.fullmatch(r"(\d+)-(\d+)", indices)
            if not match:
                raise ValueError(
                    'String indices must matched this format {small number}-{larger number}. Example: "3-5", "7-11".'
                )
            start, stop = map(int, match.groups())
            if start >= stop:
                raise ValueError(
                    'String indices must matched this format {small number}-{larger number}. Example: "3-5", "7-11".'
                )
            indices = list(range(start, stop + 1))

        # Check if indices is list
        if not isinstance(indices, list):
            raise TypeError("Features and Labels must be a list.")

        # Check list
        if all(isinstance(i, str) for i in indices):
            return [i.strip().lower() for i in indices]

        if all(isinstance(i, int) for i in indices):
            return [i.strip().lower() for i in self.dataframe.columns[sorted(indices)]]

        raise TypeError("Features and Labels must be a list of int or string.")

    @staticmethod
    def _check_limit(value: Optional[float]) -> Optional[float]:
        "Check input value for limit."
        if isinstance(value, float) and 0 < value < 1:
            return value

    @staticmethod
    def _fill_hours(df: pd.DataFrame):
        filled_rows = []
        prev_hour = 0

        def add_rows(start, end):
            filled_rows.extend(
                [
                    {col: (h if col == "hour" else 0) for col in df.columns}
                    for h in range(start, end)
                ]
            )

        for _, row in df.iterrows():
            current_hour = int(row["hour"])

            # If current hour < previous hour, a new day started
            if current_hour < prev_hour:
                add_rows(prev_hour + 1, 24)
                add_rows(0, current_hour)
            else:
                add_rows(prev_hour + 1, current_hour)

            # Append the current row
            filled_rows.append(row.to_dict())

            # Update previous hour
            prev_hour = current_hour

        return pd.DataFrame(filled_rows)

    def _limit_data(self, data: pd.DataFrame) -> pd.DataFrame:
        "Apply limit to the data."
        return data[: int(len(data) * self.data_limit)]

    def _summary(self) -> None:
        table = Table(title="[bold]Sets Distribution[/]")
        table.add_column("Set", style="cyan", no_wrap=True)
        table.add_column("Total", justify="right", style="magenta")
        table.add_column("Split", justify="right", style="green")
        for set_name, set_len in [
            ("Train", len(self.train_set)),
            ("Val", len(self.val_set)),
            ("Test", len(self.test_set)),
        ]:
            table.add_row(
                set_name, f"{set_len:,}", f"{set_len / len(self.dataset):.0%}"
            )
        print(table)
        output = [
            (
                f"[bold]Number of data:[/] {len(self.dataset):,}"
                + (
                    f" ([red]{self.data_limit:.0%}[/])"
                    if self.data_limit and self.data_limit != 1
                    else ""
                )
            ),
            f"[bold]Data path:[/] [green]{self.data_path}[/]",
        ]
        print("\n".join(output))

    def prepare_data(self):
        # Selecte and create a copy of dataframe
        data = self.dataframe.copy()

        # Fill missing hours
        data = self._fill_hours(data)

        # Limit data
        if self.data_limit:
            data = self._limit_data(data)

        # Create inputs and labels
        inputs = data[self.features].copy()
        labels = data[self.labels].copy()

        # Cyclical Features Encoding
        if "hour" in inputs.columns:
            _angle = 2 * np.pi * data["hour"] / 24
            inputs["hour_sin"] = np.sin(_angle)
            inputs["hour_cos"] = np.cos(_angle)
            inputs.drop(columns="hour", inplace=True)
            self.features.remove("hour")

        # Encode data
        if self.scaler:
            inputs[self.features] = self.encoder["input"].fit_transform(
                inputs[self.features]
            )
            labels[self.labels] = self.encoder["label"].fit_transform(
                labels[self.labels]
            )

        # Add Moving Average
        inputs["rolling_mean"] = labels.rolling(window=6, min_periods=1).mean()

        # Create time steps
        inputs = np.lib.stride_tricks.sliding_window_view(
            inputs.values, (self.time_steps, inputs.shape[1])
        )[:-1].squeeze(axis=1)
        labels = labels.values[self.time_steps :]

        # Check overlap option
        if not self.overlap:
            inputs = inputs[:: self.time_steps]
            labels = labels[:: self.time_steps]

        # Create dataset
        self.dataset = CustomDataset(inputs, labels)

        # Train, val, test split
        dataset_size = len(self.dataset)

        train_end = int(dataset_size * self.split_size[0])
        val_end = train_end + int(dataset_size * self.split_size[1])

        self.train_set = Subset(self.dataset, range(0, train_end))
        self.val_set = Subset(self.dataset, range(train_end, val_end))
        self.test_set = Subset(self.dataset, range(val_end, dataset_size))

    def setup(self, stage: str):
        if stage == "fit":
            self._summary()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.loader_config)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.loader_config)
