from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from rich import print
from rich.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from modules.utils import workers_handler


class CustomDataset(Dataset):
    """Pytorch Data Module"""

    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        if len(self.inputs) != len(self.labels):
            raise IndexError("The length of the inputs and labels do not match.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        time_steps: int = 1,
        data_limit: Optional[float] = None,
        split_size: Sequence[float] = (0.75, 0.15, 0.1),
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """
        Custom Data Module for PyTorch Lightning

        Args:
            data_path (str): Path to the dataset.
            batch_size (int, optional): Batch size for data loading. Default: 32
            time_steps (int, optional): Number of time steps to include in each input sequence. Default: 1
            data_limit (float, optional): Limit for the size of the dataset as a fraction (0 -> 1.0).
                                          If None, use the full dataset. Default: None
            split_size (Sequence[float], optional): Proportions for train, validation, and test splits.
                                                    The values should sum to 1.0. Default: (0.75, 0.15, 0.1)
            num_workers (int, optional): Number of workers for data loading in parallel. Default: 0
            pin_memory (bool, optional): Whether to pin memory for faster data transfer to GPU. Default: True
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.time_steps = time_steps
        self.data_limit = self._check_limit(data_limit)
        self.split_size = split_size
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
        }
        self.encoder = {}

    @staticmethod
    def _check_limit(value: Optional[float]) -> Optional[float]:
        "Check input value for limit."
        if isinstance(value, float) and 0 < value < 1:
            return value

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
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Load data
        match self.data_path.suffix:
            case ".csv":
                data = pd.read_csv(self.data_path, header=0)
            case ".xlsx":
                data = pd.read_excel(self.data_path, header=0)
            case _:
                raise ValueError("Only csv or xlsx file are supported.")

        # Clean up
        self.dataframe = data.dropna().drop_duplicates()

        # Limit data
        if self.data_limit:
            data = self._limit_data(data)

        # Select data
        features = [
            "Measured Power",
            "NWP Radiation",
            "NWP Rainfall",
            "NWP Temperature",
            "NWP Pressure",
            "NWP Windspeed",
            "Measured Radiation",
        ]

        data = self.dataframe[features]

        # Encode data
        for feature in features:
            self.encoder[feature] = MinMaxScaler()
            data.loc[:, feature] = (
                self.encoder[feature]
                .fit_transform(data[[feature]].values)
                .astype("object")
            )

        # Create inputs and labels
        inputs = data.values
        labels = data[["Measured Power"]].values

        # Create time steps
        inputs = [
            inputs[i : i + self.time_steps]
            for i in range(len(inputs) - self.time_steps)
        ]
        inputs = np.array(inputs)

        labels = [
            labels[i + self.time_steps] for i in range(len(labels) - self.time_steps)
        ]
        labels = np.array(labels)

        # Train, val, test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            inputs, labels, train_size=self.split_size[0], shuffle=False
        )

        val_size = self.split_size[1] / (1 - self.split_size[0])

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_size, shuffle=False
        )

        # Modulize data
        self.train_set = CustomDataset(X_train, y_train)
        self.val_set = CustomDataset(X_val, y_val)
        self.test_set = CustomDataset(X_test, y_test)

        # Finalize
        self.dataset = ConcatDataset([self.train_set, self.val_set, self.test_set])

    def setup(self, stage: str):
        if stage == "fit":
            self._summary()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set, **self.loader_config, shuffle=False, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.loader_config, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.loader_config, shuffle=False)
