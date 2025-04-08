from pathlib import Path
from collections import defaultdict
import re
from typing import Callable, List, Optional, Sequence
from numpy.lib.stride_tricks import sliding_window_view
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
        n_future: int = 1,
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
            n_future (int, optional): Number of future time steps being predicted. Default: 1
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
        self.processed_path = self._check_data(self.data_path)
        self.dataframe = self._load_data(self.data_path)
        self.features = self._check_features(features)
        self.labels = self._check_features(labels)
        self.time_steps = time_steps
        self.overlap = overlap
        self.n_future = n_future
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
    def _check_data(path: Path) -> Path:
        if len(path.suffixes) > 1 and path.suffixes[-2] == ".x":
            return path

        save_folder = path.parent / "trainable"
        save_folder.mkdir(parents=True, exist_ok=True)

        return save_folder / f"{path.stem}.x{path.suffix}"

    @staticmethod
    def _load_data(path: Path, **kargs) -> pd.DataFrame:
        "Load data into pandas Dataframe."
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        match path.suffix:
            case ".csv":
                df = pd.read_csv(path, na_filter=True, skip_blank_lines=True, **kargs)
            case ".xlsx":
                df = pd.read_excel(path, na_filter=True, skip_blank_lines=True, **kargs)
            case _:
                raise ValueError("Only csv or xlsx file are supported.")

        # Correct header
        df.columns = df.columns.str.strip().str.lower()

        return df

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
    def _fill_hours(df: pd.DataFrame, start=6, end=18):
        filled_rows = []
        prev_date = None
        prev_hour = start - 1

        def add_rows(date, start, end):
            new_rows = []
            for h in range(start, end):
                row = {}
                for col in df.columns:
                    if col == "datetime":
                        row[col] = date + pd.DateOffset(hours=h)
                    else:
                        row[col] = np.nan
                new_rows.append(row)
            filled_rows.extend(new_rows)

        for _, row in df.iterrows():
            curr_date = row["datetime"].date()
            curr_hour = row["datetime"].hour

            # If current hour < previous hour, a new day started
            if curr_hour < prev_hour:
                add_rows(prev_date, prev_hour + 1, end + 1)
                add_rows(curr_date, start, curr_hour)
            else:
                add_rows(curr_date, prev_hour + 1, curr_hour)

            # Append the current row
            filled_rows.append(row.to_dict())

            # Update previous hour
            prev_hour = curr_hour
            prev_date = curr_date

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
        # Post-processed features
        self.features.remove("hour")
        self.features.extend(["hour_sin", "hour_cos", "rolling_mean"])

        # Check exists
        if self.processed_path.exists():
            return

        # Create a copy of dataframe
        data = self.dataframe.copy()

        # Drop unnamed columns
        data = data.loc[:, ~data.columns.str.contains("^Unnamed", case=False)]

        # Convert 'date' and 'hour' columns to datetime type
        data["date"] = pd.to_datetime(data["date"])
        data["hour"] = pd.to_timedelta(data["hour"], unit="h")
        data["date"] = data["date"] + data["hour"]
        data = data.rename(columns={"date": "datetime"})
        data.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

        # Sort by date
        data = data.sort_values(by="datetime")

        # Fill missing hours
        data = self._fill_hours(data)
        data.interpolate(method="linear", inplace=True)

        # Cyclical Features Encoding
        _angle = 2 * np.pi * data["datetime"].dt.hour / 24
        data["hour_sin"] = np.sin(_angle)
        data["hour_cos"] = np.cos(_angle)

        # Moving Average
        data["rolling_mean"] = data[self.labels].rolling(window=6, min_periods=1).mean()

        # Set "datetime" as index
        data = data.set_index("datetime")

        # Drop unnecessary columns
        x_columns = set(self.features + self.labels)
        data = data.loc[:, data.columns.isin(x_columns)]

        # Finalize
        data = data.dropna().round(4)

        # Save new data
        data.to_csv(str(self.processed_path), index=True)

    def setup(self, stage: str):
        if not hasattr(self, "dataset"):
            # Load processed data
            data = self._load_data(self.processed_path)

            # Limit data
            if self.data_limit:
                data = self._limit_data(data)

            # Create inputs and labels
            inputs = data[self.features].copy()
            labels = data[self.labels].copy()

            # Encode data
            if self.scaler:
                scalable = [
                    x for x in self.features if x not in ["hour_sin", "hour_cos"]
                ]
                inputs[scalable] = self.encoder["input"].fit_transform(inputs[scalable])
                labels[self.labels] = self.encoder["label"].fit_transform(
                    labels[self.labels]
                )

            # Create time steps
            inputs = sliding_window_view(inputs.values, self.time_steps, 0)[
                : -self.n_future
            ].transpose(0, 2, 1)
            labels = sliding_window_view(labels.values, self.n_future, 0)[
                self.time_steps :
            ].squeeze(1)

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

            self.val_set = Subset(self.dataset, range(train_end, val_end))
            self.test_set = Subset(self.dataset, range(val_end, dataset_size))
            self.train_set = Subset(self.dataset, range(0, train_end))

        if stage == "fit":
            self._summary()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.loader_config)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.loader_config)
