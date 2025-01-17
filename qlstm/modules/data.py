from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from rich import print
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from modules.utils import workers_handler


class DataModule(Dataset):
    """Pytorch Data Module"""

    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]

        text = torch.tensor(text, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return text, label


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
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
            data_limit (float, optional): Limit for the size of the dataset (0 -> 1.0). Default: None
            split_size (sequence, optional): Proportions for train, validation, and test splits. Default: (0.75, 0.15, 0.1)
            num_workers (int, optional): Number of data loading workers. Default: 0
            pin_memory (bool, optional): Whether to pin memory for faster data transfer. Default: True
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.data_limit = self._check_limit(data_limit)
        self.split_size = split_size
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
        }

    @staticmethod
    def _check_limit(value: Optional[float]) -> Optional[float]:
        if isinstance(value, float) and 0 < value < 1:
            return value

    def _limit_data(self, data: Dataset) -> Subset:
        return random_split(
            dataset=data, lengths=(self.data_limit, 1 - self.data_limit)
        )[0]

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

        # Process data
        data = self.dataframe["Measured Power"]

        self.encoder = MinMaxScaler()

        data = self.encoder.fit_transform(data.values.reshape(-1, 1))

        inputs = data[:-1:]
        labels = data[1::]

        # Modulize data
        data = DataModule(inputs, labels)

        # Limit data
        if self.data_limit:
            data = self._limit_data(data)

        # Finalize
        self.dataset = data

    def setup(self, stage: str):
        if not hasattr(self, "train_set"):
            self.train_set, self.val_set, self.test_set = random_split(
                dataset=self.dataset, lengths=self.split_size
            )

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
