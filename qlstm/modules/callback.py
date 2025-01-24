from datetime import datetime
from typing import List, Union

import lightning.pytorch.callbacks as cb
import rootutils
from lightning.pytorch import LightningModule, Trainer
from rich import print

rootutils.autosetup()

from .utils import yaml_handler


class PrintResult(cb.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.start = -1
        self.prev = {}

    def _format_with_trend(
        self,
        name: str,
        value: Union[int, float],
        format_spec: str = "",
        up_green: bool = True,
    ) -> str:
        # Store the previous value and get the trend
        prev_value = self.prev.get(name, value)
        self.prev[name] = value

        # Check current vs previous
        if value > prev_value:
            trend = "green" if up_green else "red"
        elif value < prev_value:
            trend = "red" if up_green else "green"
        else:
            trend = "grey"

        # Format the value
        formatted_value = f"{value:{format_spec}}"

        return f"[{trend}]{formatted_value}[/]"

    @staticmethod
    def _format_time(x: datetime):
        return x.strftime("%d-%m-%Y %H:%M:%S")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        lr_sequence = ",".join(f"lr{i}" for i, _ in enumerate(trainer.optimizers))

        with open(f"{trainer.logger.log_dir}/train.csv", "a") as f:
            f.write(
                f"Epoch,{lr_sequence},train_loss,train_rmse,val_loss,val_rmse,epoch_time\n"
            )

        self.start = datetime.now()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total = datetime.now() - self.start

        with open(f"{trainer.logger.log_dir}/info.txt", "a") as f:
            f.write(
                "\n".join(
                    [
                        f"Start at: {self._format_time(self.start)}",
                        f"End at: {self._format_time(datetime.now())}",
                        f"Total training time: {total.total_seconds():.3f} seconds",
                    ]
                )
            )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.prev["time"] = datetime.now()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics

        lr = [
            self._format_with_trend(f"lr{i}", optim.param_groups[0]["lr"], ".1e", False)
            for i, optim in enumerate(trainer.optimizers)
        ]

        train_result = [
            f"loss: {self._format_with_trend('train_loss', results['train/loss'], '.4f', False)}",
            f"rmse: {self._format_with_trend('train_rmse', results['train/rmse'], '.3f', False)}",
        ]

        output = [
            f"[bold]Epoch[/]( {epoch} )",
            f"[bold]Lr[/]( {', '.join(lr)} )",
            f"[bold]Train[/]({', '.join(train_result)})",
        ]

        if "val/loss" in results:
            val_result = [
                f"loss: {self._format_with_trend('val_loss', results['val/loss'], '.4f', False)}",
                f"rmse: {self._format_with_trend('val_rmse', results['val/rmse'], '.3f', False)}",
            ]
            output.append(f"[bold]Val[/]({', '.join(val_result)})")

        print(" ".join(output))

        with open(f"{trainer.logger.log_dir}/train.csv", "a") as f:
            lr_values = ",".join(
                f"{optim.param_groups[0]['lr']:.2e}" for optim in trainer.optimizers
            )
            f.write(
                f"{epoch},{lr_values},"
                f"{results['train/loss']:.5f},{results['train/rmse']:.4f},"
                f"{results['val/loss']:.5f},{results['val/rmse']:.4f},"
                f"{(datetime.now() - self.prev['time']).total_seconds():.3f}\n"
            )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        results = trainer.callback_metrics

        with open(f"{trainer.logger.log_dir}/test.csv", "a") as f:
            f.write(
                f"test_loss,test_rmse\n{results['test/loss']:.5f},{results['test/rmse']:.4f}"
            )


def custom_callbacks() -> List[cb.Callback]:
    """
    Configure and return a list of custom callbacks for PyTorch Lightning.

    Returns
    -------
    List[cb.Callback]
        A list of configured PyTorch Lightning callback objects.
    """
    cfg = yaml_handler(f"{'/'.join(__file__.split('/')[:-2])}/configs/callbacks.yaml")

    callback_map = {
        "verbose": PrintResult(),
        "progress_bar": cb.RichProgressBar(),
        "lr_monitor": cb.LearningRateMonitor("epoch"),
        "enable_checkpoint": cb.ModelCheckpoint(**cfg["checkpoint"]),
        "enable_early_stopping": cb.EarlyStopping(**cfg["early_stopping"]),
    }

    return [callback for key, callback in callback_map.items() if cfg.get(key, False)]
