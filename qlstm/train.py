import shutil

import hydra
import rootutils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ls
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, open_dict
from rich import traceback

rootutils.autosetup()
traceback.install()

from models.LSTM import LSTM
from modules.callback import custom_callbacks
from modules.data import CustomDataModule
from modules.model import LitModel
from modules.scheduler import scheduler_with_warmup


@hydra.main(config_path="./configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Set precision
    torch.set_float32_matmul_precision("high")

    # Set seed
    if cfg["set_seed"]:
        seed_everything(seed=cfg["set_seed"], workers=True)

    # Define dataset
    dataset = CustomDataModule(
        **cfg["data"],
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=cfg["num_workers"] if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )

    # Define model
    model = LSTM(input_size=7, hidden_size=64, num_layers=1)

    # Setup loss
    loss = nn.SmoothL1Loss()

    # Setup optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"]["learning_rate"],
    )

    # Setup scheduler
    scheduler = scheduler_with_warmup(
        scheduler=ls.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg["trainer"]["num_epoch"]
        ),
        warmup_epochs=cfg["scheduler"]["warmup_epochs"],
        start_factor=cfg["scheduler"]["start_factor"],
    )

    # Lightning model
    lit_model = LitModel(
        model=model,
        criterion=loss,
        optimizer=[optimizer],
        scheduler=[scheduler],
        checkpoint=cfg["trainer"]["checkpoint"],
        device="auto",
    )

    # Save config
    with open_dict(cfg):
        cfg["model"]["name"] = model._get_name()
        if hasattr(model, "version"):
            cfg["model"]["version"] = model.version
    lit_model.save_hparams(cfg)

    # Lightning trainer
    trainer = Trainer(
        max_epochs=cfg["trainer"]["num_epoch"],
        precision=cfg["trainer"]["precision"],
        logger=TensorBoardLogger(save_dir="."),
        callbacks=custom_callbacks(),
        gradient_clip_val=cfg["trainer"]["gradient_clipping"],
    )

    # Lightning tuner
    # tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially
    # tuner.scale_batch_size(lit_model, datamodule=dataset)

    # Training
    trainer.fit(lit_model, dataset)

    # Testing
    trainer.test(lit_model, dataset)


if __name__ == "__main__":
    main()
