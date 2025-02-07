import shutil

import hydra
import rootutils
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ls
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, open_dict
from rich import traceback

rootutils.autosetup()
traceback.install()

from models.CustomQLSTM import QLSTM
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
    dataset = CustomDataModule(**cfg["data"])

    # Define model
    model = QLSTM(
        input_size=7,
        hidden_size=64,
        seq_length=cfg["data"]["time_steps"],
        n_qubits=2,
        n_qlayers=1,
    )

    # Setup loss
    loss = nn.SmoothL1Loss()

    # Setup optimizer
    optimizer = optim.AdamW(params=model.parameters(), **cfg["optimizer"])

    # Setup scheduler
    scheduler = scheduler_with_warmup(
        scheduler=ls.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg["trainer"]["max_epochs"]
        ),
        **cfg["scheduler"],
    )

    # Lightning model
    lit_model = LitModel(
        model=model,
        criterion=loss,
        optimizer=[optimizer],
        scheduler=[scheduler],
        **cfg["model"],
    )

    # Save config
    with open_dict(cfg):
        cfg["model"]["name"] = model._get_name()
        if hasattr(model, "version"):
            cfg["model"]["version"] = model.version
    lit_model.save_hparams(cfg)

    # Lightning trainer
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="."),
        callbacks=custom_callbacks(),
        **cfg["trainer"],
    )

    # Lightning tuner
    # tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially
    # tuner.scale_batch_size(lit_model, datamodule=dataset)

    # Training
    trainer.fit(lit_model, dataset)

    # Testing
    trainer.test(lit_model, dataset, "best")

    print(model.feature_weighting)
    print(model.temporal_weighting)
    print(model.magnitude)

    plt.figure(figsize=(18, 10))

    sns.heatmap(
        (model.feature_weighting * model.temporal_weighting)
        .squeeze(0)
        .detach()
        .numpy(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
    )
    plt.title("Weighted Matrix", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Times", fontsize=14)
    plt.savefig(f"{trainer.log_dir}/weighted_matrix.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
