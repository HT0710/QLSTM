import shutil

import hydra
import numpy as np
import rootutils
import torch
from lightning.pytorch import seed_everything
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from rich import traceback

rootutils.autosetup()
traceback.install()

from models.cLSTM import cLSTM
from models.cQLSTM import cQLSTM
from models.LSTM import LSTM
from models.QLSTM import QLSTM
from modules.data import CDM2
from modules.model import LitModel


@hydra.main(config_path="./configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    seed_everything(seed=42, workers=True)

    # Define dataset
    dataset = CDM2(
        data_path="qlstm/data/place_1_new.csv",
        features="4-10",
        labels=[4],
        time_steps=24,
        overlap=True,
    )
    dataset.prepare_data()
    dataset.setup("predict")

    # Define model
    model = QLSTM(
        input_size=10,
        hidden_size=352,
        n_qubits=4,
    )
    # Define lightning model
    lit_model3 = LitModel(
        model=model,
        checkpoint="lightning_logs/QLSTM/version_3/checkpoints/last.ckpt",
        device="cpu",
    ).eval()

    # Define model
    model = LSTM(input_size=10, hidden_size=23)
    # Define lightning model
    lit_model1 = LitModel(
        model=model,
        checkpoint="lightning_logs/LSTM/version_1/checkpoints/last.ckpt",
        device="cpu",
    ).eval()

    # Define model
    model = cLSTM(input_size=10, hidden_size=23)
    # Define lightning model
    lit_model2 = LitModel(
        model=model,
        checkpoint="lightning_logs/cLSTM/version_11/checkpoints/last.ckpt",
        device="cpu",
    ).eval()

    # Define model
    model = cQLSTM(
        input_size=10,
        hidden_size=128,
        n_qubits=4,
    )
    # Define lightning model
    lit_model4 = LitModel(
        model=model,
        checkpoint="lightning_logs/cQLSTM/version_0/checkpoints/last.ckpt",
        device="cpu",
    ).eval()

    plt.figure(figsize=(16, 9))

    X, y = dataset.val_set[86 : 86 + 23]
    # X, y = X[1:], y[:-1, :1]

    y = dataset.encoder["label"].inverse_transform(y)

    plt.plot(range(len(y)), y, label="Actual", linewidth=3)

    # Inference loop
    with torch.inference_mode():
        out = np.array(lit_model1(X))

        out = dataset.encoder["label"].inverse_transform(out)

        plt.plot(range(len(out)), out, label="LSTM", linewidth=3)

        out = np.array(lit_model2(X))

        out = dataset.encoder["label"].inverse_transform(out)

        plt.plot(range(len(out)), out, label="mLSTM", linewidth=3)

        out = np.array(lit_model3(X))

        out = dataset.encoder["label"].inverse_transform(out)

        plt.plot(range(len(out)), out, label="QLSTM", linewidth=3)

        out = np.array(lit_model4(X))

        out = dataset.encoder["label"].inverse_transform(out)

        plt.plot(range(len(out)), out, label="mQLSTM", linewidth=3)

    # plt.title("Modified QLSTM", fontsize=24, pad=15)
    plt.xlabel("Time (hour)", fontsize=20)
    plt.ylabel("Power (kWh)", fontsize=20)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=20, loc="upper right")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    # plt.show()

    plt.savefig("./first_4model.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
