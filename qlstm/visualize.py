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
from modules.data import CustomDataModule
from modules.model import LitModel


@hydra.main(config_path="./configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)

    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Define dataset
    dataset = CustomDataModule(
        data_path="qlstm/data/place_1_new.csv",
        features="4-10",
        labels=[4],
        time_steps=24,
        overlap=True,
        n_future=1,
    )
    dataset.prepare_data()
    dataset.setup("predict")

    # Define model
    model = cLSTM(
        input_size=10,
        hidden_size=128,
        # n_qubits=2,
    )

    # Define lightning model
    lit_model = LitModel(
        model=model,
        output_size=dataset.n_future,
        checkpoint="lightning_logs/cLSTM/version_1/checkpoints/epoch=24-step=1500.ckpt",
    ).eval()

    # Inference loop
    with torch.inference_mode():
        X, y = dataset.val_set[:168]
        X, y = X[1:], y[:-1, :1]

        out = np.array([lit_model(x.unsqueeze(0)).squeeze(0) for x in X])

        y = dataset.encoder["label"].inverse_transform(y)
        outs = dataset.encoder["label"].inverse_transform(out).transpose(1, 0)

        plt.figure(figsize=(28, 20))
        plt.plot(range(len(y)), y, label="Test Actual", linewidth=3)

        [
            plt.plot(
                range(len(out)),
                np.insert(out[:-1], 0, 0),
                label=f"Test Prediction {i}",
                linewidth=3,
            )
            for i, out in enumerate(outs[:1])
        ]
        plt.title("Modified QLSTM", fontsize=24, pad=15)
        plt.xlabel("Time (hour)", fontsize=20)
        plt.ylabel("Power (kWh)", fontsize=20)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=20, loc="upper right")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

        # plt.savefig("./cQLSTMf.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
