import shutil

import hydra
import numpy as np
import rootutils
import torch
from lightning.pytorch import seed_everything
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from rich import traceback
from sklearn.preprocessing import MinMaxScaler

rootutils.autosetup()
traceback.install()

from models.cQLSTMf import QLSTM
from models.CustomQLSTM import QLSTM as QLSTM4
from modules.data import CustomDataModule
from modules.model import LitModel


@hydra.main(config_path="./configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)

    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Define dataset
    dataset1 = CustomDataModule(
        data_path=cfg["data_path"],
        time_steps=1,
        overlap=False,
    )
    dataset1.prepare_data()

    dataset2 = CustomDataModule(
        data_path=cfg["data_path"],
        time_steps=cfg["time_steps"],
        overlap=True,
        scaler=MinMaxScaler,
    )
    dataset2.prepare_data()

    # Define model
    model1 = QLSTM(
        input_size=8,
        hidden_size=128,
        n_qubits=2,
    )
    # Define model
    model2 = QLSTM4(
        input_size=7,
        hidden_size=128,
        n_qubits=2,
    )

    # Define lightning model
    lit_model1 = LitModel(
        model=model1,
        checkpoint="lightning_logs/cQLSTMf/version_3/checkpoints/epoch=3-step=8796.ckpt",
    ).eval()
    lit_model2 = LitModel(
        model=model2,
        checkpoint="lightning_logs/QLSTM4/24_128_2/checkpoints/epoch=49-step=3400.ckpt",
    ).eval()

    # Inference loop
    with torch.inference_mode():
        t = 12
        X1, y1 = dataset1.val_set[t + 6 : (24 * 14) + t + 6]
        X2, y2 = dataset2.val_set[t : (24 * 14) + t]

        out1 = np.array([lit_model1(x.unsqueeze(0)).squeeze(-1) for x in X1])
        out2 = lit_model2(X2[:, :, 1:])

        # print(out1.shape)
        # print(out2.shape)
        # exit()

        y1 = dataset1.encoder["Measured Power"].inverse_transform(y1.reshape(-1, 1))
        y2 = dataset2.encoder["Measured Power"].inverse_transform(y2.reshape(-1, 1))
        out1 = dataset1.encoder["Measured Power"].inverse_transform(out1)
        out2 = dataset2.encoder["Measured Power"].inverse_transform(out2)

        # plt.figure(figsize=(12, 12))

        # plt.plot(range(len(y)), y, label="Test Actual", alpha=0.7)

        # plt.plot(range(len(out)), out, label="Test Prediction", alpha=0.7)
        # plt.xlabel("Hour")
        # plt.ylabel("Power")
        # plt.title("Test Actual vs Prediction Values (Denormalized)")
        # plt.legend()
        # plt.show()
        # plt.savefig("./CQLSTM.png", dpi=300, bbox_inches="tight")

        # 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 14))

        # Plot for the left subplot
        ax1.plot(range(len(y1)), y1, label="Test Actual", linewidth=3)
        ax1.plot(
            range(len(out1)),
            out1,
            label="Test Prediction",
            color="orange",
            linewidth=3,
        )
        ax1.set_title("QLSTM", fontsize=24, pad=15)
        ax1.set_xlabel("Time (hour)", fontsize=20)
        ax1.set_ylabel("Power (kW)", fontsize=20)
        ax1.tick_params(axis="both", labelsize=16)
        ax1.legend(fontsize=18, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Plot for the right subplot (using the same data for demonstration)
        ax2.plot(range(len(y2)), y2, label="Test Actual", linewidth=3)
        ax2.plot(
            range(len(out2)),
            out2,
            label="Test Prediction",
            color="orange",
            linewidth=3,
        )
        ax2.set_title("Modified QLSTM", fontsize=24, pad=15)
        ax2.set_xlabel("Time (hour)", fontsize=20)
        ax2.set_ylabel("Power (kW)", fontsize=20)
        ax2.tick_params(axis="both", labelsize=16)
        ax2.legend(fontsize=18, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

        # plt.savefig("./QLSTMvsMQLSTM.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
