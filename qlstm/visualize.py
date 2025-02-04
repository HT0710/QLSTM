# Plot the test and denormalized prediction data for the first 24 hours
import shutil

import hydra
import rootutils
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from rich import traceback

rootutils.autosetup()
traceback.install()

from models.CustomQLSTM import QLSTM
from modules.data import CustomDataModule
from modules.model import LitModel


@hydra.main(config_path="./configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Define dataset
    dataset = CustomDataModule(data_path=cfg["data_path"], time_steps=cfg["time_steps"])
    dataset.prepare_data()

    # Define model
    model = QLSTM(
        input_size=7,
        hidden_size=128,
        seq_length=cfg["time_steps"],
        n_qubits=4,
        n_qlayers=1,
    )

    # Define lightning model
    lit_model = LitModel(model=model, checkpoint=cfg["checkpoint"]).eval()

    # Inference loop
    with torch.inference_mode():
        times = len(dataset.test_set)
        X, y = dataset.test_set[:]

        out = lit_model(X)

        y = dataset.encoder["Measured Power"].inverse_transform(y.reshape(-1, 1))
        out = dataset.encoder["Measured Power"].inverse_transform(out)

        plt.figure(figsize=(36, 12))

        plt.plot(range(len(y)), y, label="Test Actual", alpha=0.7)

        plt.plot(range(len(out)), out, label="Test Prediction", alpha=0.7)

        plt.plot(
            range(len(X)),
            dataset.dataframe["NWP Radiation"][:times],
            label="NWP",
            alpha=0.7,
            linestyle="--",
        )

        plt.xlabel("Hour")
        plt.ylabel("Power")
        plt.title("Test Actual vs Prediction Values (Denormalized)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
