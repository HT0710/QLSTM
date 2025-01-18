import shutil

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from rich import print, traceback

rootutils.autosetup()
traceback.install()

from models.LSTM import LSTM
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
    model = LSTM(input_size=7, hidden_size=128, num_layers=4)

    # Define lightning model
    lit_model = LitModel(model=model, checkpoint=cfg["checkpoint"]).eval()

    torch.set_printoptions(4, sci_mode=False)
    # Inference loop
    with torch.inference_mode():
        while True:
            index = input("Index: ")

            try:
                index = int(index)
            except Exception:
                exit()

            X, y = dataset.dataset[index]
            print(X)

            out = lit_model(X.unsqueeze(0))

            y = dataset.encoder["Measured Power"].inverse_transform(y.reshape(-1, 1))
            out = dataset.encoder["Measured Power"].inverse_transform(out)

            print(f"Label: {y.item()}")
            print(f"Output: {out.item()}\n")


if __name__ == "__main__":
    main()
