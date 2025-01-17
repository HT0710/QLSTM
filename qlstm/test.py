import shutil

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from rich import print, traceback
from rich.prompt import Prompt

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
    dataset = CustomDataModule(data_path=cfg["data_path"])
    dataset.prepare_data()

    # Define model
    model = LSTM(input_size=1, hidden_size=64)

    # Define lightning model
    lit_model = LitModel(model=model, checkpoint=cfg["checkpoint"]).eval()

    # Inference loop
    with torch.inference_mode():
        while True:
            X = torch.tensor([float(v) for v in Prompt.ask("Input").split(",")])

            X = dataset.encoder.transform(X.reshape(-1, 1))

            out = lit_model(torch.tensor(X, dtype=torch.float32))

            out = dataset.encoder.inverse_transform(out)

            print(f"Output: {out.reshape(-1)}\n")


if __name__ == "__main__":
    main()
