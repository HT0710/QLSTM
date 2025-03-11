from calendar import monthrange
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import rootutils
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")
rootutils.autosetup(".gitignore")

from qlstm.models.cLSTM import LSTM as cLSTM
from qlstm.models.cQLSTMf import QLSTM as cQLSTMf
from qlstm.models.LSTM import LSTM
from qlstm.modules.data import CustomDataModule
from qlstm.modules.model import LitModel


class DemoTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.model_path = self.root / "models"
        self.dataset = list(self.data_path.glob("*.csv"))
        self.datasets = {}
        self.models = {
            "LSTM": {
                "init": LSTM(9, 128),
                "checkpoint": "lightning_logs/LSTM/version_0/checkpoints/epoch=2-step=3582.ckpt",
            },
            # "LSTMf": {"init": LSTMf(8, 128), "checkpoint": None},
            "cLSTM": {
                "init": cLSTM(9, 128),
                "checkpoint": "lightning_logs/cLSTM/version_0/checkpoints/epoch=14-step=17910.ckpt",
            },
            # "cLSTMf": {"init": cLSTMf(8, 128), "checkpoint": None},
            # "QLSTM": {"init": QLSTM(8, 128, 2), "checkpoint": None},
            # "QLSTMf": {"init": QLSTMf(8, 128, 2), "checkpoint": None},
            # "cQLSTM": {"init": cQLSTM(8, 128, 2), "checkpoint": None},
            "cQLSTMf": {
                "init": cQLSTMf(9, 128, 2),
                "checkpoint": "lightning_logs/cQLSTMf/version_9/checkpoints/last.ckpt",
            },
        }
        self.steps = None

    def _load_data(self, data_name):
        return pd.read_csv(str(self.data_path / data_name))

    def _calculate_hours(self, year_1, month_1, day_1, year_2, month_2, day_2):
        dt_1 = datetime(year_1, month_1, day_1).timestamp()
        dt_2 = datetime(year_2, month_2, day_2).timestamp()

        delta = int((dt_2 - dt_1) / 3600)

        self.steps = max(0, delta)

    def _update_days(self, year, month, day):
        _, num_days = monthrange(year, month)
        return gr.Dropdown(choices=range(1, num_days + 1), value=min(day, num_days))

    def _update_years(self, data_name: str):
        key = "Year" if data_name.startswith("place") else "year"
        years = self._load_data(data_name).get(key, np.array([2025]))
        _range = range(years.min(), years.max() + 1)
        return [gr.Dropdown(choices=_range, value=int(years.min()))] * 2

    def _update_days(self, year, month, day):
        _, num_days = monthrange(year, month)
        return gr.Dropdown(choices=range(1, num_days + 1), value=min(day, num_days))

    def _predict(self, model_name, data_name):
        # Define dataset
        if data_name not in self.datasets:
            self.datasets[data_name] = CustomDataModule(
                data_path=str(self.data_path / data_name),
                features="5-11",
                labels=[11],
                time_steps=24,
                overlap=True,
            )
            self.datasets[data_name].prepare_data()
        data = self.datasets[data_name]

        # Define lightning model
        model = self.models[model_name]
        if "loaded" not in model:
            model["loaded"] = LitModel(
                model=model["init"], checkpoint=model["checkpoint"]
            ).eval()

        # Inference loop
        with torch.inference_mode():
            plt.figure(figsize=(24, 7))
            if self.steps == 0:
                return plt

            X, y = data.dataset[: self.steps]
            out = np.array([model["loaded"](x.unsqueeze(0)).squeeze(-1) for x in X])

            # Denormalize
            y = data.encoder["label"].inverse_transform(y.reshape(-1, 1))
            out = data.encoder["label"].inverse_transform(out)

            plt.plot(range(len(y)), y, label="Actual", alpha=0.7, linewidth=2)
            plt.plot(range(len(out)), out, label="Prediction", alpha=0.7, linewidth=2)
            plt.xlabel("Time (hour)")
            plt.ylabel("Power (kW)")
            plt.legend()
            plt.tight_layout()

        return plt

    def __call__(self):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration")
                model_dropdown = gr.Dropdown(
                    choices=self.models.keys(),
                    label="Model",
                    interactive=True,
                )
                data_dropdown = gr.Dropdown(
                    choices=[str(x.name) for x in self.dataset],
                    label="Dataset",
                    interactive=True,
                )
                button = gr.Button("Run")

            with gr.Column():
                gr.Markdown("### From")
                with gr.Row():
                    fday = gr.Dropdown(
                        range(1, 32),
                        label="Day",
                        min_width=50,
                        interactive=True,
                    )
                    fmonth = gr.Dropdown(
                        range(1, 13),
                        label="Month",
                        min_width=50,
                        interactive=True,
                    )
                    fyear = gr.Dropdown(
                        label="Year",
                        min_width=50,
                        interactive=True,
                    )

                    fmonth.change(self._update_days, [fyear, fmonth, fday], fday)
                    fyear.change(self._update_days, [fyear, fmonth, fday], fday)

                gr.Markdown("### To")
                with gr.Row():
                    tday = gr.Dropdown(
                        range(1, 32),
                        label="Day",
                        min_width=50,
                        interactive=True,
                    )
                    tmonth = gr.Dropdown(
                        range(1, 13),
                        label="Month",
                        min_width=50,
                        interactive=True,
                    )
                    tyear = gr.Dropdown(
                        label="Year",
                        min_width=50,
                        interactive=True,
                    )

                    tmonth.change(self._update_days, [tyear, tmonth, tday], tday)
                    tyear.change(self._update_days, [tyear, tmonth, tday], tday)

        gr.Markdown("### Result")
        with gr.Row():
            plot = gr.Plot(show_label=False)

        data_dropdown.change(self._update_years, [data_dropdown], [fyear, tyear])
        self.parent.load(self._update_years, [data_dropdown], [fyear, tyear])

        button.click(self._calculate_hours, [fyear, fmonth, fday, tyear, tmonth, tday])
        button.click(self._predict, [model_dropdown, data_dropdown], plot)
