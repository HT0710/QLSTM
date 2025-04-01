from calendar import monthrange
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
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
        self.dataset = sorted(
            [i.name for i in self.data_path.glob("*.csv") if not i.match("*.x.csv")]
        )
        self.datasets = {}
        self.models = {
            "LSTM": {
                "init": LSTM(9, 128),
                "checkpoint": "lightning_logs/LSTM/base/checkpoints/last.ckpt",
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
        self.current = {
            "data": None,
            "from": None,
            "to": None,
        }

    def _select_data(self, data_name):
        data = CustomDataModule(
            data_path=str(self.data_path / data_name),
            features="5-11",
            labels=[11],
            time_steps=24,
            overlap=True,
        )
        data.prepare_data()
        data.setup("predict")

        datetime = pd.read_csv(
            data.processed_path, parse_dates=["datetime"], index_col="datetime"
        ).index

        self.current["data"] = pd.DataFrame(
            [{"datetime": t, "batch": b} for t, b in zip(datetime, data.dataset)]
        ).set_index("datetime")
        self.current["encoder"] = data.encoder

        years = datetime.year
        year_range = range(years.min(), years.max() + 1)

        self.current["from"] = datetime.min()
        self.current["to"] = datetime.min() + pd.DateOffset(days=7)

        return (
            self._predict(),
            gr.update(choices=year_range, value=int(years.min())),
            gr.update(choices=year_range, value=int(years.min())),
        )

    def _select_time(self, y, m, d, indicator):
        _, new_num_days = monthrange(y, m)
        d = min(d, new_num_days)

        self.current[indicator] = datetime(y, m, d)

        return gr.update(choices=range(1, new_num_days + 1), value=d)

    def _select_model(self, model_name):
        model = self.models[model_name]
        self.current["model"] = LitModel(
            model["init"], checkpoint=model["checkpoint"]
        ).eval()

    def _predict(self):
        plt.close()

        plt.figure(figsize=(24, 7))

        with torch.inference_mode():
            subset = self.current["data"][self.current["from"] : self.current["to"]]

            outs = []
            labels = []
            for X, y in subset["batch"].values:
                outs.append(self.current["model"](X.unsqueeze(0)).squeeze(-1))
                labels.append(y)

            outs = np.array(outs)
            labels = np.array(labels)

            # Denormalize
            labels = self.current["encoder"]["label"].inverse_transform(
                labels.reshape(-1, 1)
            )
            outs = self.current["encoder"]["label"].inverse_transform(outs)

        for y in [outs, labels]:
            sns.lineplot(x=subset.reset_index()["datetime"], y=y.ravel())

        plt.xlabel("Time")
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
                    choices=self.dataset, label="Dataset", interactive=True
                )
                button = gr.Button("Run")

            with gr.Column():
                gr.Markdown("### From")
                with gr.Row():
                    fday = gr.Dropdown(
                        choices=range(1, 32), label="Day", min_width=0, interactive=True
                    )
                    fmonth = gr.Dropdown(
                        choices=range(1, 13),
                        label="Month",
                        min_width=0,
                        interactive=True,
                    )
                    fyear = gr.Dropdown(label="Year", min_width=0, interactive=True)

                gr.Markdown("### To")
                with gr.Row():
                    tday = gr.Dropdown(
                        choices=range(1, 32),
                        value=8,
                        label="Day",
                        min_width=0,
                        interactive=True,
                    )
                    tmonth = gr.Dropdown(
                        choices=range(1, 13),
                        label="Month",
                        min_width=0,
                        interactive=True,
                    )
                    tyear = gr.Dropdown(label="Year", min_width=0, interactive=True)

        gr.Markdown("### Result")
        with gr.Row():
            plot = gr.Plot(show_label=False)

        for key, values in {
            "from": [fyear, fmonth, fday],
            "to": [tyear, tmonth, tday],
        }.items():
            for value in values:
                value.select(
                    partial(self._select_time, indicator=key), values, values[-1]
                )

        button.click(self._predict, None, plot)

        data_dropdown.select(self._select_data, data_dropdown, [plot, fyear, tyear])
        model_dropdown.select(self._select_model, model_dropdown)

        self.parent.load(self._select_data, data_dropdown, [plot, fyear, tyear])
        self.parent.load(self._select_model, model_dropdown)
