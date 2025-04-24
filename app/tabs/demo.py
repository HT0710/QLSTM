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
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
rootutils.autosetup(".gitignore")

from qlstm.models.cLSTM import cLSTM
from qlstm.models.cQLSTM import cQLSTM
from qlstm.models.LSTM import LSTM
from qlstm.models.QLSTM import QLSTM
from qlstm.modules.data import CustomDataModule
from qlstm.modules.model import LitModel


class DemoTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.model_path = self.root / "models"
        self.dataset = sorted([i.name for i in self.data_path.glob("*.csv")])
        self.models = {
            "LSTM": {
                "init": LSTM(9, 128),
                "checkpoint": "lightning_logs/LSTM/base/checkpoints/last.ckpt",
            },
            "cLSTM": {
                "init": cLSTM(9, 128),
                "checkpoint": "lightning_logs/cLSTM/base/checkpoints/last.ckpt",
            },
            "QLSTM_2q": {
                "init": QLSTM(9, 128, n_qubits=2),
                "checkpoint": "lightning_logs/QLSTM/2q/checkpoints/last.ckpt",
            },
            "QLSTM_4q": {
                "init": QLSTM(9, 128, n_qubits=4),
                "checkpoint": "lightning_logs/QLSTM/4q/checkpoints/last.ckpt",
            },
            "cQLSTM_2q": {
                "init": cQLSTM(9, 128, n_qubits=2),
                "checkpoint": "lightning_logs/cQLSTM/2q_post_2/checkpoints/last.ckpt",
            },
            "cQLSTM_4q": {
                "init": cQLSTM(9, 128, n_qubits=4),
                "checkpoint": "lightning_logs/cQLSTM/4q_post_2/checkpoints/last.ckpt",
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
            features="6-11",
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
        self.current["decoder"] = data.encoder["label"].inverse_transform

        years = datetime.year
        year_range = range(years.min(), years.max() + 1)

        self.current["from"] = datetime.min()
        self.current["to"] = datetime.min() + pd.DateOffset(days=7)

        return (
            gr.update(choices=year_range, value=int(years.min())),
            gr.update(choices=year_range, value=int(years.min())),
        )

    def _select_model(self, model_name):
        model = self.models[model_name]
        self.current["model"] = LitModel(
            model["init"], checkpoint=model["checkpoint"], device="cpu"
        ).eval()

    def _select_time(self, y, m, d, indicator):
        _, new_num_days = monthrange(y, m)
        d = min(d, new_num_days)

        self.current[indicator] = datetime(y, m, d)

        return gr.update(choices=range(1, new_num_days + 1), value=d)

    def _predict(self):
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 5), dpi=200)

        subset = self.current["data"][
            self.current["from"] : self.current["to"]
        ].reset_index()

        if subset.empty:
            return fig

        outs = []
        labels = []

        with torch.inference_mode():
            for X, y in subset["batch"]:
                outs.append(self.current["model"](X.unsqueeze(0)).squeeze(-1))
                labels.append(y)

        # Denormalize
        outs = torch.tensor(self.current["decoder"](np.array(outs)))
        labels = torch.tensor(self.current["decoder"](np.array(labels)))

        self.current["metrics"] = {
            "R2": f"{r2_score(outs, labels):.2f} %",
            "MAE": f"{mean_absolute_error(outs, labels):.2f} kWh",
            "RMSE": f"{mean_squared_error(outs, labels, squared=False):.2f} kWh",
        }

        # Denormalize
        outs = outs.numpy()
        labels = labels.numpy()

        times = []
        outs_tilda = []
        labels_tilda = []

        for i in range(len(subset["datetime"])):
            if i == 0:
                times.append(subset["datetime"][i])
                outs_tilda.append(outs[i])
                labels_tilda.append(labels[i])
                continue

            delta = subset["datetime"][i] - subset["datetime"][i - 1]

            for j in range(1, delta.components.hours):
                times.append(subset["datetime"][i - 1] + pd.Timedelta(hours=j))
                outs_tilda.append(torch.zeros(1))
                labels_tilda.append(torch.zeros(1))

            times.append(subset["datetime"][i])
            outs_tilda.append(outs[i])
            labels_tilda.append(labels[i])

        times, outs_tilda, labels_tilda = map(
            np.array, [times, outs_tilda, labels_tilda]
        )

        for k, v in [("Actual", labels_tilda), ("Predicted", outs_tilda)]:
            sns.lineplot(x=times, y=v.ravel(), ax=ax, label=k)

        fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

        ax.set_title("PV Power Output", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Time", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Power (kW)", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _update_time(self, y, m, d, indicator):
        date = self._select_time(y, m, d, indicator)
        plot = self._predict()

        return plot, date

    def _update_data(self, data_name):
        fyear, tyear = self._select_data(data_name)
        plot = self._predict()

        return plot, fyear, tyear

    def _update_model(self, model_name):
        self._select_model(model_name)
        plot = self._predict()

        return plot

    def _init(self, data_name, model_name):
        fyear, tyear = self._select_data(data_name)
        self._select_model(model_name)
        plot = self._predict()

        return plot, fyear, tyear

    def __call__(self):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Options")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=self.models.keys(), label="Model", interactive=True
                    )
                    data_dropdown = gr.Dropdown(
                        choices=self.dataset, label="Dataset", interactive=True
                    )

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

            with gr.Column():
                gr.Markdown("### To")
                with gr.Row():
                    tday = gr.Dropdown(
                        choices=range(1, 32),
                        value=7,
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
            r2_lb = gr.Label(label="R²")
            mae_lb = gr.Label(label="MAE")
            rmse_lb = gr.Label(label="RMSE")

        plot = gr.Plot(show_label=False)

        plot.change(
            lambda: [i for i in self.current["metrics"].values()],
            None,
            [r2_lb, mae_lb, rmse_lb],
        )

        for key, values in {
            "from": [fyear, fmonth, fday],
            "to": [tyear, tmonth, tday],
        }.items():
            for block in values:
                block.select(
                    partial(self._update_time, indicator=key),
                    values,
                    [plot, values[-1]],
                )

        data_dropdown.select(self._update_data, data_dropdown, [plot, fyear, tyear])
        model_dropdown.select(self._update_model, model_dropdown, plot)

        self.parent.select(
            self._init, [data_dropdown, model_dropdown], [plot, fyear, tyear]
        )
