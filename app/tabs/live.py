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

from qlstm.models.cLSTM import cLSTM
from qlstm.models.cQLSTMf import cQLSTMf
from qlstm.models.LSTM import LSTM
from qlstm.modules.data import CustomDataModule
from qlstm.modules.model import LitModel


class LiveTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.model_path = self.root / "models"
        self.models = {
            "LSTM": {
                "init": LSTM(9, 128),
                "checkpoint": "lightning_logs/LSTM/base/checkpoints/last.ckpt",
            },
            "cLSTM": {
                "init": cLSTM(9, 128),
                "checkpoint": "lightning_logs/cLSTM/version_0/checkpoints/epoch=14-step=17910.ckpt",
            },
            "cQLSTMf": {
                "init": cQLSTMf(9, 128, 2),
                "checkpoint": "lightning_logs/cQLSTMf/version_9/checkpoints/last.ckpt",
            },
        }
        self.current = {}

    def _select_model(self, model_name):
        model = self.models[model_name]
        self.current["model"] = LitModel(
            model["init"], checkpoint=model["checkpoint"]
        ).eval()

    def _predict(self):
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 5), dpi=200)

        subset = self.current["data"][:100].reset_index()

        if subset.empty:
            return fig

        outs = []
        labels = []

        with torch.inference_mode():
            for X, y in subset["batch"]:
                outs.append(self.current["model"](X.unsqueeze(0)).squeeze(-1))
                labels.append(y)

        # Denormalize
        outs = self.current["decoder"](np.array(outs))
        labels = self.current["decoder"](np.array(labels))

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

        data = pd.DataFrame(
            {"Actual": labels_tilda.ravel(), "Predicted": outs_tilda.ravel()}
        )

        actual = pd.DataFrame({"Value": data["Actual"], "Label": "Actual"})
        predicted = pd.DataFrame({"Value": data["Predicted"], "Label": "Predicted"})

        data = pd.concat([actual[:50], predicted[50:100]], ignore_index=True)
        data["Time"] = times[:100]
        data["Time"] = data["Time"].dt.tz_localize("Asia/Bangkok")

        join = data[49:50].copy()
        join["Label"] = "Predicted"

        data = pd.concat([data[:50], join, data[50:100]], ignore_index=True)

        fig = gr.LinePlot(data, x="Time", y="Value", color="Label")

        # for k, v in [("Actual", labels_tilda), ("Predicted", outs_tilda)]:
        #     sns.lineplot(x=times, y=v.ravel(), ax=ax, label=k)

        # fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

        # ax.set_title("PV Power Output", fontsize=16, fontweight="bold", pad=14)
        # ax.set_xlabel("Time", fontsize=14, fontweight="bold", labelpad=14)
        # ax.set_ylabel("Power (kW)", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _update_model(self, model_name):
        self._select_model(model_name)
        plot = self._predict()

        return plot

    def _init(self, model_name):
        data = CustomDataModule(
            data_path="qlstm/data/Albany_WA.csv",
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

        self.current["decoder"] = data.encoder["label"].inverse_transform

        self._select_model(model_name)
        plot = self._predict()

        return plot

    def __call__(self):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    model_dd = gr.Dropdown(
                        choices=self.models.keys(), label="Model", interactive=True
                    )
                    version_dd = gr.Dropdown(
                        ["1", "2"], label="Version", interactive=True
                    )

            # with gr.Column(scale=1):
            #     gr.DateTime(label="From")
            #     gr.DateTime(label="From")

        gr.Markdown("### PV power")
        with gr.Row():
            group_dd = gr.Dropdown(
                [
                    ("Hour", "None"),
                    ("Day", "D"),
                    ("Week", "W"),
                    ("Month", "ME"),
                    ("Year", "YE"),
                ],
                label="Group by",
                interactive=True,
            )
            forecast_dd = gr.Slider(
                minimum=0,
                maximum=6,
                step=1,
                label="Forecasting",
                interactive=True,
            )

        plot = gr.LinePlot(show_label=False)

        model_dd.select(self._update_model, model_dd, plot)
        self.parent.select(self._init, model_dd, plot)
