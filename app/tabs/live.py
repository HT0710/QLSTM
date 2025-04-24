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
from qlstm.models.cQLSTM import cQLSTM
from qlstm.models.LSTM import LSTM
from qlstm.modules.data import CDM_Hour, CDM_Day, CDM_Month
from qlstm.modules.model import LitModel


class LiveTab:
    def __init__(self, parent):
        self.parent = parent
        self.models = {
            "LSTM": {
                "init": LSTM(9, 128),
                "checkpoint": "lightning_logs/LSTM/base/checkpoints/last.ckpt",
            },
            "cLSTM": {
                "init": cLSTM(9, 128),
                "checkpoint": "lightning_logs/cLSTM/base/checkpoints/last.ckpt",
            },
            "cQLSTM": {
                "init": cQLSTM(9, 128, n_qubits=2),
                "checkpoint": "lightning_logs/cQLSTM/2q_post_2/checkpoints/last.ckpt",
            },
        }
        self.current = {"i": 0, "data": None, "model": None, "group": "hour"}

    def _select_group(self, group):
        self.current["group"] = group

        match group:
            case "hour":
                data = CDM_Hour(
                    data_path="qlstm/data/Albany_WA.csv",
                    features="6-11",
                    labels=[11],
                    time_steps=24,
                    overlap=True,
                )

            case "day":
                data = CDM_Day(
                    data_path="qlstm/data/Albany_WA.csv",
                    features="6-11",
                    labels=[11],
                    time_steps=7,
                    overlap=True,
                )

            case "month":
                data = CDM_Month(
                    data_path="qlstm/data/Albany_WA.csv",
                    features="6-11",
                    labels=[11],
                    time_steps=4,
                    overlap=True,
                )

        data.prepare_data()
        data.setup("predict")

        df = pd.DataFrame(
            [{"datetime": t, "batch": b} for t, b in zip(data.index, data.dataset)]
        )

        self.current["data"] = df.set_index("datetime")

        self.current["decoder"] = data.encoder["label"].inverse_transform

    def _select_model(self, model_name):
        model = self.models[model_name]
        self.current["model"] = LitModel(
            model["init"], checkpoint=model["checkpoint"], device="cpu"
        ).eval()

    def _predict(self):
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 5), dpi=200)

        subset = self.current["data"][self.current["i"] : self.current["i"] + 23]

        if subset.empty:
            return fig

        outs = []
        labels = []

        with torch.inference_mode():
            for X, y in subset["batch"]:
                y = self.current["decoder"]([y])

                if y <= 1:
                    out = y
                else:
                    out = self.current["decoder"](self.current["model"](X.unsqueeze(0)))

                outs.append(max(np.zeros((1, 1)), out))
                labels.append(y)

        outs = np.array(outs)
        labels = np.array(labels)

        data = pd.DataFrame({"Actual": labels.ravel(), "Forecasted": outs.ravel()})

        actual = pd.DataFrame({"Value": data["Actual"], "Label": "Actual"})
        predicted = pd.DataFrame({"Value": data["Forecasted"], "Label": "Forecasted"})

        data = pd.concat([actual[:12], predicted[12:23]], ignore_index=True)

        data["Time"] = subset.index[:23].tz_localize("Asia/Ho_Chi_Minh")

        self.current["join"] = join = data[11:12].copy()
        join["Label"] = "Forecasted"

        data = pd.concat([data[:12], join, data[12:23]], ignore_index=True)

        fig = gr.LinePlot(
            data,
            x="Time",
            y="Value",
            x_title=f"Time ({self.current['group']})",
            y_title="Energy (kWh)",
            color="Label",
        )

        return fig

    def _update_model(self, model_name):
        self._select_model(model_name)
        plot = self._predict()

        return plot

    def _init(self, model_name):
        data = CDM_Hour(
            data_path="qlstm/data/Albany_WA.csv",
            features="6-11",
            labels=[11],
            time_steps=24,
            overlap=True,
        )
        data.prepare_data()
        data.setup("predict")

        df = pd.DataFrame(
            [{"datetime": t, "batch": b} for t, b in zip(data.index, data.dataset)]
        )

        self.current["data"] = df.set_index("datetime")

        self.current["decoder"] = data.encoder["label"].inverse_transform

        self._select_model(model_name)

        return self._predict(), gr.update(active=True)

    def _plus(self):
        self.current["i"] += 1

        match self.current["group"]:
            case "hour":
                time_format = "%H:%M - %a, %d %b %Y"

            case "day":
                time_format = "%A, %d %b %Y"

            case "month":
                time_format = "%B %Y"

        return (
            self._predict(),
            f"{self.current['join']['Time'].dt.strftime(time_format).values[0]}",
            f"{round(self.current['join']['Value'].values[0]):,} kWh",
        )

    def __call__(self):
        gr.Markdown("### Current")
        with gr.Row():
            time_lb = gr.Label(label="Time")
            power_lb = gr.Label(label="Energy", scale=2)

        gr.Markdown("### PV power")
        plot = gr.LinePlot(show_label=False)

        with gr.Row():
            group_dd = gr.Dropdown(
                [
                    ("Hour", "hour"),
                    ("Day", "day"),
                    ("Month", "month"),
                ],
                label="Group by",
                interactive=True,
            )
            model_dd = gr.Dropdown(
                choices=self.models.keys(), label="Model", interactive=True
            )

        model_dd.select(self._update_model, model_dd, plot)

        group_dd.select(self._select_group, group_dd, None)

        time = gr.Timer(1, active=False)

        time.tick(self._plus, None, [plot, time_lb, power_lb])

        self.parent.select(self._init, model_dd, [plot, time])
