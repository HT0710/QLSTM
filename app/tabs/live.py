import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import rootutils
import torch

matplotlib.use("Agg")
rootutils.autosetup(".gitignore")

from common.models import MODELS
from qlstm.modules.data import CDM_Hour, CDM_Day, CDM_Month
from qlstm.modules.model import LitModel


class LiveTab:
    def __init__(self, parent):
        self.parent = parent
        self.models = MODELS
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
        subset = self.current["data"][self.current["i"] : self.current["i"] + 23]

        outs = []
        labels = []

        for X, y in subset["batch"]:
            outs.append(X)

            y = self.current["decoder"]([y])

            labels.append(y)

        X = torch.stack(outs)

        with torch.inference_mode():
            outs = self.current["decoder"](self.current["model"](X).detach())

        outs = np.clip(outs, a_min=0, a_max=None).astype(int).ravel()
        labels = np.array(labels).astype(int).ravel()

        for i, label in enumerate(labels):
            if label == 0:
                outs[i] = 0

        actual = pd.DataFrame({"Value": labels, "Label": "Actual"})
        predicted = pd.DataFrame({"Value": outs, "Label": "Forecasted"})

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

        passed = self.df.iloc[self.current["i"]]

        time = str(passed.iloc[0])
        values = map(lambda x: str(round(x, 2)), passed.iloc[1:].tolist())

        with open("app/history.csv", "+a") as f:
            f.write(time + "," + ",".join(values) + "\n")

        return fig

    def _update_model(self, model_name):
        self._select_model(model_name)
        plot = self._predict()

        return plot

    def _init(self, model_name):
        if self.current["i"] == 0:
            data = CDM_Hour(
                data_path="qlstm/data/Albany_WA.csv",
                features="6-11",
                labels=[11],
                time_steps=24,
                overlap=True,
            )
            data.prepare_data()
            data.setup("predict")

            df = data.dataframe.copy()

            # Drop unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]

            # Convert 'date' and 'hour' columns to datetime type
            df["date"] = pd.to_datetime(df["date"])
            df["hour"] = pd.to_timedelta(df["hour"], unit="h")
            df["date"] = df["date"] + df["hour"]
            df = df.rename(columns={"date": "datetime"})
            df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

            df = df.sort_values(by="datetime")

            # Fill missing hours
            df = data._fill_hours(df, start=6, end=18)
            df.interpolate(method="linear", inplace=True)
            df = data._fill_hours(df, value=0)

            self.df = df

            with open("app/history.csv", "+w") as f:
                f.write(",".join(df.columns) + "\n")

            df = pd.DataFrame(
                [{"datetime": t, "batch": b} for t, b in zip(data.index, data.dataset)]
            )

            self.current["data"] = df.set_index("datetime")

            self.current["encoder"] = data.encoder["input"].inverse_transform
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
