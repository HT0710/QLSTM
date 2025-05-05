from calendar import monthrange
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import rootutils
import torch
from matplotlib import pyplot as plt
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
rootutils.autosetup(".gitignore")

from common.models import MODELS

from qlstm.modules.data import CustomDataModule
from qlstm.modules.model import LitModel


class DemoTab:
    def __init__(self, parent):
        self.parent = parent
        self.data_path = Path("./qlstm/data")
        self.datasets = sorted([i.name for i in self.data_path.glob("*.csv")])
        self.models = MODELS
        self.current = {
            "data": None,
            "from": None,
            "to": None,
        }

    def _faq(self):
        gr.Markdown("### FAQ")

        with gr.Accordion("1. What is R² (Coefficient of Determination)?", open=False):
            gr.Markdown(r"""
                Measures how much variance in the target variable is explained by the model.  

                - **R² = 1.0** → Perfect prediction.
                - **R² = 0.0** → Model predicts the mean.
                - **R² < 0.0** → Worse than predicting the mean.

                **Example:** A model predicting achieves **R² = 0.85**, meaning it explains 85% of the variance.
                
                **Formula:**  
                $$ R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} $$
            """)

        with gr.Accordion("2. What is MAE (Mean Absolute Error)?", open=False):
            gr.Markdown(r"""
                Measures the average absolute difference between predicted and true values.  

                - Lower MAE = better.
                - Same units as the output variable.

                **Example:** MAE = **5.3** → Predictions are off by about **5.3 units** on average.

                **Formula:**  
                $$ MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i | $$
            """)

        with gr.Accordion("3. What is RMSE (Root Mean Squared Error)?", open=False):
            gr.Markdown(r"""
                Measures the square root of the average squared difference between predictions and actuals.  

                - Lower RMSE = better.
                - Same units as output variable.

                **Example:** RMSE = **7.2** → Predictions are off by about **7.2 units**, especially penalizing large mistakes.

                **Formula:**  
                $$ RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } $$
            """)

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

        subset = self.current["data"][
            self.current["from"] : self.current["to"]
        ].reset_index()

        if subset.empty:
            return None

        outs = []
        labels = []

        for X, y in subset["batch"]:
            outs.append(X)

            y = self.current["decoder"]([y])

            labels.append(y.squeeze(0))

        outs = torch.stack(outs)

        with torch.inference_mode():
            outs = self.current["decoder"](self.current["model"](outs).detach())

        outs = torch.tensor(outs)
        labels = torch.tensor(np.array(labels))

        self.current["metrics"] = {
            "R2": f"{r2_score(outs, labels):.2f} %",
            "MAE": f"{mean_absolute_error(outs, labels):.2f} kWh",
            "RMSE": f"{mean_squared_error(outs, labels, squared=False):.2f} kWh",
        }

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

        actual = pd.DataFrame(
            {
                "Time": times,
                "Value": labels_tilda.ravel(),
                "Label": "Actual",
            }
        )
        predicted = pd.DataFrame(
            {
                "Time": times,
                "Value": outs_tilda.ravel(),
                "Label": "Forecasted",
            }
        )

        data = pd.concat([actual, predicted], ignore_index=True)

        return (
            gr.update(
                value=data,
                x="Time",
                y="Value",
                x_title="Time",
                y_title="Energy (kWh)",
                color="Label",
            ),
            *[i for i in self.current["metrics"].values()],
        )

    def _update_time(self, y, m, d, indicator):
        date = self._select_time(y, m, d, indicator)
        plot, r2_lb, mae_lb, rmse_lb = self._predict()

        return plot, r2_lb, mae_lb, rmse_lb, date

    def _update_data(self, data_name):
        fyear, tyear = self._select_data(data_name)
        plot, r2_lb, mae_lb, rmse_lb = self._predict()

        return plot, r2_lb, mae_lb, rmse_lb, fyear, tyear

    def _update_model(self, model_name):
        self._select_model(model_name)
        plot, r2_lb, mae_lb, rmse_lb = self._predict()

        return plot, r2_lb, mae_lb, rmse_lb

    def _init(self, data_name, model_name):
        fyear, tyear = self._select_data(data_name)
        self._select_model(model_name)
        plot, r2_lb, mae_lb, rmse_lb = self._predict()

        return plot, r2_lb, mae_lb, rmse_lb, fyear, tyear

    def __call__(self):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Options")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=sorted(self.models.keys()),
                        label="Model",
                        interactive=True,
                    )
                    data_dropdown = gr.Dropdown(
                        choices=self.datasets, label="Dataset", interactive=True
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

        plot = gr.LinePlot(show_label=False)

        self._faq()

        ### EVENTS ###
        for key, values in {
            "from": [fyear, fmonth, fday],
            "to": [tyear, tmonth, tday],
        }.items():
            for block in values:
                block.select(
                    partial(self._update_time, indicator=key),
                    values,
                    [plot, r2_lb, mae_lb, rmse_lb, values[-1]],
                )

        data_dropdown.select(
            self._update_data,
            data_dropdown,
            [plot, r2_lb, mae_lb, rmse_lb, fyear, tyear],
        )
        model_dropdown.select(
            self._update_model, model_dropdown, [plot, r2_lb, mae_lb, rmse_lb]
        )

        self.parent.select(
            self._init,
            [data_dropdown, model_dropdown],
            [plot, r2_lb, mae_lb, rmse_lb, fyear, tyear],
        )
