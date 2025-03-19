from pathlib import Path

import gradio as gr
import pandas as pd
from scipy.stats import kurtosis


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.datasets = self.data_path.glob("*.csv")

    def _load_data(self, data_name):
        return pd.read_csv(str(self.data_path / data_name), index_col=0)

    def _change_mode(self, mode):
        if mode == "Review":
            return gr.update(visible=True), gr.update(visible=False)
        if mode == "EDA":
            return gr.update(visible=False), gr.update(visible=True)

    def _eda(self, mode, data_name):
        if mode == "Summary Statistics":
            df = self._load_data(data_name)
            df["date"] = pd.to_datetime(df["date"])
            df["hour"] = pd.to_timedelta(df["hour"], unit="h")
            df["date"] = df["date"] + df["hour"]
            df = df.rename(columns={"date": "datetime"})
            df.drop(["year", "hour"], axis=1, inplace=True)

            df.index = df["datetime"]

            # Select numerical columns only
            num_cols = df.select_dtypes(include=["number"]).columns
            if len(num_cols) == 0:
                return "No numerical columns found in the dataset."

            df = df[num_cols]  # Keep only numerical columns

            summary = df.describe().T
            summary["skewness"] = df.skew()
            summary["kurtosis"] = df.apply(kurtosis)

            return gr.update(visible=True), summary.round(4)
        else:
            return gr.update(visible=False), None

    def __call__(self):
        with gr.Tabs(selected=1):
            with gr.Tab("Overview", id=0):
                gr.Markdown("# Comming Soon")

            with gr.Tab("Detail", id=1):
                with gr.Row():
                    data_dropdown = gr.Dropdown(
                        choices=[str(x.name) for x in self.datasets],
                        label="Dataset",
                        interactive=True,
                    )
                    display_dropdown = gr.Dropdown(
                        choices=["Review", "EDA"],
                        label="Display",
                        interactive=True,
                    )
                    eda_dropdown = gr.Dropdown(
                        choices=["All", "Summary Statistics"],
                        label="Filter",
                        visible=False,
                        interactive=True,
                    )

                df = gr.Dataframe(max_height=800)

                data_dropdown.change(self._load_data, [data_dropdown], df, queue=False)
                display_dropdown.change(
                    self._change_mode,
                    [display_dropdown],
                    [df, eda_dropdown],
                    queue=False,
                )
                eda_dropdown.change(
                    self._eda, [eda_dropdown, data_dropdown], [df, df], queue=False
                )

        self.parent.load(self._load_data, [data_dropdown], df, queue=False)
