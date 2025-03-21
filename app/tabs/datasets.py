from pathlib import Path

import gradio as gr
import pandas as pd


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.datasets = [
            i for i in self.data_path.glob("*.csv") if not i.match("*.x.csv")
        ]
        self.loaded = None

    def _change_data(self, data_name):
        data = pd.read_csv(str(self.data_path / data_name))
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
        self.loaded = data

        return gr.Tabs(selected=0), data

    def _count(self):
        return len(self.loaded), len(self.loaded.columns)

    def _eda(self):
        df = self.loaded.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = pd.to_timedelta(df["hour"], unit="h")
        df["date"] = df["date"] + df["hour"]
        df = df.rename(columns={"date": "datetime"})
        df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

        df.index = df["datetime"]

        # Select numerical columns only
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return "No numerical columns found in the dataset."

        df = df[num_cols]  # Keep only numerical columns

        summary = df.describe().T.reset_index()
        summary = summary.rename(columns={"index": "Statistic"})

        # summary["skewness"] = df.skew()
        # summary["kurtosis"] = df.apply(kurtosis)

        return summary.round(4)

    def __call__(self):
        data_dropdown = gr.Dropdown(
            choices=map(str, self.datasets),
            label="Dataset",
            interactive=True,
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Review", id=0):
                df = gr.Dataframe(max_height=800)

                data_dropdown.change(self._change_data, data_dropdown, [tabs, df])
                self.parent.load(self._change_data, data_dropdown, [tabs, df])

            with gr.Tab("Overview", id=1) as t1:
                with gr.Row():
                    n_rows = gr.Label(label="Number of data points")
                    n_colums = gr.Label(label="Number of features")

                df = gr.Dataframe()

                t1.select(self._count, None, [n_rows, n_colums])
                t1.select(self._eda, None, df)
