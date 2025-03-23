from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.datasets = sorted(
            [i.name for i in self.data_path.glob("*.csv") if not i.match("*.x.csv")]
        )
        self.current = {}

    def _change_data(self, data_name):
        df = pd.read_csv(str(self.data_path / data_name))
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        self.current["data"] = df
        self.current["summary"] = self._summary(df)

        return gr.Tabs(selected=0), self.current["data"]

    def _count(self):
        return len(self.current["data"]), len(self.current["data"].columns)

    def _summary(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = pd.to_timedelta(df["hour"], unit="h")
        df["date"] = df["date"] + df["hour"]
        df = df.rename(columns={"date": "datetime"})
        df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

        df = df.set_index("datetime")

        # Select numerical columns only
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return None

        df = df[num_cols]

        summary = df.describe().T.reset_index()
        summary = summary.rename(columns={"index": "Statistic"})

        return summary.round(4)

    def _heatmap(self):
        df = self.current["summary"][["Statistic", "mean", "std", "min", "max"]]
        df = df.set_index("Statistic")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)

        fig.subplots_adjust(left=0.2, right=1, top=0.85, bottom=0.15)

        ax.set_title("Descriptive Statistics", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Statistic", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Feature", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _boxplot(self):
        df = self.current["summary"][["Statistic", "25%", "50%", "75%"]]

        df_melted = df.melt(
            id_vars=["Statistic"], var_name="Percentile", value_name="Value"
        )

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.boxplot(
            df_melted, x="Statistic", y="Value", ax=ax, hue="Statistic", palette="Set2"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Distribution Percentiles", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Feature", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Value", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _barplot(self):
        df = self.current["summary"][["Statistic", "mean"]]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        sns.barplot(
            df, x="Statistic", y="mean", ax=ax, hue="Statistic", palette="viridis"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Mean Value", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Feature", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Value", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def __call__(self):
        data_dropdown = gr.Dropdown(
            choices=map(str, self.datasets), label="Dataset", interactive=True
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Review", id=0):
                df = gr.Dataframe(max_height=800)

                data_dropdown.change(self._change_data, data_dropdown, [tabs, df])
                self.parent.load(self._change_data, data_dropdown, [tabs, df])

            with gr.Tab("Overview", id=1) as t1:
                gr.Markdown("## Dataset Overview")
                with gr.Row():
                    n_rows = gr.Label(label="Number of data points")
                    n_colums = gr.Label(label="Number of features")

                gr.Markdown("## Feature Statistics")
                df = gr.Dataframe()

                gr.Markdown("---")
                gr.Markdown("## Descriptive Statistics")
                gr.Markdown(
                    "A visual representation of core statistical measures, offering insights into data distribution, variability, and overall trends. "
                )
                plt_heatmap = gr.Plot()

                gr.Markdown("---")
                gr.Markdown("## Distribution Percentiles")
                gr.Markdown(
                    "Illustrate the spread and distribution of each feature, highlighting key percentiles, quartile ranges, and potential outliers."
                )
                plt_box = gr.Plot()

                gr.Markdown("---")
                gr.Markdown("## Mean Value")
                gr.Markdown("A comparison of the average values of different features.")
                plt_bar = gr.Plot()

                t1.select(self._count, None, [n_rows, n_colums])
                t1.select(lambda: self.current["summary"], None, df)
                t1.select(self._heatmap, None, plt_heatmap)
                t1.select(self._boxplot, None, plt_box)
                t1.select(self._barplot, None, plt_bar)
