from calendar import monthrange
from functools import partial
from pathlib import Path
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.datasets = sorted([i.name for i in self.data_path.glob("*.csv")])
        self.current = {
            "data": None,
            "processed": None,
            "group": "None",
            "from": None,
            "to": None,
        }

    def _show_data(self, df):
        if self.current["group"] != "None":
            df = df.resample(self.current["group"]).mean()

            match self.current["group"]:
                case "D" | "W":
                    df.index = df.index.to_period("D")
                case "ME":
                    df.index = df.index.to_period("M")
                case "YE":
                    df.index = df.index.to_period("Y")

        df = df.loc[self.current["from"] : self.current["to"]]

        return gr.update(
            value=df.round(2).reset_index(), label=f"Number of Rows: {len(df)}"
        )

    def _select_data(self, data_name):
        df = pd.read_csv(str(self.data_path / data_name))

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Format datetime
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = pd.to_timedelta(df["hour"], unit="h")
        df["date"] = df["date"] + df["hour"]
        df = df.rename(columns={"date": "Datetime"})
        df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

        # Sort and set "Datetime" as index
        df = df.sort_values(by="Datetime")
        df = df.set_index("Datetime")

        # Select numerical columns only
        num_cols = df.select_dtypes(include=["number"]).columns
        df = df[num_cols] if len(num_cols) != 0 else None

        self.current["group"] = "None"
        self.current["processed"] = self.current["data"] = df

        date_min = self.current["from"] = self.current["data"].index.min()
        date_max = self.current["to"] = self.current["data"].index.max()

        year_range = range(date_min.year, date_max.year + 1)

        min_month_range = range(date_min.month, 13)
        max_month_range = range(1, date_max.month + 1)

        _, min_num_days = monthrange(date_min.year, date_min.month)
        min_day_range = range(date_min.day, min_num_days + 1)

        max_day_range = range(1, date_max.day + 1)

        return (
            self._show_data(self.current["data"]),
            gr.update(choices=year_range, value=date_min.year),
            gr.update(choices=min_month_range, value=date_min.month),
            gr.update(choices=min_day_range, value=date_min.day),
            gr.update(choices=year_range, value=date_max.year),
            gr.update(choices=max_month_range, value=date_max.month),
            gr.update(choices=max_day_range, value=date_max.day),
        )

    def _select_group(self, group):
        self.current["group"] = group

        return self._show_data(self.current["processed"])

    def _select_time(self, y, m, d, indicator):
        _, new_num_days = monthrange(y, m)
        d = min(d, new_num_days)

        self.current[indicator] = datetime(y, m, d)

        return (
            gr.update(choices=range(1, new_num_days + 1), value=d),
            self._show_data(self.current["processed"]),
        )

    def _fill_missing_hours(self, state):
        def fill_hours(df: pd.DataFrame, start, end):
            filled_rows = []
            prev_date = None
            prev_hour = start - 1

            def add_rows(date, start, end):
                new_rows = []
                for h in range(start, end):
                    row = {}
                    for col in df.columns:
                        if col == "Datetime":
                            row[col] = date + pd.DateOffset(hours=h)
                        else:
                            row[col] = np.nan
                    new_rows.append(row)
                filled_rows.extend(new_rows)

            for _, row in df.iterrows():
                curr_date = row["Datetime"].date()
                curr_hour = row["Datetime"].hour

                if curr_hour < prev_hour:
                    add_rows(prev_date, prev_hour + 1, end + 1)
                    add_rows(curr_date, start, curr_hour)
                else:
                    add_rows(curr_date, prev_hour + 1, curr_hour)

                # Append the current row
                filled_rows.append(row.to_dict())

                # Update previous hour
                prev_hour = curr_hour
                prev_date = curr_date

            return pd.DataFrame(filled_rows)

        if state == "On":
            df = fill_hours(self.current["data"].reset_index(), 6, 18)

            df.interpolate(method="linear", inplace=True)

            self.current["processed"] = df.set_index("Datetime")

        else:
            self.current["processed"] = self.current["data"]

        return self._show_data(self.current["processed"])

    def _moving_average(self, size: int):
        if size > 1:
            self.current["processed"] = (
                self.current["processed"].rolling(window=size, min_periods=1).mean()
            )

        return self._show_data(self.current["processed"])

    def _select_vis(self, df, mode):
        plot = y_axis = None

        if mode == "On":
            y_axis = gr.update(
                choices=[c for c in df.columns if c != "Datetime"],
                value=[df.columns[-1]],
            )

            plot = self._vis_plot(df, mode, y_axis["value"])

        return (
            gr.update(visible=mode == "Off"),
            gr.update(visible=mode == "On"),
            plot,
            y_axis,
        )

    def _vis_plot(self, df, mode, y):
        if not (mode == "On" or y):
            return None

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        for feature in y:
            sns.lineplot(x=pd.to_datetime(df["Datetime"]), y=df[feature], ax=ax)

        fig.subplots_adjust(top=0.95, bottom=0.15)

        ax.set_xlabel("Time", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.describe().T.round(2).reset_index()
        summary = summary.rename(columns={"index": "Statistic"})

        return summary

    def _heatmap(self, df):
        df = df[["Statistic", "mean", "std", "min", "max"]]
        df = df.set_index("Statistic")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)

        fig.subplots_adjust(left=0.2, right=1, top=0.85, bottom=0.15)

        ax.set_title("Descriptive Statistics", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Statistics", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Features", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _boxplot(self, df):
        df = df[["Statistic", "25%", "50%", "75%"]]

        df_melted = df.melt(
            id_vars=["Statistic"], var_name="Percentile", value_name="Value"
        )

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.boxplot(
            df_melted, x="Statistic", y="Value", ax=ax, hue="Statistic", palette="Set2"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Distribution Percentiles", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Features", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _barplot(self, df):
        df = df[["Statistic", "mean"]]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        sns.barplot(
            df, x="Statistic", y="mean", ax=ax, hue="Statistic", palette="viridis"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Mean Value", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Features", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _select_corr(self):
        df = self.current["processed"]

        corr = df.corr().round(2).reset_index()
        corr = corr.rename(columns={"index": "Features"})

        features = list(df.columns)

        return (
            corr,
            gr.update(choices=features, value=features[0]),
            gr.update(choices=features, value=features[-1]),
            self._corr_pairwise(features[0], features[-1]),
        )

    def _corr_heatmap(self, df):
        df = df.set_index("Features")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)

        fig.subplots_adjust(left=0.25, right=0.95, top=0.85, bottom=0.25)

        ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Features", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Features", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _corr_pairwise(self, x, y):
        df = self.current["processed"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        sns.scatterplot(df, x=x, y=y, ax=ax)

        fig.subplots_adjust(top=0.95, bottom=0.15)

        ax.set_xlabel(x, fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel(y, fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def __call__(self):
        with gr.Row(equal_height=True):
            with gr.Column():
                data_dropdown = gr.Dropdown(
                    choices=map(str, self.datasets), label="Dataset", interactive=True
                )
            with gr.Column():
                with gr.Row(equal_height=True):
                    fmh_radio = gr.Radio(
                        ["On", "Off"], value="Off", label="Fill missing hours"
                    )
                    ma_slider = gr.Slider(
                        minimum=1, maximum=24, label="Moving Average", step=1
                    )

        with gr.Tabs() as tabs:
            with gr.Tab("Review", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Options")
                        with gr.Row(equal_height=True):
                            vis_radio = gr.Radio(
                                ["On", "Off"],
                                value="Off",
                                label="Visualize Mode",
                                interactive=True,
                            )
                            group_dropdown = gr.Dropdown(
                                [
                                    ("None", "None"),
                                    ("Day", "D"),
                                    ("Week", "W"),
                                    ("Month", "ME"),
                                    ("Year", "YE"),
                                ],
                                label="Group by",
                                interactive=True,
                                scale=2,
                            )

                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### From")
                        with gr.Row():
                            fday = gr.Dropdown(
                                label="Day", min_width=50, interactive=True
                            )
                            fmonth = gr.Dropdown(
                                label="Month", min_width=50, interactive=True
                            )
                            fyear = gr.Dropdown(
                                label="Year", min_width=50, interactive=True
                            )

                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### To")
                        with gr.Row():
                            tday = gr.Dropdown(
                                label="Day", min_width=50, interactive=True
                            )
                            tmonth = gr.Dropdown(
                                label="Month", min_width=50, interactive=True
                            )
                            tyear = gr.Dropdown(
                                label="Year", min_width=50, interactive=True
                            )

                rev_df = gr.Dataframe(
                    max_height=600, show_fullscreen_button=True, show_search="search"
                )

                with gr.Row(visible=False) as vis:
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### Configuration")
                        features_dropdown = gr.Dropdown(
                            label="Features", multiselect=True, interactive=True
                        )

                    with gr.Column(scale=4):
                        vis_plot = gr.Plot(show_label=False)

                vis_radio.select(
                    self._select_vis,
                    [rev_df, vis_radio],
                    [rev_df, vis, vis_plot, features_dropdown],
                    scroll_to_output=True,
                )
                features_dropdown.select(
                    self._vis_plot,
                    [rev_df, vis_radio, features_dropdown],
                    vis_plot,
                    scroll_to_output=True,
                )
                rev_df.change(
                    self._vis_plot,
                    [rev_df, vis_radio, features_dropdown],
                    vis_plot,
                    scroll_to_output=True,
                )

                for key, values in {
                    "from": [fyear, fmonth, fday],
                    "to": [tyear, tmonth, tday],
                }.items():
                    for value in values:
                        value.select(
                            fn=partial(self._select_time, indicator=key),
                            inputs=values,
                            outputs=[values[-1], rev_df],
                            scroll_to_output=True,
                        )

                group_dropdown.select(
                    self._select_group, group_dropdown, rev_df, scroll_to_output=True
                )
                ma_slider.release(
                    lambda: gr.update(selected=0), None, tabs, scroll_to_output=True
                )
                fmh_radio.select(
                    lambda: gr.update(selected=0), None, tabs, scroll_to_output=True
                )
                data_dropdown.select(
                    lambda: gr.update(selected=0), None, tabs, scroll_to_output=True
                )
                ma_slider.release(
                    self._moving_average, ma_slider, rev_df, scroll_to_output=True
                )
                fmh_radio.select(
                    self._fill_missing_hours, fmh_radio, rev_df, scroll_to_output=True
                )
                data_dropdown.select(
                    self._select_data,
                    data_dropdown,
                    [rev_df, fyear, fmonth, fday, tyear, tmonth, tday],
                )
                self.parent.load(
                    self._select_data,
                    data_dropdown,
                    [rev_df, fyear, fmonth, fday, tyear, tmonth, tday],
                )

            with gr.Tab("Statistics", id=1) as stat_tab:
                gr.Markdown("## Review Table")
                stat_df = gr.Dataframe()

                gr.Markdown("---")
                gr.Markdown("## Descriptive Statistics")
                gr.Markdown(
                    "A visual representation of core statistical measures, offering insights into data distribution, variability, and overall trends. "
                )
                heatmap_plot = gr.Plot(show_label=False)

                gr.Markdown("---")
                gr.Markdown("## Distribution Percentiles")
                gr.Markdown(
                    "Illustrate the spread and distribution of each feature, highlighting key percentiles, quartile ranges, and potential outliers."
                )
                box_plot = gr.Plot(show_label=False)

                gr.Markdown("---")
                gr.Markdown("## Mean Value")
                gr.Markdown("A comparison of the average values of different features.")
                bar_plot = gr.Plot(show_label=False)

                stat_df.change(self._barplot, stat_df, bar_plot)
                stat_df.change(self._boxplot, stat_df, box_plot)
                stat_df.change(self._heatmap, stat_df, heatmap_plot)
                stat_tab.select(
                    lambda: self._summary(self.current["processed"]), None, stat_df
                )

            with gr.Tab("Correlation", id=2) as corr_tab:
                gr.Markdown("## Review Table")
                corr_df = gr.Dataframe()

                gr.Markdown("---")
                gr.Markdown("## Correlation Heatmap")
                gr.Markdown(
                    "A visual representation of feature correlations, highlighting the strength and direction of relationships among variables."
                )
                heatmap_plot = gr.Plot(show_label=False)

                gr.Markdown("---")
                gr.Markdown("## Pairwise Relationships")
                gr.Markdown(
                    "Scatter plots showing relationships between feature pairs, useful for detecting linear and non-linear trends."
                )

                with gr.Row():
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### Configuration")
                        x_dropdown = gr.Dropdown(
                            label="X-Axis (Horizontal)", interactive=True
                        )
                        y_dropdown = gr.Dropdown(
                            label="Y-Axis (Vertical)", interactive=True
                        )

                    with gr.Column(scale=4):
                        pairwise_plot = gr.Plot(show_label=False)

                x_dropdown.select(
                    self._corr_pairwise,
                    [x_dropdown, y_dropdown],
                    pairwise_plot,
                    scroll_to_output=True,
                )
                y_dropdown.select(
                    self._corr_pairwise,
                    [x_dropdown, y_dropdown],
                    pairwise_plot,
                    scroll_to_output=True,
                )

                corr_df.change(self._corr_heatmap, corr_df, heatmap_plot)

                corr_tab.select(
                    self._select_corr,
                    None,
                    [corr_df, x_dropdown, y_dropdown, pairwise_plot],
                )

        tabs.change(lambda: plt.close())
