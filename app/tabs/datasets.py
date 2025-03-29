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
        self.datasets = sorted(
            [i.name for i in self.data_path.glob("*.csv") if not i.match("*.x.csv")]
        )
        self.current = {
            "data": None,
            "processed": None,
            "ma_size": 1,
            "group": "None",
            "from": None,
            "to": None,
        }

    def _show_data(self, df):
        if self.current["ma_size"] > 1:
            df = df.rolling(window=self.current["ma_size"], min_periods=1).mean()

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

    def _change_data(self, data_name):
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
        self.current["summary"] = self._summary(df)

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

    def _change_group(self, group):
        self.current["group"] = group

        return self._show_data(self.current["processed"])

    def _change_time(self, y, m, d, indicator):
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
        self.current["ma_size"] = size

        return self._show_data(self.current["processed"])

    def _summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.describe().T.reset_index()
        summary = summary.rename(columns={"index": "Statistic"})

        return summary

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

        fig.subplots_adjust(top=0.95, bottom=0.15)

        ax.set_title("Mean Value", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Feature", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Value", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _select_vis(self, df, mode):
        plot = x_axis = y_axis = None

        if mode == "On":
            x_axis = gr.update(choices=["Datetime"], value="Datetime")
            y_axis = gr.update(
                choices=[c for c in df.columns if c != "Datetime"],
                value=[df.columns[-1]],
            )

            plot = self._vis_plot(df, mode, x_axis["value"], y_axis["value"])

        return (
            gr.update(visible=mode == "Off"),
            gr.update(visible=mode == "On"),
            plot,
            x_axis,
            y_axis,
        )

    def _vis_plot(self, df, mode, x, y):
        fig = None

        if mode == "On" and x and y:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

            for feature in y:
                sns.lineplot(x=pd.to_datetime(df[x]), y=df[feature], ax=ax)

            fig.subplots_adjust(top=0.95, bottom=0.15)

            ax.set_xlabel(x, fontsize=14, fontweight="bold", labelpad=14)
            ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

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
                            index_radio = gr.Radio(
                                ["On", "Off"], value="On", label="Show index"
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
                                scale=1,
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

                df = gr.Dataframe(
                    max_height=600,
                    show_row_numbers=True,
                    show_fullscreen_button=True,
                    show_search="search",
                )

                with gr.Row(visible=False) as vis:
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("### Configuration")
                        x_dropdown = gr.Dropdown(
                            label="X-Axis (Horizontal)", interactive=True
                        )
                        y_dropdown = gr.Dropdown(
                            label="Y-Axis (Vertical)",
                            multiselect=True,
                            interactive=True,
                        )

                    with gr.Column(scale=4):
                        vis_plot = gr.Plot(show_label=False)

                vis_radio.select(
                    self._select_vis,
                    [df, vis_radio],
                    [df, vis, vis_plot, x_dropdown, y_dropdown],
                    scroll_to_output=True,
                )

                x_dropdown.select(
                    self._vis_plot,
                    [df, vis_radio, x_dropdown, y_dropdown],
                    vis_plot,
                    scroll_to_output=True,
                )
                y_dropdown.select(
                    self._vis_plot,
                    [df, vis_radio, x_dropdown, y_dropdown],
                    vis_plot,
                    scroll_to_output=True,
                )

                df.change(
                    self._vis_plot,
                    [df, vis_radio, x_dropdown, y_dropdown],
                    vis_plot,
                    scroll_to_output=True,
                )

                index_radio.select(
                    lambda x: gr.update(show_row_numbers=x == "On"), index_radio, df
                )

                for field in [fday, fmonth, fyear]:
                    field.select(
                        fn=partial(self._change_time, indicator="from"),
                        inputs=[fyear, fmonth, fday],
                        outputs=[fday, df],
                        scroll_to_output=True,
                    )

                for field in [tday, tmonth, tyear]:
                    field.select(
                        fn=partial(self._change_time, indicator="to"),
                        inputs=[tyear, tmonth, tday],
                        outputs=[tday, df],
                        scroll_to_output=True,
                    )

                group_dropdown.select(
                    self._change_group, group_dropdown, df, scroll_to_output=True
                )
                fmh_radio.select(
                    self._fill_missing_hours, fmh_radio, df, scroll_to_output=True
                )
                ma_slider.release(
                    self._moving_average, ma_slider, df, scroll_to_output=True
                )
                data_dropdown.select(
                    lambda: gr.update(selected=0), None, tabs, scroll_to_output=True
                )
                data_dropdown.change(
                    self._change_data,
                    data_dropdown,
                    [df, fyear, fmonth, fday, tyear, tmonth, tday],
                )
                self.parent.load(
                    self._change_data,
                    data_dropdown,
                    [df, fyear, fmonth, fday, tyear, tmonth, tday],
                )

            with gr.Tab("Statistic", id=1) as stat_tab:
                gr.Markdown("## Review Table")
                df = gr.Dataframe()

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

                stat_tab.select(lambda: self.current["summary"], None, df)
                stat_tab.select(self._heatmap, None, heatmap_plot)
                stat_tab.select(self._boxplot, None, box_plot)
                stat_tab.select(self._barplot, None, bar_plot)

        tabs.change(lambda: plt.close())
