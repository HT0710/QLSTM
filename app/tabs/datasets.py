from calendar import monthrange
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.data_path = Path("./qlstm/data")
        self.datasets = sorted([i for i in self.data_path.glob("*.csv")])
        self.current = {
            "choice": None,
            "data": None,
            "processed": None,
            "group": "None",
            "method": "Sum",
            "from": None,
            "to": None,
        }

    def _upload_data(self, files):
        for file in files:
            if not str(file.name).endswith((".csv", ".xlsx")):
                gr.Error('Only ".csv" or ".xlsx" file are accepted.')

            self.datasets.append(Path(file))

    def _show_data(self, df):
        if self.current["group"] != "None":
            df = df.resample(self.current["group"])

            df = df.sum() if self.current["method"] == "Sum" else df.mean()

            match self.current["group"]:
                case "D" | "W":
                    df.index = df.index.to_period("D")
                case "ME":
                    df.index = df.index.to_period("M")
                case "YE":
                    df.index = df.index.to_period("Y")

        df = df.loc[self.current["from"] : self.current["to"]]

        return gr.update(
            value=df.round(2).reset_index(),
            label=f"Number of Rows: {len(df)}",
            column_widths=[
                f"{1 / (len(df.columns) + 1) * 100}%" for _ in range(len(df.columns))
            ],
        )

    def _select_data(self, data_path: str):
        df = pd.read_csv(data_path)

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Change all columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        if "datetime" not in df.columns:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

                if "hour" in df.columns:
                    # Combine "date" and "hour" to "datetime"
                    df["hour"] = pd.to_timedelta(df["hour"], unit="h")
                    df["date"] = df["date"] + df["hour"]
                    df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

                # Rename "date" to "Datetime"
                df.rename(columns={"date": "datetime"}, inplace=True)

            else:
                gr.Error('Data must contain "date" or "datetime" column.')

        self.current["choice"] = data_path

        # Sort and set "Datetime" as index
        df = df.sort_values(by="datetime")
        df = df.set_index("datetime")

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

    def _select_method(self, method):
        self.current["method"] = method

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
                        if col == "datetime":
                            row[col] = date + pd.DateOffset(hours=h)
                        else:
                            row[col] = np.nan
                    new_rows.append(row)
                filled_rows.extend(new_rows)

            for _, row in df.iterrows():
                curr_date = row["datetime"].date()
                curr_hour = row["datetime"].hour

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

            self.current["processed"] = df.set_index("datetime")

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
        plot = y_axis = gr.update(visible=False)

        if mode == "On":
            y_axis = gr.update(
                choices=[c for c in df.columns if c != "datetime"],
                value=df.columns[-1],
                visible=True,
            )

            plot = self._vis_plot(df, mode, y_axis["value"])

        return (
            gr.update(visible=mode == "Off"),
            plot,
            y_axis,
        )

    def _vis_plot(self, df, mode, y):
        if mode == "Off":
            return gr.update(visible=False)

        df["datetime"] = pd.to_datetime(df["datetime"])

        return gr.LinePlot(
            df,
            x="datetime",
            y=y,
            x_title=f"Time ({self.current['group']})",
            y_title="Energy (kWh)",
            visible=True,
        )

    def _summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.describe().T.round(2).reset_index()
        summary = summary.rename(columns={"index": "statistic"})

        return summary

    def _heatmap(self, df):
        df = df[["statistic", "mean", "std", "min", "max"]]
        df = df.set_index("statistic")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)

        fig.subplots_adjust(left=0.2, right=1, top=0.85, bottom=0.15)

        ax.set_title("Descriptive Statistics", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Statistics", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Features", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _boxplot(self, df):
        df = df[["statistic", "25%", "50%", "75%"]]

        df_melted = df.melt(
            id_vars=["statistic"], var_name="percentile", value_name="value"
        )

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.boxplot(
            df_melted, x="statistic", y="value", ax=ax, hue="statistic", palette="Set2"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Distribution Percentiles", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Features", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _barplot(self, df):
        df = df[["statistic", "mean"]]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        sns.barplot(
            df, x="statistic", y="mean", ax=ax, hue="statistic", palette="viridis"
        )

        fig.subplots_adjust(top=0.85, bottom=0.15)

        ax.set_title("Mean Value", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Features", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Values", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _select_corr(self):
        df = self.current["processed"]

        corr = df.corr().round(2).reset_index()
        corr = corr.rename(columns={"index": "features"})

        features = list(df.columns)

        return (
            corr,
            gr.update(choices=features, value=features[0]),
            gr.update(choices=features, value=features[-1]),
            self._corr_pairwise(features[0], features[-1]),
        )

    def _corr_heatmap(self, df):
        df = df.set_index("features")

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
        with gr.Row():
            with gr.Column():
                data_dropdown = gr.Dropdown(
                    choices=[(i.name, str(i)) for i in self.datasets],
                    label="Select data",
                    interactive=True,
                )
                upload_bt = gr.UploadButton(
                    label="Upload data",
                    file_count="multiple",
                    file_types=[".csv", ".xlsx"],
                )

            with gr.Column():
                with gr.Row(equal_height=True):
                    fmh_radio = gr.Radio(
                        ["On", "Off"],
                        value="Off",
                        label="Fill missing hours",
                        min_width=0,
                    )
                    ma_slider = gr.Slider(
                        minimum=1, maximum=24, label="Moving Average", step=1
                    )

            upload_bt.upload(self._upload_data, upload_bt, upload_bt).success(
                lambda: [
                    gr.Success("New data added successfully!"),
                    gr.update(
                        choices=sorted([(i.name, str(i)) for i in set(self.datasets)]),
                        value=self.current["choice"],
                    ),
                ],
                None,
                [data_dropdown, data_dropdown],
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
                            features_dd = gr.Dropdown(
                                label="Features",
                                interactive=True,
                                visible=False,
                                min_width=60,
                            )
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
                                min_width=60,
                            )
                            method_dd = gr.Dropdown(
                                ["Sum", "Mean"],
                                label="Method",
                                interactive=True,
                                min_width=60,
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
                    max_height=600,
                    show_fullscreen_button=True,
                    show_search="search",
                    interactive=False,
                )

                vis_plot = gr.LinePlot(
                    show_label=False, interactive=False, visible=False
                )

                ### EVENTS ###

                # Change group and method
                group_dd.select(self._select_group, group_dd, rev_df)
                method_dd.select(self._select_method, method_dd, rev_df)

                # Change ma and fmh
                ma_slider.release(self._moving_average, ma_slider, rev_df)
                fmh_radio.select(self._fill_missing_hours, fmh_radio, rev_df)

                # Select vis
                vis_radio.select(
                    self._select_vis,
                    [rev_df, vis_radio],
                    [rev_df, vis_plot, features_dd],
                )

                # Change plot on change df
                gr.on(
                    triggers=[features_dd.select, rev_df.change],
                    fn=self._vis_plot,
                    inputs=[rev_df, vis_radio, features_dd],
                    outputs=vis_plot,
                )

                # Change time
                for key, values in {
                    "from": [fyear, fmonth, fday],
                    "to": [tyear, tmonth, tday],
                }.items():
                    for value in values:
                        value.select(
                            fn=partial(self._select_time, indicator=key),
                            inputs=values,
                            outputs=[values[-1], rev_df],
                        )

                # Back to "Review" tab
                gr.on(
                    [ma_slider.release, fmh_radio.select, data_dropdown.select],
                    fn=lambda: gr.update(selected=0),
                    inputs=None,
                    outputs=tabs,
                )

                # Change data
                gr.on(
                    [data_dropdown.select, self.parent.select],
                    fn=self._select_data,
                    inputs=data_dropdown,
                    outputs=[rev_df, fyear, fmonth, fday, tyear, tmonth, tday],
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

                gr.on(
                    [x_dropdown.select, y_dropdown.select],
                    fn=self._corr_pairwise,
                    inputs=[x_dropdown, y_dropdown],
                    outputs=pairwise_plot,
                )

                corr_df.change(self._corr_heatmap, corr_df, heatmap_plot)

                corr_tab.select(
                    self._select_corr,
                    None,
                    [corr_df, x_dropdown, y_dropdown, pairwise_plot],
                )

        tabs.change(lambda: plt.close())
