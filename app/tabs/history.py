from calendar import monthrange
from datetime import datetime
from functools import partial

import gradio as gr
import pandas as pd


class HistoryTab:
    def __init__(self, parent):
        self.parent = parent
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
            value=df.round(2).reset_index().sort_values(by="datetime", ascending=False),
            label=f"Number of Rows: {len(df)}",
            column_widths=[
                f"{1 / (len(df.columns) + 1) * 100}%" for _ in range(len(df.columns))
            ],
        )

    def _select_time(self, y, m, d, indicator):
        _, new_num_days = monthrange(y, m)
        d = min(d, new_num_days)

        self.current[indicator] = datetime(y, m, d)

        return (
            gr.update(choices=range(1, new_num_days + 1), value=d),
            self._show_data(self.current["data"]),
        )

    def _select_group(self, group):
        self.current["group"] = group

        return self._show_data(self.current["processed"])

    def _select_vis(self, df, mode):
        plot = y_axis = None

        if mode == "On":
            y_axis = gr.update(
                choices=[c for c in df.columns if c != "datetime"],
                value=df.columns[-1],
                visible=True,
            )

            plot = self._vis_plot(df, mode, y_axis["value"])

        else:
            y_axis = gr.update(visible=False)

        return (
            gr.update(visible=mode == "Off"),
            gr.update(visible=mode == "On"),
            plot,
            y_axis,
            gr.update(scale=1 if mode == "On" else 2),
        )

    def _vis_plot(self, df, mode, y):
        if not (mode == "On" or y):
            return None

        df["datetime"] = pd.to_datetime(df["datetime"])

        return gr.LinePlot(
            df,
            x="datetime",
            y=y,
            x_title=f"Time ({self.current['group']})",
            y_title="Energy (kWh)",
        )

    def _update(self):
        df = pd.read_csv("app/history.csv")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

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

    def __call__(self):
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
                        label="Features", interactive=True, visible=False
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
                        scale=2,
                    )

            with gr.Column(scale=1, min_width=0):
                gr.Markdown("### From")
                with gr.Row():
                    fday = gr.Dropdown(label="Day", min_width=50, interactive=True)
                    fmonth = gr.Dropdown(label="Month", min_width=50, interactive=True)
                    fyear = gr.Dropdown(label="Year", min_width=50, interactive=True)

            with gr.Column(scale=1, min_width=0):
                gr.Markdown("### To")
                with gr.Row():
                    tday = gr.Dropdown(label="Day", min_width=50, interactive=True)
                    tmonth = gr.Dropdown(label="Month", min_width=50, interactive=True)
                    tyear = gr.Dropdown(label="Year", min_width=50, interactive=True)

        df = gr.Dataframe(
            max_height=600,
            show_fullscreen_button=True,
            show_search="search",
            interactive=False,
        )

        with gr.Row(visible=False) as vis:
            plot = gr.LinePlot(show_label=False)

        vis_radio.select(
            self._select_vis,
            [df, vis_radio],
            [df, vis, plot, features_dd, group_dd],
            scroll_to_output=True,
        )

        gr.on(
            triggers=[features_dd.select, df.change],
            fn=self._vis_plot,
            inputs=[df, vis_radio, features_dd],
            outputs=plot,
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
                    outputs=[values[-1], df],
                    scroll_to_output=True,
                )

        group_dd.select(self._select_group, group_dd, df, scroll_to_output=True)

        self.parent.select(
            self._update,
            outputs=[df, fyear, fmonth, fday, tyear, tmonth, tday],
        )
