from collections import defaultdict, deque
from datetime import datetime

import gradio as gr
import numpy as np
import pandas as pd
import psutil


class SystemTab:
    def __init__(self, parent):
        self.parent = parent
        self.boot_time = datetime.now()
        self.history = defaultdict(lambda: deque(maxlen=60))

    def _update(self):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()

        self.history["cpu"].append(cpu)
        self.history["ram"].append(ram.percent)

        uptime = datetime.now() - self.boot_time

        cpu_df = pd.DataFrame(
            {
                "time": range(len(self.history["cpu"]), 0, -1),
                "cpu": self.history["cpu"],
            }
        )

        ram_df = pd.DataFrame(
            {
                "time": range(len(self.history["ram"]), 0, -1),
                "ram": self.history["ram"],
            }
        )

        return (
            str(uptime).split(".")[0],
            f"{np.mean(list(self.history['cpu'])[-3:]):.1f} %",
            f"{np.mean(list(self.history['ram'])[-3:]):.1f} %",
            cpu_df,
            ram_df,
        )

    def __call__(self):
        gr.Markdown("## Current")
        with gr.Row():
            uptime = gr.Label(label="Uptime")
            cpu = gr.Label(label="CPU")
            ram = gr.Label(label="RAM")

        gr.Markdown("## CPU")
        cpu_graph = gr.BarPlot(
            x="time",
            y="cpu",
            x_title="Time (s)",
            y_title="Usage (%)",
            x_lim=[60, 0],
            y_lim=[0, 100],
        )

        gr.Markdown("## RAM")
        ram_graph = gr.BarPlot(
            x="time",
            y="ram",
            x_title="Time (s)",
            y_title="Usage (%)",
            x_lim=[60, 0],
            y_lim=[0, 100],
        )

        gr.Timer(1).tick(self._update, None, [uptime, cpu, ram, cpu_graph, ram_graph])
