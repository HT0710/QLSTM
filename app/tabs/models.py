import re
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
from matplotlib import pyplot as plt

rootutils.autosetup(".gitignore")

from common.models import MODELS

from qlstm.modules.utils import yaml_handler


class ModelsTab:
    def __init__(self, parent):
        self.parent = parent
        self.data_path = Path("./qlstm/data")
        self.datasets = sorted([i.name for i in self.data_path.glob("*.csv")])
        self.models = MODELS
        self.current = {}

    def _update_plot(self):
        names = []
        results = {"train": [], "test": []}

        for name, model in self.models.items():
            names.append(name)

            version_path = Path(model["checkpoint"]).parents[1]
            hparams = yaml_handler(str(version_path / "hparams.yaml"))
            trained_data_name = hparams["data"]["data_path"].split("/")[-1]

            if trained_data_name != self.current["data"]:
                results["train"].append(0)
                results["test"].append(0)
                continue

            with open(str(version_path / "info.txt")) as f:
                info = f.read()
                results["train"].append(
                    float(re.search(r"- Train loss:\s*([0-9.]+)", info).group(1))
                )
                results["test"].append(
                    float(re.search(r"- Val loss:\s*([0-9.]+)", info).group(1))
                )

        combined = list(zip(names, results["train"], results["test"]))

        combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)

        names, train, test = zip(*combined_sorted)

        df = pd.DataFrame(
            {
                "name": np.repeat(names, 2),
                "Label": ["Train", "Test"] * len(names),
                "value": np.concatenate([train, test]),
            }
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        sns.barplot(x="name", y="value", hue="Label", data=df, ax=ax)

        fig.subplots_adjust(top=0.95, bottom=0.15)

        ax.set_xlabel("Model", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Loss", fontsize=14, fontweight="bold", labelpad=14)

        return fig

    def _select_data(self, data_name):
        self.current["data"] = data_name
        fig = self._update_plot()

        return fig

    def __call__(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Options")
                data_dd = gr.Dropdown(
                    choices=self.datasets, label="Dataset", interactive=True
                )

            with gr.Column(scale=4):
                gr.Markdown("### Result")
                data_plt = gr.Plot(show_label=False)

        data_dd.select(self._select_data, data_dd, data_plt)
        self.parent.select(self._select_data, data_dd, data_plt)
