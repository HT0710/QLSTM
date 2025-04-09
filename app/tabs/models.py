import re
from pathlib import Path

import gradio as gr
import numpy as np
import rootutils
import seaborn as sns
from matplotlib import pyplot as plt

rootutils.autosetup(".gitignore")

from qlstm.models.cLSTM import LSTM as cLSTM
from qlstm.models.cQLSTMf import QLSTM as cQLSTMf
from qlstm.models.LSTM import LSTM
from qlstm.modules.utils import yaml_handler


class ModelsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.checkpoint_path = Path("lightning_logs")
        self.datasets = sorted([i.name for i in self.data_path.glob("*.csv")])
        self.models = {
            "LSTM": {"init": LSTM(9, 128), "version": "base"},
            "cLSTM": {"init": cLSTM(9, 128), "version": "version_0"},
            "cQLSTMf": {"init": cQLSTMf(9, 128, 2), "version": "version_9"},
        }
        self.current = {}

    def _update_plot(self):
        names = []
        results = []

        for name, model in self.models.items():
            names.append(name)

            version_path = self.checkpoint_path / name / model["version"]
            hparams = yaml_handler(str(version_path / "hparams.yaml"))
            trained_data_name = hparams["data"]["data_path"].split("/")[-1]

            if trained_data_name != self.current["data"]:
                results.append(0)
                continue

            with open(str(version_path / "info.txt")) as f:
                results.append(
                    float(re.search(r"- Val loss:\s*([0-9.]+)", f.read()).group(1))
                )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        names, results = zip(*sorted(zip(names, results), reverse=True))

        sns.barplot(x=np.array(names), y=np.array(results), ax=ax, hue=np.array(names))

        fig.subplots_adjust(top=0.95, bottom=0.15)

        ax.set_xlabel("Model", fontsize=14, fontweight="bold", labelpad=14)
        ax.set_ylabel("Validation Loss", fontsize=14, fontweight="bold", labelpad=14)

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
        self.parent.load(self._select_data, data_dd, data_plt)
