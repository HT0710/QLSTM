import re
from collections import defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd
import rootutils

rootutils.autosetup(".gitignore")

from qlstm.modules.utils import yaml_handler


class ModelsTab:
    def __init__(self, parent):
        self.parent = parent

    def _read_folder(self, path: Path) -> dict:
        tree = {}
        for entry in path.iterdir():
            if entry.is_dir():
                tree[entry.name] = self._read_folder(entry)
            else:
                tree.setdefault("__files__", []).append(entry.name)
        return tree

    def _select_data(self, data_name):
        df = self.datasets[data_name]
        return gr.update(
            value=df,
            label=f"Number of Rows: {len(df)}",
            column_widths=[
                f"{1 / (len(df.columns) + 1) * 100}%" for _ in range(len(df.columns))
            ],
        )

    def _init(self):
        datasets = defaultdict(list)

        logs_folder = Path("lightning_logs")

        for model, versions in self._read_folder(logs_folder).items():
            model_folder = logs_folder / model
            for version in versions:
                version_folder = model_folder / version
                hparams = yaml_handler(str(version_folder / "hparams.yaml"))
                data_name = hparams["data"]["data_path"].split("/")[-1]

                with open(str(version_folder / "info.txt")) as f:
                    info = f.read()
                    train = float(
                        re.search(r"- Train loss:\s*([0-9.]+)", info).group(1)
                    )

                    test = float(re.search(r"- Val loss:\s*([0-9.]+)", info).group(1))

                datasets[data_name].append(
                    {
                        "Model": model,
                        "Version": version,
                        "Train loss": train,
                        "Test loss": test,
                    }
                )

        for data_name, items in datasets.items():
            datasets[data_name] = pd.DataFrame(items)

        self.datasets = datasets
        default = list(datasets.keys())[0]

        return (
            gr.update(choices=datasets.keys(), value=default),
            self._select_data(data_name=default),
        )

    def __call__(self):
        with gr.Row():
            with gr.Column(scale=1, min_width=160):
                gr.Markdown("### Options")
                data_radio = gr.Radio(label="Dataset", interactive=True)

            with gr.Column(scale=5):
                df = gr.Dataframe(
                    show_fullscreen_button=True,
                    show_search="search",
                    interactive=False,
                )

        data_radio.select(self._select_data, data_radio, df)

        self.parent.select(self._init, None, [data_radio, df])
