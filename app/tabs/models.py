from collections import defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd
import rootutils
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

                result = {
                    "Model": model,
                    "Version": version,
                }

                ea = EventAccumulator(str(list(version_folder.glob("events.*"))[0]))
                ea.Reload()

                metrics = sorted(
                    [
                        str(i)
                        for i in ea.Tags()["scalars"]
                        if str(i).startswith(("train", "val"))
                    ]
                )

                for metric in metrics:
                    event = ea.Scalars(metric)[-1]
                    result[metric] = f"{event.value:.4f}"

                datasets[data_name].append(result)

        for data_name, items in datasets.items():
            datasets[data_name] = pd.DataFrame(items)

        self.datasets = datasets
        default = list(datasets.keys())[0]

        return (
            gr.update(choices=datasets.keys(), value=default),
            self._select_data(data_name=default),
        )

    def __call__(self):
        data_radio = gr.Radio(label="Dataset", interactive=True)
        df = gr.Dataframe(
            show_fullscreen_button=True,
            show_search="search",
            interactive=False,
            max_height=600,
        )

        data_radio.select(self._select_data, data_radio, df)

        self.parent.select(self._init, None, [data_radio, df])
