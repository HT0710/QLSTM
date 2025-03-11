from pathlib import Path

import gradio as gr
import pandas as pd


class DatasetsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.data_path = self.root / "data"
        self.datasets = self.data_path.glob("*.csv")

    def _load_data(self, data_name):
        return pd.read_csv(str(self.data_path / data_name))

    def __call__(self):
        with gr.Tabs(selected=1):
            with gr.Tab("Overview", id=0):
                gr.Markdown("# Comming Soon")

            with gr.Tab("Detail", id=1):
                with gr.Row():
                    data_dropdown = gr.Dropdown(
                        choices=[str(x.name) for x in self.datasets],
                        label="Dataset",
                        interactive=True,
                    )
                    display_dropdown = gr.Dropdown(
                        choices=["Review"],
                        label="Display",
                        # interactive=True,
                    )

                df = gr.Dataframe(max_height=800)

                data_dropdown.change(self._load_data, [data_dropdown], df, queue=False)

        self.parent.load(self._load_data, [data_dropdown], df, queue=False)
