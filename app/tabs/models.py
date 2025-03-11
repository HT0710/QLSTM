from pathlib import Path

import gradio as gr
from rich import traceback

traceback.install()


class ModelsTab:
    def __init__(self, parent):
        self.parent = parent
        self.root = Path("./qlstm")
        self.model_path = self.root / "models"
        self.models = self.model_path.glob("*.py")

    def _read_file(self, model_name):
        try:
            with open(
                str(self.model_path / f"{model_name}.py"), "r", encoding="utf-8"
            ) as f:
                content = f.read()
            return content
        except Exception as e:
            gr.Error(f"Error reading file: {str(e)}")
            return None

    def __call__(self):
        with gr.Tabs(selected=1):
            with gr.Tab("Overview", id=0):
                gr.Markdown("# Comming Soon")

            with gr.Tab("Detail", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=[
                                x.stem
                                for x in self.models
                                if x.stem not in ["__init__", "utils"]
                            ],
                            label="Model",
                            value="LSTM",
                            interactive=True,
                        )
                        display_dropdown = gr.Dropdown(
                            choices=["Code"],
                            label="Display",
                            # interactive=True,
                        )

                    with gr.Column(scale=4):
                        code_display = gr.Code(
                            language="python",
                            label="",
                            interactive=False,
                        )

                    model_dropdown.change(
                        self._read_file, [model_dropdown], code_display, queue=False
                    )
        self.parent.load(self._read_file, [model_dropdown], code_display, queue=False)
