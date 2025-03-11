import gradio as gr
from rich import traceback

traceback.install()

from tabs.datasets import DatasetsTab
from tabs.demo import DemoTab
from tabs.models import ModelsTab


def main():
    css = """
        h1 {
            text-align: center;
            display:block;
        }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# PV Power Forcasting App")

        with gr.Tabs(selected=1):
            with gr.Tab(label="Live", id=0):
                gr.Markdown("# Comming Soon")

            with gr.Tab(label="Demo", id=1):
                DemoTab(demo)()

            with gr.Tab(label="Datasets", id=2):
                DatasetsTab(demo)()

            with gr.Tab(label="Models", id=3):
                ModelsTab(demo)()

    demo.launch()


if __name__ == "__main__":
    main()
