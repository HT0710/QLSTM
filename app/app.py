import gradio as gr
from rich import traceback

traceback.install()

from tabs.datasets import DatasetsTab
from tabs.demo import DemoTab
from tabs.home import HomeTab
from tabs.live import LiveTab
from tabs.models import ModelsTab


def main():
    css = """
        h1 {
            text-align: center;
            display:block;
        }

        .home-tab {
            margin-right: 50px;
            margin-left: 50px;
            width: auto;
        }
    """

    with gr.Blocks(css=css) as app:
        gr.Markdown("# PV Power Forcasting App")

        with gr.Tab(label="Home", elem_classes="home-tab"):
            HomeTab(app)()

        with gr.Tab(label="Live") as live_tab:
            LiveTab(live_tab)()

        with gr.Tab(label="Demo") as demo_tab:
            DemoTab(demo_tab)()

        with gr.Tab(label="Datasets") as datasets_tab:
            DatasetsTab(datasets_tab)()

        with gr.Tab(label="Models") as models_tab:
            ModelsTab(models_tab)()

    app.launch()


if __name__ == "__main__":
    main()
