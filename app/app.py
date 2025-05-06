import argparse

import gradio as gr
from rich import traceback

traceback.install()

from tabs.datasets import DatasetsTab
from tabs.demo import DemoTab
from tabs.estimate import EstimateTab
from tabs.history import HistoryTab
from tabs.home import HomeTab
from tabs.live import LiveTab
from tabs.models import ModelsTab
from tabs.system import SystemTab


def main(args):
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

        with gr.Tab("User"):
            with gr.Tab(label="Home", elem_classes="home-tab"):
                HomeTab(app)()

            with gr.Tab(label="Live") as live_tab:
                LiveTab(live_tab)()

            with gr.Tab(label="History") as history_tab:
                HistoryTab(history_tab)()

            with gr.Tab(label="Estimate") as estimate_tab:
                EstimateTab(estimate_tab)()

        with gr.Tab("Developer"):
            with gr.Tab(label="Home", elem_classes="home-tab"):
                HomeTab(app)()

            with gr.Tab(label="Demo") as demo_tab:
                DemoTab(demo_tab)()

            with gr.Tab(label="Datasets") as datasets_tab:
                DatasetsTab(datasets_tab)()

            with gr.Tab(label="Models") as models_tab:
                ModelsTab(models_tab)()

            with gr.Tab(label="System") as system_tab:
                SystemTab(system_tab)()

    app.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PV Power Forecasting App")
    parser.add_argument(
        "--share",
        "-s",
        action="store_true",
        help="Launch Gradio app with public share link",
    )
    args = parser.parse_args()

    main(args)
