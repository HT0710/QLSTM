import gradio as gr

_SCRIPTS = {
    "Introduction": [
        "PV Power Forcasting App is a research-driven demo platform showcasing a novel approach to solar photovoltaic (PV) power forecasting using advanced AI-Quantum hybrid models. This app is the interactive companion to an ongoing project exploring how Modified LSTMs enhanced with Variational Quantum Circuits (VQCs) can push the boundaries of long-term sequence modeling in renewable energy forecasting."
    ],
    "What Makes This Research Different?": [
        "### The Model: Modified QLSTM (M-QLSTM)",
        "Unlike standard deep learning or classical quantum-enhanced models, this research proposes a novel model that addresses long-term dependency degradation, parameter inefficiency, and quantum circuit complexity. Our contributions include:",
        "### 1. Persistent-Memory LSTM:",
        "- Enhances memory retention by feeding the previous cell state directly into the forget gate, improving long-range learning and reducing vanishing gradients.",
        "- Allows finer memory control over what should be retained or forgotten during temporal modeling.",
        "### 2. Quantum Integration with Fewer Circuits:",
        "- Uses a single Variational Quantum Circuit (VQC) shared across all LSTM gates.",
        "- Reduces hardware noise, parameter complexity, and computation time compared to typical QLSTM designs with 4-6 VQCs.",
        "### 3. Exponential Weighted Features (EWF):",
        "- A lightweight layer that helps accelerate convergence during training by emphasizing temporally relevant features.",
        "- Bridges quantum inference with classical efficiency for faster and more accurate PV forecasting.",
    ],
    "System Architecture": [
        "### Input Pipeline:",
        "- Multivariate time-series input includes historical PV output, numerical weather predictions (NWP), temperature, wind, and radiation metrics.",
        "- Real-time compatible with APIs from weather services or on-site SCADA sensors.",
        "### Model Stack:",
        "- Preprocessing → Exponential Weighted Feature Layer → Quantum Gate Embedding (RY, RZ, CNOT, Rotation gates) → Custom LSTM Cells → Output Layer",
        "### Quantum Simulation:",
        "- Executed on 2 and 4 qubit configurations using PennyLane with backend support for quantum simulators and future compatibility with quantum hardware.",
    ],
    "Performance Highlights": [
        "**Result**: Faster training, lower test error, and reduced quantum resource usage — even under limited-qubit setups.",
    ],
    "Learn More": [
        "- [Research Paper](...)",
        "- [Github](...)",
        "- [Contact](...)",
    ],
    "Team Members": [
        "- Phan Ha Vu",
        "- Pham Tan Hung",
    ],
}


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self.scripts = _SCRIPTS

    def __call__(self):
        gr.Image(
            "app/headline.png",
            container=False,
            show_label=False,
            interactive=False,
            show_download_button=False,
            show_fullscreen_button=False,
        )

        gr.Markdown("## Introduction")
        gr.Markdown("\n".join(self.scripts["Introduction"]))
        gr.Markdown("---\n")
        gr.Markdown()

        gr.Markdown("## What Makes This Research Different?")
        gr.Markdown("\n".join(self.scripts["What Makes This Research Different?"]))
        gr.Markdown("---\n")
        gr.Markdown()

        gr.Markdown("## System Architecture")
        gr.Markdown("\n".join(self.scripts["System Architecture"]))
        gr.Markdown("---\n")
        gr.Markdown()

        gr.Markdown("## Performance Highlights")
        gr.Markdown("\n".join(self.scripts["Performance Highlights"]))
        gr.Markdown("---\n")
        gr.Markdown()

        gr.Markdown("## Learn More")
        gr.Markdown("\n".join(self.scripts["Learn More"]))
        gr.Markdown("---\n")
        gr.Markdown()

        gr.Markdown("## Team Members")
        gr.Markdown("\n".join(self.scripts["Team Members"]))
        gr.Markdown("---\n")
        gr.Markdown()
