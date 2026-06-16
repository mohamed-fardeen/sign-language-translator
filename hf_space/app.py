"""Hugging Face Space entry point.

Serves a Gradio UI that links to the static browser app served by the
Render-hosted FastAPI service. The Space itself is intentionally thin:
it provides a public landing page with a link to the full browser
demo, plus a small "About" panel describing the model.
"""
from __future__ import annotations

import os

import gradio as gr

API_BASE = os.environ.get("SIGNLANG_API_BASE", "https://signlang-api.onrender.com")
DEMO_URL = os.environ.get("SIGNLANG_DEMO_URL", f"{API_BASE.rstrip('/')}/web/")


def about() -> str:
    return (
        "# signlang\n\n"
        "ASL isolated-sign recognizer (500-gloss vocabulary, "
        "Transformer + CTC).\n\n"
        f"- Browser demo: {DEMO_URL}\n"
        f"- API: {API_BASE}\n"
        "- Model: per-stream MLPs -> Transformer encoder -> CTC head\n"
        "- Inference: ONNX Runtime Web (browser) or FastAPI (server)\n"
    )


with gr.Blocks(title="signlang", theme=gr.themes.Soft()) as demo:
    gr.Markdown(about)
    with gr.Row():
        open_btn = gr.Button("Open browser demo", link=DEMO_URL)
    gr.Markdown(
        "Tip: allow camera access in the demo. Click **Start**, then "
        "**Record** to capture a 2-3 s clip, then **Predict**."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
