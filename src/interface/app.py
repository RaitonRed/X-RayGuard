import gradio as gr
from src.interface import functions

with gr.Blocks() as Interface:
    gr.Markdown("# **X-RayGuard**")
    with gr.Tabs():
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    image = gr.Image(label="Your X-Ray Image", type="filepath")

                with gr.Column(scale=1, min_width=350):
                    result = gr.Textbox(
                        label="Analysis Report",
                        interactive=False,
                        lines=8,
                        max_lines=12,
                        placeholder="Results will appear here..."
                    )
                    submit_btn = gr.Button("Analyze Image", variant="primary")
            submit_btn.click(
                functions.predict,
                inputs=[image],
                outputs=result
            )
        with gr.Tab("Grad-Cam"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    image = gr.Image(label="You X-Ray Image", type="filepath")

                with gr.Column(scale=1, min_width=350):
                    result = gr.Image(label="Grad-Cam Output")
                    submit_btn = gr.Button("Analyze Image", variant="primary")
                    submit_btn.click(
                        functions.grad_cam,
                        inputs=[image],
                        outputs=result
                    )
    gr.Markdown("___\nMade by **Raiton** with ❤")


def launch():
    # Launch the interface
    try:
        Interface.queue(max_size=20).launch(
            server_port=7860,
            show_error=True
        )
    except TypeError:
        Interface.queue().launch(
            server_port=7860,
            show_error=True
        )


if __name__ == "__main__":
    launch()
