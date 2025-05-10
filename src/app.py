import gradio as gr
from predict import LungDiseasePredictor

# Initialize predictor
predictor = LungDiseasePredictor()


def predict(image):
    prediction = predictor.predict(image_path=image)

    formatted_output = (
        f"🔍 Diagnosis Results:\n\n"
        f"🏷️ Predicted Class: {prediction['class']}\n"
        f"🎯 Confidence Level: {prediction['confidence'] * 100:.2f}%\n\n"
        "📊 Class Probabilities:\n"
    )

    for cls, prob in prediction['probabilities'].items():
        formatted_output += f"• {cls}: {prob * 100:.2f}%\n"

    return formatted_output


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
                predict,
                inputs=[image],
                outputs=result
            )

    gr.Markdown("___\nMade by **Raiton** with ❤")

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