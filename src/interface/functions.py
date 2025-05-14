import tensorflow as tf
import numpy as np
import os
from src.predict import LungDiseasePredictor
from src.grad_cam import GradCAM


# Initialize Predictor
predictor = LungDiseasePredictor()

model_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),  # Location of functions.py (src/interface/)
    '..', '..', 'models',       # Go up 2 levels to X-RayGuard/, then into models/
    'best_model.h5'
))

if not os.path.exists(model_dir):
    raise RuntimeError(f"Model not found at: {model_dir}")


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


def grad_cam(image, model=model_dir, target_layer='block_2_project_BN'):
    # Load model
    print(f"Loading model from {model}...")
    full_model = tf.keras.models.load_model(model, compile=False)

    # Build model with dummy input
    dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
    _ = full_model(dummy_input)  # Critical for Sequential models

    # Use MobileNetV2 submodel directly
    mobilenet_submodel = full_model.get_layer('mobilenetv2_0.35_96')
    _ = mobilenet_submodel(dummy_input)  # Build submodel

    # Initialize Grad-Cam
    grad = GradCAM(mobilenet_submodel, layer_name=target_layer)

    # Generate visualizations
    visualizations = grad.visualize(
        image,
        img_size=(96, 96)
    )

    return visualizations