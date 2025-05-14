import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import src.options as options
from src.predict import LungDiseasePredictor


class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM wrapper.

        Args:
            model: Keras model (top-level or sub-model)
            layer_name: Name of target convolutional layer
        """
        self.model = model
        self.layer_name = layer_name
        self.predictor = LungDiseasePredictor()

        # Ensure model is built
        if not self.model.built:
            raise ValueError("Model must be built before initializing GradCAM.")

        # Auto-detect target layer if not provided
        if not self.layer_name:
            print("Auto-searching for last Conv2D layer...")
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    print(f"Auto-selected layer: {self.layer_name}")
                    break
            if not self.layer_name:
                raise ValueError("No Conv2D layer found in model!")

        # Verify target layer exists
        try:
            self.target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            raise ValueError(f"Layer '{self.layer_name}' not found in model.")

        # Create Grad-CAM model
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.target_layer.output, self.model.output]
        )

    def generate_heatmap(self, image, eps=1e-8):
        """
        Generate Grad-CAM heatmap.

        Args:
            image: Preprocessed input image (with batch dimension)
            eps: Small value to avoid division by zero

        Returns:
            heatmap, predicted_class_idx
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)

            if len(predictions.shape) > 2:
                predictions = tf.reduce_mean(predictions, axis=[1, 2])

            predicted_class_idx = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class_idx]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Failed to compute gradients. Check model architecture.")

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + eps)

        return heatmap.numpy(), predicted_class_idx.numpy()

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            original_image: RGB image (H, W, 3)
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Heatmap opacity

        Returns:
            Combined image (uint8)
        """
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Combine images
        combined = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
        return combined

    def predict(self, image_path, img_size=(96, 96)):
        prediction = self.predictor.predict(image_path=image_path)

        return prediction['class']

    def visualize(self, image_path, img_size=(96, 96), save_path=None):
        """
        Generate and visualize Grad-CAM.

        Args:
            image_path: Path to input image
            img_size: Model input size (H, W)
            save_path: Path to save visualization
        """
        # Load and preprocess image
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, (img_size[1], img_size[0]))  # (W, H)
        preprocessed = resized_image / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dim

        # Generate heatmap
        heatmap, class_idx = self.generate_heatmap(preprocessed)

        # Create overlay
        cam_image = self.overlay_heatmap(original_image, heatmap)

        # Predict
        prediction = self.predict(image_path=image_path)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM (Class {class_idx})")
        plt.suptitle(f"Model Prediction: {prediction}")
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save', type=str, help='Path to save output')
    parser.add_argument('--model', type=str, default=os.path.join(options.MODELS_DIR, 'best_model.h5'),
                        help='Path to model file')
    parser.add_argument('--target_layer', type=str, default=None,
                        help='Name of target Conv2D layer (e.g., "block_2_project_BN")')
    args = parser.parse_args()

    try:
        # Load model
        print(f"Loading model from {args.model}...")
        full_model = tf.keras.models.load_model(args.model, compile=False)

        # Build model with dummy input
        dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
        _ = full_model(dummy_input)  # Critical for Sequential models

        # Use MobileNetV2 submodel directly
        mobilenet_submodel = full_model.get_layer('mobilenetv2_0.35_96')
        _ = mobilenet_submodel(dummy_input)  # Build submodel

        # Create Grad-CAM with submodel
        grad_cam = GradCAM(mobilenet_submodel, layer_name=args.target_layer)
        print(f"Using target layer: {grad_cam.layer_name}")

        # Generate visualization
        grad_cam.visualize(
            args.image,
            img_size=(96, 96),
            save_path=args.save
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
