import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
import options


class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM wrapper.

        Args:
            model: The full Keras model.
            layer_name: The name of the convolutional layer to target.
        """
        self.model = model
        self.layer_name = layer_name

        # Automatically find last Conv2D layer if not provided
        if self.layer_name is None:
            print("Searching for last Conv2D layer...")
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
                elif isinstance(layer, tf.keras.Model):
                    for inner_layer in reversed(layer.layers):
                        if isinstance(inner_layer, tf.keras.layers.Conv2D):
                            self.layer_name = inner_layer.name
                            break
            if self.layer_name is None:
                raise ValueError("No Conv2D layer found in the model!")

        # Try to resolve the output of the target layer
        try:
            target_layer_output = self.model.get_layer(self.layer_name).output
        except ValueError:
            # Search in sub-models if not in the top-level model
            found = False
            for sub in self.model.layers:
                if isinstance(sub, tf.keras.Model):
                    try:
                        target_layer_output = sub.get_layer(self.layer_name).output
                        found = True
                        break
                    except ValueError:
                        continue
            if not found:
                raise ValueError(f"Could not find layer {self.layer_name} in model or sub-models.")

        # Create a new model that outputs both the target layer and the final prediction
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer_output, self.model.output]
        )

    def generate_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Generate Grad-CAM heatmap for the given image.

        Args:
            image: Preprocessed input image.
            class_idx: Target class index. If None, use predicted class.
            eps: Small epsilon to avoid division by zero.

        Returns:
            heatmap and predicted class index.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            if len(predictions.shape) == 2:
                loss = predictions[:, class_idx]
            else:
                loss = tf.reduce_mean(predictions, axis=[1, 2, 3])  # fallback

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy(), class_idx

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        Overlay the heatmap on the original image.

        Args:
            original_image: RGB image before preprocessing.
            heatmap: Grad-CAM heatmap.
            alpha: Opacity of the heatmap overlay.

        Returns:
            Combined image.
        """
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = heatmap * alpha + original_image * (1 - alpha)
        return np.clip(combined, 0, 255).astype(np.uint8)

    def visualize(self, image_path, img_size=(96, 96), save_path=None):
        """
        Load image, apply Grad-CAM, and display or save result.

        Args:
            image_path: Path to input image.
            img_size: Input size expected by the model.
            save_path: If given, saves output image to this path.
        """
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, img_size)
        preprocessed = resized_image / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)

        heatmap, class_idx = self.generate_heatmap(preprocessed)
        cam_image = self.overlay_heatmap(original_image, heatmap)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM (Class {class_idx})")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save', type=str, help='Path to save output visualization')
    parser.add_argument('--model', type=str, default=options.MODELS_DIR + 'best_model.h5', help='Path to model file')
    args = parser.parse_args()

    try:
        print(f"Loading model from {args.model}...")
        model = tf.keras.models.load_model(args.model, compile=False)

        # Debug zone
        mobilenet_base = model.get_layer('mobilenetv2_0.35_96')
        mobilenet_base.summary()
        exit()

        # Access the internal sub-model (Functional) for Grad-CAM
        base_model = model

        # Run the base model on dummy input to build its graph
        dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
        _ = base_model(dummy_input)

        print("Model summary:")
        base_model.summary()

        # Choose a convolutional layer from base_model
        target_layer = base_model.get_layer('mobilenetv2_0.35_96')  # You can change this if needed

        print("Creating Grad-CAM...")
        grad_cam = GradCAM(base_model, layer_name=target_layer.name)

        print(f"Using layer: {grad_cam.layer_name}")

        print(f"Generating visualization for {args.image}...")
        if args.save:
            grad_cam.visualize(args.image, save_path=args.save)
            print(f"Saved to {args.save}")
        else:
            grad_cam.visualize(args.image)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
