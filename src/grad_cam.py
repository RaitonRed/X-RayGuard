import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse


class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM

        Args:
            model: Keras model
            layer_name (str): Name of target layer
        """
        self.model = model
        self.layer_name = layer_name

        # Find last convolutional layer if not specified
        if self.layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    self.layer_name = layer.name
                    break

        # Create model that outputs target layer and predictions
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

    def generate_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Generate heatmap using Grad-CAM

        Args:
            image: Preprocessed input image
            class_idx (int): Class index (None for predicted class)
            eps (float): Small value to prevent division by zero

        Returns:
            tuple: (heatmap, predicted_class)
        """
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Compute gradients of conv output with respect to loss
        grads = tape.gradient(loss, conv_outputs)

        # Compute importance weights for each channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply weights with conv output and sum over channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy(), class_idx

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image

        Args:
            original_image: Original unpreprocessed image
            heatmap: Generated heatmap
            alpha (float): Transparency factor

        Returns:
            np.array: Overlayed image
        """
        # Resize heatmap to original image dimensions
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Combine original and heatmap
        superimposed_img = heatmap * alpha + original_image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img

    def visualize(self, image_path, img_size=(300, 300), save_path=None):
        """
        Generate and display Grad-CAM visualization

        Args:
            image_path (str): Path to input image
            img_size (tuple): Image dimensions
            save_path (str): Path to save visualization (None to display)
        """
        # Read and preprocess image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        preprocessed_image = cv2.resize(original_image, img_size)
        preprocessed_image = preprocessed_image / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Generate heatmap
        heatmap, class_idx = self.generate_heatmap(preprocessed_image)

        # Overlay on original image
        superimposed_img = self.overlay_heatmap(original_image, heatmap)

        # Display or save results
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title(f'Grad-CAM (Class: {self.model.output_names[0]}_{class_idx})')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save', type=str, help='Path to save output visualization')
    args = parser.parse_args()

    # Load model and create Grad-CAM
    model = tf.keras.models.load_model('models/best_model.h5')
    grad_cam = GradCAM(model)

    # Generate visualization
    if args.save:
        grad_cam.visualize(args.image, save_path=args.save)
    else:
        grad_cam.visualize(args.image)