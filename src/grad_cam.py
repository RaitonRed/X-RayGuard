import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
import options


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

        # Ensure the model has been built
        if not model.built:
            # Get input shape from the first layer if available
            if hasattr(model, 'input_shape') and model.input_shape is not None:  # Check model.input_shape is not None
                input_shape = model.input_shape
            elif hasattr(model.layers[0], 'input_shape') and model.layers[
                0].input_shape is not None:  # Fallback to first layer
                input_shape = model.layers[0].input_shape
            else:
                # Default input shape if we can't determine it and model isn't built
                # This might need adjustment based on your specific model expectations
                print(
                    "Warning: Model not built and input shape couldn't be reliably determined. Using default (None, 224, 224, 3).")
                input_shape = (None, 224, 224, 3)

            # Create a dummy input to build the model if input_shape is valid
            if input_shape[1:] and all(isinstance(dim, int) and dim > 0 for dim in input_shape[1:]):
                dummy_input_shape = [1 if dim is None else dim for dim in input_shape]  # Replace None with 1 for batch
                # Ensure all dimensions are specified for the dummy input, skip batch if None
                dummy_input = tf.zeros(tuple(d for d in dummy_input_shape if d is not None))
                if dummy_input.ndim == len(input_shape) - 1:  # if batch dim was None and skipped
                    dummy_input = tf.expand_dims(dummy_input, axis=0)

                try:
                    _ = model(dummy_input)
                except Exception as e:
                    print(f"Warning: Failed to build model with dummy input: {e}")
                    print("Grad-CAM might fail if the model is not properly built or layer names are incorrect.")
            else:
                print(f"Warning: Could not create valid dummy input from shape {input_shape}. Model building skipped.")

        # Find last convolutional layer if layer_name is not specified
        if self.layer_name is None:
            print("Info: `layer_name` not specified. Attempting to find the last Conv2D layer...")
            found_layer_in_auto_search = False
            # Iterate through the main model's layers in reverse
            for top_level_layer in reversed(self.model.layers):
                if isinstance(top_level_layer, tf.keras.layers.Conv2D):
                    self.layer_name = top_level_layer.name
                    found_layer_in_auto_search = True
                    break
                # If the top-level layer is a Model, search its layers
                elif isinstance(top_level_layer, tf.keras.Model):
                    for inner_layer in reversed(top_level_layer.layers):
                        if isinstance(inner_layer, tf.keras.layers.Conv2D):
                            self.layer_name = inner_layer.name  # This name is relative to the inner_model
                            # For GradCAM model construction, we need its output directly or the sub_model context
                            found_layer_in_auto_search = True
                            break
                    if found_layer_in_auto_search:
                        break
            if not found_layer_in_auto_search:
                print(
                    "Warning: Could not automatically find a Conv2D layer. Using the second to last layer of the main model if available.")
                if len(self.model.layers) > 1:
                    self.layer_name = self.model.layers[-2].name
                else:
                    raise ValueError(
                        "GradCAM: Model doesn't have enough layers to auto-select for Grad-CAM, and no Conv2D layer found.")
            print(f"Info: Automatically selected layer: '{self.layer_name}'")

        # --- MODIFIED BLOCK FOR RESOLVING TARGET LAYER OUTPUT ---
        resolved_target_layer_output = None
        try:
            # Attempt to get the layer directly from the main model.
            resolved_target_layer_output = self.model.get_layer(self.layer_name).output
            # print(f"Info: Target layer '{self.layer_name}' found in the top-level model.")
        except ValueError:
            # If not found in top-level, search in direct sub-models.
            # print(f"Info: Target layer '{self.layer_name}' not in top-level. Searching sub-models...")
            found_in_submodel_flag = False
            for sub_model_candidate in self.model.layers:
                if isinstance(sub_model_candidate, tf.keras.Model):
                    try:
                        target_sub_layer = sub_model_candidate.get_layer(self.layer_name)
                        resolved_target_layer_output = target_sub_layer.output
                        # print(f"Info: Target layer '{self.layer_name}' found in sub-model '{sub_model_candidate.name}'.")
                        found_in_submodel_flag = True
                        break
                    except ValueError:
                        # Layer not in this sub-model, continue.
                        pass

            if not found_in_submodel_flag:
                detailed_error_msg = f"Error creating Grad-CAM model: Layer '{self.layer_name}' not found in the model or its direct sub-models.\n"
                detailed_error_msg += "Top-level layers of the model:\n"
                for i, L_top in enumerate(self.model.layers):
                    detailed_error_msg += f"  [{i}] {L_top.name} (Type: {type(L_top).__name__})\n"
                    if isinstance(L_top, tf.keras.Model):
                        detailed_error_msg += f"    Sub-layers of '{L_top.name}':\n"
                        try:
                            for j, L_sub in enumerate(L_top.layers):
                                detailed_error_msg += f"      [{j}] {L_sub.name} (Type: {type(L_sub).__name__})\n"
                        except Exception:
                            detailed_error_msg += "      (Could not list sub-layers)\n"
                print(detailed_error_msg)  # Print to console for user.
                raise ValueError(
                    f"GradCAM: Layer '{self.layer_name}' could not be resolved. Check layer name and model structure.")

        if resolved_target_layer_output is None:
            # This should be caught by the logic above, but as a safeguard.
            raise ValueError(
                f"GradCAM: Failed to obtain output for layer '{self.layer_name}'. This indicates an unexpected issue.")

        # Create model that outputs target layer's activations and final model predictions.
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[resolved_target_layer_output, self.model.output]
        )
        # --- END OF MODIFIED BLOCK ---

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

    def visualize(self, image_path, img_size=(96, 96), save_path=None):
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
    parser.add_argument('--model', type=str, default=options.MODELS_DIR+'best_model.h5', help='Path to model file')
    args = parser.parse_args()

    try:
        # Load model and create Grad-CAM
        print(f"Loading model from {args.model}...")
        model = tf.keras.models.load_model(args.model)
        mobile_net_layer = model.get_layer('mobilenetv2_0.35_96')

        for idx, layer in enumerate(mobile_net_layer.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"Layer {idx}: {layer.name}")

        # Print model summary for debugging
        print("Model summary:")
        model.summary()

        print("Creating Grad-CAM...")
        mobile_net_layer = model.get_layer('mobilenetv2_0.35_96')
        target_layer = mobile_net_layer.get_layer('Conv_1')
        grad_cam = GradCAM(model, layer_name=target_layer.name)

        print(f"Using layer: {grad_cam.layer_name} for Grad-CAM")

        # Generate visualization
        print(f"Generating visualization for {args.image}...")
        if args.save:
            grad_cam.visualize(args.image, save_path=args.save)
            print(f"Visualization saved to {args.save}")
        else:
            grad_cam.visualize(args.image)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
