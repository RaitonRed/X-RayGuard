import tensorflow as tf
import numpy as np
import cv2
import os
import argparse


class LungDiseasePredictor:
    def __init__(self, model_path='models/best_model.h5', img_size=(300, 300)):
        """
        Initialize predictor

        Args:
            model_path (str): Path to saved model
            img_size (tuple): Input image dimensions
        """
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.class_names = ['COVID-19', 'Normal', 'Pneumonia']  # Should match training data

    def preprocess_image(self, image_path):
        """
        Preprocess input image

        Args:
            image_path (str): Path to input image

        Returns:
            np.array: Preprocessed image
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image_path):
        """
        Predict class for input image

        Args:
            image_path (str): Path to input image

        Returns:
            dict: Prediction results with class and probabilities
        """
        # Preprocess image
        image = self.preprocess_image(image_path)

        # Make prediction
        probs = self.model.predict(image)[0]
        pred_class = np.argmax(probs)

        # Format results
        result = {
            'class': self.class_names[pred_class],
            'confidence': float(probs[pred_class]),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.class_names, probs)
            }
        }

        return result


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    # Initialize predictor
    predictor = LungDiseasePredictor()

    # Make prediction
    if os.path.exists(args.image):
        prediction = predictor.predict(args.image)
        print("Prediction Results:")
        print(f"Class: {prediction['class']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print("Probabilities:")
        for cls, prob in prediction['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    else:
        print(f"Image {args.image} not found. Please provide a valid path.")