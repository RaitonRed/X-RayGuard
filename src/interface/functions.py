from src.predict import LungDiseasePredictor

# Initialize Predictor
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