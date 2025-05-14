import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import options

def evaluate_model(model, test_dataset, class_names, save_dir=options.RESULTS_DIR):
    """
    Evaluate model performance on test data

    Args:
        model: Trained model
        test_dataset: Test dataset
        class_names: List of class names
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Predict on test data
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Save results
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write(f"\nOverall Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    # Load model
    model = tf.keras.models.load_model(os.path.join(options.MODELS_DIR, 'best_model.h5'))

    # Load test data
    from data_preprocessing import load_and_preprocess_data

    _, _, test_dataset, class_names = load_and_preprocess_data(options.DATA_DIR)

    # Evaluate model
    evaluate_model(model, test_dataset, class_names)