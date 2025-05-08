import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models, optimizers, callbacks
from data_preprocessing import load_and_preprocess_data
import matplotlib.pyplot as plt
import os


def build_model(input_shape=(300, 300, 3), num_classes=3):
    """
    Build model based on EfficientNetB3 with transfer learning

    Args:
        input_shape (tuple): Input image dimensions
        num_classes (int): Number of output classes

    Returns:
        tf.keras.Model: Compiled model
    """
    # Load base model with ImageNet weights (excluding top)
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )

    # Freeze base layers for fine-tuning
    base_model.trainable = False

    # Build complete model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(data_dir, model_save_path='models/best_model.h5', epochs=20):
    """
    Train model with input data

    Args:
        data_dir (str): Path to data directory
        model_save_path (str): Path to save trained model
        epochs (int): Number of training epochs

    Returns:
        tuple: (trained model, training history)
    """
    # Create directory for saving models
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset, class_names = load_and_preprocess_data(data_dir)

    # Build model
    model = build_model()
    model.summary()

    # Define callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        )
    ]

    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks_list
    )

    # Save final model
    model.save(model_save_path)

    return model, history, test_dataset, class_names


def plot_training_history(history, save_dir='results/training_plots'):
    """
    Plot training accuracy and loss metrics

    Args:
        history: History object returned from model.fit()
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()


if __name__ == '__main__':
    # Example usage
    DATA_DIR = 'data/raw/COVID-19_Radiography_Dataset'
    model, history, test_dataset, class_names = train_model(DATA_DIR)
    plot_training_history(history)
    print("Model training completed successfully!")