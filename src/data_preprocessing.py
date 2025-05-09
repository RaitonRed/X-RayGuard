import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2


def load_and_preprocess_data(data_dir, img_size=(128, 128), test_size=0.2, val_size=0.1, seed=42):
    """
    Load and preprocess image data from classified directories

    Args:
        data_dir (str): Path to root data directory
        img_size (tuple): Output image dimensions
        test_size (float): Proportion for test split
        val_size (float): Proportion for validation from training data
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_gen, val_gen, test_gen, class_names)
    """
    classes = ['COVID', 'Normal', 'Viral Pneumonia']
    class_labels = {'COVID': 0, 'Normal': 1, 'Viral Pneumonia': 2}

    # Collect file paths and labels
    file_paths = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name, 'images')
        if not os.path.exists(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(class_labels[class_name])

    # Split data into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=seed
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    def preprocess_image(image_path, label):
        """Convert file path to preprocessed image"""
        image = cv2.imread(image_path.decode('utf-8'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image / 255.0
        return image.astype(np.float32), label

    def create_dataset(file_paths, labels, batch_size=32, shuffle=False, augment=False):
        """Create TensorFlow dataset from file paths"""
        def create_dataset(paths, labels, batch_size=16, shuffle=False):
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            dataset = dataset.map(
                lambda x, y: tf.numpy_function(preprocess_image, [x, y], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            if shuffle:
                dataset = dataset.shuffle(buffer_size=1024)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset

    batch_size = 32
    train_dataset = create_dataset(X_train, y_train, shuffle=True)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, classes