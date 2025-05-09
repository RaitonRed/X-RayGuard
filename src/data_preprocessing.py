import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

def load_and_preprocess_data(data_dir, img_size=(96, 96), test_size=0.2, val_size=0.1, seed=42, batch_size=16):
    """
    Load and preprocess image data from classified directories

    Args:
        data_dir (str): Path to root data directory
        img_size (tuple): Output image dimensions (64x64 برای سیستم ضعیف)
        test_size (float): Proportion for test split
        val_size (float): Proportion for validation from training data
        seed (int): Random seed for reproducibility
        batch_size (int): Batch size for dataset iteration

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names)
    """
    classes = ['COVID', 'Normal', 'Viral Pneumonia']
    class_labels = {'COVID': 0, 'Normal': 1, 'Viral Pneumonia': 2}

    file_paths = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name, 'images')
        print(f"Checking directory: {class_dir}")
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(class_labels[class_name])

    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size), # Correct calculation for val_size from remaining training data
        stratify=y_train,
        random_state=seed
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    def preprocess_image(image_path, label): # image_path is a byte string from tf.numpy_function
        """Convert file path to preprocessed image"""
        # Decode byte string to normal string for file operations
        image = cv2.imread(image_path.decode('utf-8'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size) # img_size is defined in the outer scope
        image = image / 255.0
        image = image.astype(np.float32)
        return image, np.int32(label)

    def create_dataset(paths, current_labels, current_batch_size=16, shuffle=False): # Renamed to avoid conflict
        """Create TensorFlow dataset from file paths"""
        dataset = tf.data.Dataset.from_tensor_slices((paths, current_labels))

        def wrapped_preprocess(path_tensor, label_tensor):
            # tf.numpy_function processes Python functions.
            # preprocess_image expects a string path and an integer label.
            img, lbl = tf.numpy_function(
                preprocess_image,
                [path_tensor, label_tensor],
                [tf.float32, tf.int32]
            )
            # Crucially, set the shape for the outputs of tf.numpy_function.
            # img_size is captured from the outer scope of load_and_preprocess_data
            img.set_shape((*img_size, 3)) # e.g., (96, 96, 3)
            lbl.set_shape([]) # Label is a scalar, so its shape is empty.
            return img, lbl

        dataset = dataset.map(wrapped_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # The explicit reshape previously done might now be redundant if set_shape works effectively.
        # If issues persist, you might try re-adding it, but set_shape is generally preferred.
        # dataset = dataset.map(
        #     lambda img, label: (tf.reshape(img, (*img_size, 3)), label)
        # )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(paths) if paths else 1024) # Adjust buffer size
        dataset = dataset.batch(current_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(X_train, y_train, current_batch_size=16, shuffle=True)
    val_dataset = create_dataset(X_val, y_val, current_batch_size=16)
    test_dataset = create_dataset(X_test, y_test, current_batch_size=16)

    return train_dataset, val_dataset, test_dataset, classes