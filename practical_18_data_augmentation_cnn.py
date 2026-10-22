"""
Practical 18: Image Classification Using Data Augmentation
Compares CNN performance and overfitting with vs without Data Augmentation.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)


def experiment_18_data_augmentation():
    print("=" * 65)
    print("PRACTICAL 18: IMAGE CLASSIFICATION WITH DATA AUGMENTATION")
    print("Comparing Overfitting & Accuracy: Baseline vs Augmented CNN")
    print("=" * 65)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Small training set to clearly demonstrate overfitting reduction via augmentation
    x_train = x_train[:2000].astype("float32")[..., np.newaxis] / 255.0
    y_train = y_train[:2000]
    x_test = x_test[:500].astype("float32")[..., np.newaxis] / 255.0
    y_test = y_test[:500]

    # Data Augmentation pipeline
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="Augmentation_Pipeline")

    def build_cnn(use_augmentation=False):
        inputs = keras.Input(shape=(28, 28, 1))
        x = inputs
        if use_augmentation:
            x = data_augmentation(x)
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    # 1. Baseline Model (No Augmentation)
    print("\n1. Training Baseline CNN (Without Data Augmentation)...")
    base_model = build_cnn(use_augmentation=False)
    h_base = base_model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.2, verbose=0)
    base_loss, base_acc = base_model.evaluate(x_test, y_test, verbose=0)

    # 2. Augmented Model
    print("2. Training Augmented CNN (With Rotation, Flip, Zoom, Shift)...")
    aug_model = build_cnn(use_augmentation=True)
    h_aug = aug_model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.2, verbose=0)
    aug_loss, aug_acc = aug_model.evaluate(x_test, y_test, verbose=0)

    # Calculate Overfitting gap (Train Accuracy - Val Accuracy)
    base_gap = (h_base.history['accuracy'][-1] - h_base.history['val_accuracy'][-1]) * 100
    aug_gap = (h_aug.history['accuracy'][-1] - h_aug.history['val_accuracy'][-1]) * 100

    print("\n" + "=" * 65)
    print("DATA AUGMENTATION COMPARISON:")
    print("-" * 65)
    print(f"{'Model':<25} | {'Test Accuracy':<15} | {'Overfitting Gap (Train-Val)'}")
    print("-" * 65)
    print(f"{'Without Augmentation':<25} | {base_acc * 100:>13.2f}% | {base_gap:>10.2f}%")
    print(f"{'With Augmentation':<25} | {aug_acc * 100:>13.2f}% | {aug_gap:>10.2f}%")
    print("=" * 65)
    print("Conclusion: Data Augmentation synthesizes diverse visual variations,")
    print("reducing the generalization gap and preventing overfitting.")
    print("=" * 65)


if __name__ == "__main__":
    experiment_18_data_augmentation()
