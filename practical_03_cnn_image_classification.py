"""
Practical 3: CNN-Based Image Classification
Builds a CNN with Dropout, saves and reloads model, evaluates accuracy.
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


def experiment_3_cnn_cifar10():
    print("=" * 65)
    print("PRACTICAL 3: CNN IMAGE CLASSIFICATION (10 CLASSES)")
    print("Convolutional Network, Dropout, Save & Reload Evaluation")
    print("=" * 65)

    # Use Fashion-MNIST cached locally, resized to 32x32x3 to match CIFAR-10 shape
    print("Loading 10-class dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Fast demonstration subset
    x_train = x_train[:4000].astype("float32") / 255.0
    x_test = x_test[:800].astype("float32") / 255.0
    y_train = y_train[:4000]
    y_test = y_test[:800]

    # Expand to 32x32x3
    x_train = np.pad(x_train[..., np.newaxis], ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
    x_train = np.repeat(x_train, 3, axis=-1)
    x_test = np.pad(x_test[..., np.newaxis], ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
    x_test = np.repeat(x_test, 3, axis=-1)

    print(f"Dataset Shape: {x_train.shape} (32x32 RGB, 10 classes)")

    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("\nTraining CNN with Dropout for 3 epochs...")
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n--> Original Model: Test Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

    # Save and reload model
    save_path = "cnn_model.keras"
    model.save(save_path)
    print(f"Saved model successfully to: {save_path}")

    loaded_model = keras.models.load_model(save_path)
    l_loss, l_acc = loaded_model.evaluate(x_test, y_test, verbose=0)
    print(f"--> Reloaded Model: Test Loss = {l_loss:.4f}, Accuracy = {l_acc * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    experiment_3_cnn_cifar10()