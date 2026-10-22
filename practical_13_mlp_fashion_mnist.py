"""
Practical 13: Multilayer Perceptron (MLP) for Classification
Compares MLP Depth, Width, and Activations (ReLU, Tanh, Sigmoid) on Fashion-MNIST.
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


def experiment_13_fashion_mnist_mlp():
    print("=" * 65)
    print("PRACTICAL 13: MLP ARCHITECTURES ON FASHION-MNIST")
    print("Comparing Layers, Neurons, and Activations (ReLU vs Tanh vs Sigmoid)")
    print("=" * 65)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # Fast training subset
    x_train = x_train[:10000].astype("float32") / 255.0
    y_train = y_train[:10000]
    x_test = x_test[:2000].astype("float32") / 255.0
    y_test = y_test[:2000]

    configurations = [
        ("1 Hidden Layer (64, ReLU)", [64], "relu"),
        ("2 Hidden Layers (128, 64, ReLU)", [128, 64], "relu"),
        ("2 Hidden Layers (128, 64, Tanh)", [128, 64], "tanh"),
        ("2 Hidden Layers (128, 64, Sigmoid)", [128, 64], "sigmoid"),
    ]

    results = []
    for label, units_list, activation in configurations:
        print(f"\nTraining configuration: {label}...")
        model_layers = [keras.Input(shape=(28, 28)), layers.Flatten()]
        for units in units_list:
            model_layers.append(layers.Dense(units, activation=activation))
        model_layers.append(layers.Dense(10, activation="softmax"))

        model = keras.Sequential(model_layers)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=0)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        results.append((label, accuracy, loss))
        print(f"--> {label}: Test Accuracy = {accuracy * 100:.2f}%, Loss = {loss:.4f}")

    print("\n" + "=" * 65)
    print("MLP ARCHITECTURE COMPARISON RESULTS:")
    print("-" * 65)
    for label, accuracy, loss in results:
        print(f"{label:<38} | Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    experiment_13_fashion_mnist_mlp()
