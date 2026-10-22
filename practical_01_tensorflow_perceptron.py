"""Easy Deep Learning Practicals (experiments 1, 2, 3, 5, 6, 13, 14, 15, 16, 17)

Run one experiment at a time:
    python easy_deep_learning_practicals.py
Then enter an experiment number when prompted.

Install dependencies:
    pip install tensorflow scikit-learn pandas matplotlib numpy

Notes:
- Some datasets are downloaded on first use and therefore require internet access.
- Experiment 2 uses scikit-learn's California Housing data downloader.
- Experiment 16 downloads the Jena Climate dataset from the URL used in the
  TensorFlow time-series tutorial.
"""

import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
)

np.random.seed(42)
tf.random.set_seed(42)


def experiment_1_perceptron_and_mlp():
    """TensorFlow tensor operations; compare single-layer perceptron and MLP on MNIST."""
    print("\n--- Basic TensorFlow tensor operations ---")
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    print("Tensor a:", a.numpy())
    print("Tensor b:", b.numpy())
    print("Addition:", (a + b).numpy())
    print("Element-wise multiplication:", (a * b).numpy())
    print("2 x 3 matrix reshaped from 1..6:\n", tf.reshape(tf.range(1, 7), (2, 3)).numpy())

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # A single-layer perceptron: no hidden dense layer.
    perceptron = keras.Sequential([
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ])
    perceptron.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    perceptron.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
    p_loss, p_acc = perceptron.evaluate(x_test, y_test, verbose=0)

    # An MLP adds a hidden layer.
    mlp = keras.Sequential([
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    mlp.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
    m_loss, m_acc = mlp.evaluate(x_test, y_test, verbose=0)

    print(f"Perceptron test accuracy: {p_acc:.4f}")
    print(f"MLP test accuracy:        {m_acc:.4f}")


if __name__ == "__main__":
    experiment_1_perceptron_and_mlp()   