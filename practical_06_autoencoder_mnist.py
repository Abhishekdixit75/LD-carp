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


def experiment_6_basic_autoencoder():
    """Compress and reconstruct MNIST digit images with a dense autoencoder."""
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # Fast demonstration subset
    x_train = x_train[:6000].astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test[:1000].astype("float32").reshape(-1, 784) / 255.0

    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(32, activation="relu", name="encoded"),
        layers.Dense(128, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    model.fit(x_train, x_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

    reconstructed = model.predict(x_test[:10], verbose=0)
    reconstruction_mse = np.mean((x_test - model.predict(x_test, batch_size=256, verbose=0)) ** 2)
    print(f"\n--> Test Reconstruction MSE: {reconstruction_mse:.6f}")

    fig, axes = plt.subplots(2, 10, figsize=(13, 3))
    for i in range(10):
        axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title("Original")
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("autoencoder_reconstruction.png")
    plt.close()
    print("Reconstruction plot saved to: autoencoder_reconstruction.png")


if __name__ == "__main__":
    experiment_6_basic_autoencoder()  