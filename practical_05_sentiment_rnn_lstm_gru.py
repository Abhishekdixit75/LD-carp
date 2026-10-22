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



def experiment_5_imdb_rnn_lstm_gru():
    """Compare SimpleRNN, LSTM, and GRU for positive/negative IMDB review sentiment."""
    vocab_size = 10000
    max_length = 200
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    # Fast demonstration subset for quick lab execution
    x_train, y_train = x_train[:3000], y_train[:3000]
    x_test, y_test = x_test[:800], y_test[:800]
    x_train = keras.utils.pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_test = keras.utils.pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")

    results = {}
    for cell_name in ["SimpleRNN", "LSTM", "GRU"]:
        print(f"\nTraining {cell_name} model...")
        if cell_name == "SimpleRNN":
            recurrent_layer = layers.SimpleRNN(32)
        elif cell_name == "LSTM":
            recurrent_layer = layers.LSTM(32)
        else:
            recurrent_layer = layers.GRU(32)

        model = keras.Sequential([
            keras.Input(shape=(max_length,), dtype="int32"),
            layers.Embedding(vocab_size, 64),
            recurrent_layer,
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        start = time.time()
        model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=1)
        seconds = time.time() - start
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        results[cell_name] = (accuracy, seconds)
        print(f"{cell_name}: test accuracy={accuracy:.4f}; training time={seconds:.1f}s")

    print("\nComparison:")
    for name, (accuracy, seconds) in results.items():
        print(f"{name:10s} accuracy={accuracy:.4f}, training_time={seconds:.1f}s")


if __name__ == "__main__":
    experiment_5_imdb_rnn_lstm_gru()
