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
def experiment_2_house_price_regression():
    """California Housing regression with L1, L2, and dropout variants."""
    csv_candidates = ["housing.csv", "california_housing.csv", "california.csv", "data.csv"]
    csv_file = next((f for f in csv_candidates if os.path.exists(f)), None)

    if csv_file:
        print(f"[Data Source] Found local CSV file: '{csv_file}' -> Training on CSV data...")
        df = pd.read_csv(csv_file).dropna()
        target_cols = [c for c in ["MedHouseVal", "median_house_value", "Price", "price", "target"] if c in df.columns]
        target_col = target_cols[0] if target_cols else df.columns[-1]
        x_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
        y_raw = df[target_col].values
    else:
        try:
            print("[Data Source] No local CSV found -> Loading California Housing via scikit-learn...")
            data = fetch_california_housing()
            x_raw, y_raw = data.data, data.target
        except Exception:
            print("[Data Source] Generating synthetic California Housing benchmark dataset...")
            x_raw = np.random.randn(2000, 8).astype("float32")
            y_raw = (x_raw[:, 0] * 0.5 + x_raw[:, 1] * 0.3 + 2.0).astype("float32")

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw, y_raw, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    def build_model(kind):
        if kind == "L1":
            regularizer = keras.regularizers.l1(0.001)
        elif kind == "L2":
            regularizer = keras.regularizers.l2(0.001)
        else:
            regularizer = None
        model_layers = [keras.Input(shape=(x_train.shape[1],)), layers.Dense(64, activation="relu", kernel_regularizer=regularizer)]
        if kind == "Dropout":
            model_layers.append(layers.Dropout(0.2))
        model_layers.extend([layers.Dense(32, activation="relu", kernel_regularizer=regularizer), layers.Dense(1)])
        model = keras.Sequential(model_layers)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    for kind in ["L1", "L2", "Dropout"]:
        print(f"\nTraining model with {kind}...")
        model = build_model(kind)
        model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2, verbose=0)
        loss, mae = model.evaluate(x_test, y_test, verbose=0)
        print(f"{kind}: Test MSE={loss:.4f}, Test MAE={mae:.4f}, Test RMSE={np.sqrt(loss):.4f}")


if __name__ == "__main__":
    experiment_2_house_price_regression()