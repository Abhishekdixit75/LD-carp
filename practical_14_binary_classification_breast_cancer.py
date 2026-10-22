"""
Practical 14: Binary Classification Using Deep Neural Networks
Classifies tumors as malignant or benign using Wisconsin Breast Cancer dataset.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
tf.random.set_seed(42)


def experiment_14_breast_cancer():
    print("=" * 65)
    print("PRACTICAL 14: BINARY CLASSIFICATION (BREAST CANCER)")
    print("Feedforward Deep Neural Network with Binary Cross-Entropy")
    print("=" * 65)

    csv_candidates = ["breast_cancer.csv", "cancer.csv", "data.csv", "wdbc.data.csv"]
    csv_file = next((f for f in csv_candidates if os.path.exists(f)), None)

    if csv_file:
        print(f"[Data Source] Found local CSV file: '{csv_file}' -> Training on CSV data...")
        df = pd.read_csv(csv_file).dropna()
        target_cols = [c for c in ["diagnosis", "target", "class", "label", "Outcome"] if c in df.columns]
        target_col = target_cols[0] if target_cols else df.columns[-1]
        y_series = df[target_col]
        if y_series.dtype == object:
            y_raw = y_series.map({"M": 0, "B": 1, "malignant": 0, "benign": 1}).fillna(0).values
        else:
            y_raw = y_series.values
        x_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    else:
        print("[Data Source] No local CSV found -> Using scikit-learn Wisconsin Breast Cancer dataset...")
        data = load_breast_cancer()
        x_raw, y_raw = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    model = keras.Sequential([
        keras.Input(shape=(x_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    print("\nTraining Deep Neural Network...")
    model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluation
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype("int32")
    test_acc = np.mean(predictions == y_test)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS:")
    print("-" * 65)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))
    print("=" * 65)


if __name__ == "__main__":
    experiment_14_breast_cancer()
