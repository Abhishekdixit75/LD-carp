"""
Practical 24: Deep Learning for Anomaly Detection (Credit Card Fraud)
Autoencoder trained on normal transactions only; flags anomalies via reconstruction error.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, f1_score

np.random.seed(42)
tf.random.set_seed(42)


def experiment_24_anomaly_detection():
    print("=" * 65)
    print("PRACTICAL 24: ANOMALY DETECTION USING AUTOENCODER")
    print("Fraud Detection via High Reconstruction Error")
    print("=" * 65)

    num_features = 20
    num_normal = 2000
    num_fraud = 100

    # 1. Synthesize normal (Gaussian N(0, 1)) vs fraudulent (shifted variance/mean) transactions
    x_normal = np.random.normal(0.0, 1.0, (num_normal, num_features)).astype("float32")
    x_fraud = np.random.normal(2.5, 2.0, (num_fraud, num_features)).astype("float32")

    # Split: Train ONLY on normal data (semi-supervised anomaly detection)
    x_train_normal = x_normal[:1500]
    x_test_normal = x_normal[1500:]

    # Test set combines remaining normal + all fraudulent transactions
    x_test = np.vstack([x_test_normal, x_fraud])
    y_test = np.array([0] * len(x_test_normal) + [1] * len(x_fraud))

    # 2. Build Autoencoder
    model = keras.Sequential([
        keras.Input(shape=(num_features,)),
        layers.Dense(14, activation="relu"),
        layers.Dense(7, activation="relu", name="bottleneck"),
        layers.Dense(14, activation="relu"),
        layers.Dense(num_features),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    print("\nTraining Autoencoder exclusively on NORMAL transactions...")
    model.fit(x_train_normal, x_train_normal, epochs=15, batch_size=32, validation_split=0.1, verbose=0)

    # 3. Compute Reconstruction Errors
    reconstructions = model.predict(x_test, verbose=0)
    mse_errors = np.mean((x_test - reconstructions) ** 2, axis=1)

    # Choose threshold: 95th percentile of normal training errors
    train_reconstructions = model.predict(x_train_normal, verbose=0)
    train_mse = np.mean((x_train_normal - train_reconstructions) ** 2, axis=1)
    threshold = np.percentile(train_mse, 95)

    # Flag fraud if error > threshold
    predicted_fraud = (mse_errors > threshold).astype(int)

    print("\n" + "=" * 65)
    print(f"ANOMALY DETECTION THRESHOLD: {threshold:.4f}")
    print("-" * 65)
    print("Confusion Matrix (0 = Normal, 1 = Fraud):")
    print(confusion_matrix(y_test, predicted_fraud))
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_fraud, target_names=["Normal", "Fraud"]))
    print(f"F1-Score for Fraud Detection: {f1_score(y_test, predicted_fraud):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    experiment_24_anomaly_detection()
