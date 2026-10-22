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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

np.random.seed(42)
tf.random.set_seed(42)


def experiment_24_anomaly_detection():
    print("=" * 65)
    print("PRACTICAL 24: ANOMALY DETECTION USING AUTOENCODER")
    print("Fraud Detection via High Reconstruction Error")
    print("=" * 65)

    csv_candidates = ["creditcard.csv", "credit_card_fraud.csv", "fraud.csv", "transactions.csv"]
    csv_file = next((f for f in csv_candidates if os.path.exists(f)), None)

    if csv_file:
        print(f"[Data Source] Found local CSV file: '{csv_file}' -> Training on CSV transactions...")
        df = pd.read_csv(csv_file).dropna()
        target_col = "Class" if "Class" in df.columns else df.columns[-1]
        features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        scaler = StandardScaler()
        norm_features = scaler.fit_transform(features).astype("float32")

        normal_mask = (df[target_col] == 0).values
        fraud_mask = (df[target_col] == 1).values

        x_normal = norm_features[normal_mask][:2000]
        x_fraud = norm_features[fraud_mask][:100]
    else:
        print("[Data Source] No local CSV found -> Generating synthetic Credit Card Fraud dataset...")
        num_features = 20
        num_normal = 2000
        num_fraud = 100
        x_normal = np.random.normal(0.0, 1.0, (num_normal, num_features)).astype("float32")
        x_fraud = np.random.normal(2.5, 2.0, (num_fraud, num_features)).astype("float32")

    num_features = x_normal.shape[1]
    # Split: Train ONLY on normal data (semi-supervised anomaly detection)
    train_count = int(len(x_normal) * 0.75)
    x_train_normal = x_normal[:train_count]
    x_test_normal = x_normal[train_count:]

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
