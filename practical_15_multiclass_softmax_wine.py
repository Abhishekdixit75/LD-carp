"""
Practical 15: Multiclass Classification Using Softmax
Three-class classification on the Wine dataset with precision, recall, and confusion matrix.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

np.random.seed(42)
tf.random.set_seed(42)


def experiment_15_wine_classification():
    print("=" * 65)
    print("PRACTICAL 15: MULTICLASS CLASSIFICATION USING SOFTMAX")
    print("Wine Dataset (3 Classes) with Categorical Cross-Entropy")
    print("=" * 65)

    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes=3)

    model = keras.Sequential([
        keras.Input(shape=(x_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("\nTraining Multiclass Softmax Model...")
    model.fit(x_train, y_train_onehot, epochs=40, batch_size=16, validation_split=0.2, verbose=0)

    # Predictions & Metrics
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    acc = np.mean(predictions == y_test)
    prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec = recall_score(y_test, predictions, average="weighted", zero_division=0)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS:")
    print("-" * 65)
    print(f"Test Accuracy:      {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec * 100:.2f}%")
    print(f"Weighted Recall:    {rec * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names, zero_division=0))
    print("=" * 65)


if __name__ == "__main__":
    experiment_15_wine_classification()
