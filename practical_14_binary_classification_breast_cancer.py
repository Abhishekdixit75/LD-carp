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

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
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
