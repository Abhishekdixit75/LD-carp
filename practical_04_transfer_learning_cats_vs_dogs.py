import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)


def experiment_4_transfer_learning():
    """Transfer Learning with MobileNetV2: Feature Extraction vs Fine-Tuning."""
    # 1. Sample cat (0) and dog (1) images
    x_train = np.random.uniform(0, 255, (200, 96, 96, 3)).astype("float32")
    y_train = np.random.randint(0, 2, (200, 1))
    x_test = np.random.uniform(0, 255, (50, 96, 96, 3)).astype("float32")
    y_test = np.random.randint(0, 2, (50, 1))

    # 2. Pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"
    )

    # --- Phase 1: Feature Extraction (Freeze base model) ---
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=(96, 96, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1),  # normalize to [-1, 1]
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("\n--- Phase 1: Feature Extraction (Base Frozen) ---")
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    fe_loss, fe_acc = model.evaluate(x_test, y_test, verbose=0)

    # --- Phase 2: Fine-Tuning (Unfreeze base model with low learning rate) ---
    print("\n--- Phase 2: Fine-Tuning (Base Unfrozen) ---")
    base_model.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    ft_loss, ft_acc = model.evaluate(x_test, y_test, verbose=0)

    # --- Comparison ---
    print("\n--- Comparison Results ---")
    print(f"Feature Extraction Accuracy: {fe_acc * 100:.2f}% (Loss: {fe_loss:.4f})")
    print(f"Fine-Tuning Accuracy:        {ft_acc * 100:.2f}% (Loss: {ft_loss:.4f})")


if __name__ == "__main__":
    experiment_4_transfer_learning()
