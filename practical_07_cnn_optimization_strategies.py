"""
Practical 7: CNN Optimization Strategies
Compare Weight Initializers (Xavier vs He) and Optimizers (SGD, RMSprop, Adam)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)


def experiment_7_cnn_optimization():
    print("=" * 65)
    print("PRACTICAL 7: CNN OPTIMIZATION STRATEGIES")
    print("Comparing Initializations (Xavier vs He) and Optimizers")
    print("=" * 65)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Use subset for fast demonstration
    x_train = x_train[:5000].astype("float32")[..., np.newaxis] / 255.0
    y_train = y_train[:5000]
    x_test = x_test[:1000].astype("float32")[..., np.newaxis] / 255.0
    y_test = y_test[:1000]

    strategies = [
        ("Xavier (Glorot) + SGD", "glorot_uniform", "tanh", keras.optimizers.SGD(0.01)),
        ("He Normal + RMSprop", "he_normal", "relu", keras.optimizers.RMSprop(0.001)),
        ("He Normal + Adam (LR Schedule)", "he_normal", "relu", keras.optimizers.schedules.ExponentialDecay(0.001, 100, 0.9)),
    ]

    results = []
    for name, init, act, opt in strategies:
        print(f"\nTraining CNN with: {name}...")
        model = keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation=act, kernel_initializer=init),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation=act, kernel_initializer=init),
            layers.Dense(10, activation="softmax"),
        ])
        
        # If opt is a schedule, wrap in Adam
        optimizer = keras.optimizers.Adam(opt) if isinstance(opt, keras.optimizers.schedules.LearningRateSchedule) else opt
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        history = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        results.append((name, acc, loss))
        print(f"--> {name}: Test Accuracy = {acc * 100:.2f}%, Test Loss = {loss:.4f}")

    print("\n" + "=" * 65)
    print("OPTIMIZATION STRATEGY COMPARISON:")
    print("-" * 65)
    for name, acc, loss in results:
        print(f"{name:<35} | Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    experiment_7_cnn_optimization()
