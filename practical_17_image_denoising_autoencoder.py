"""
Practical 17: Image Denoising Using Autoencoders
Restores noisy MNIST digit images using an autoencoder; reports MSE and PSNR.
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


def experiment_17_denoising_autoencoder():
    print("=" * 65)
    print("PRACTICAL 17: IMAGE DENOISING AUTOENCODER")
    print("Reconstructing Noisy MNIST Images; Evaluating MSE and PSNR")
    print("=" * 65)

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # Fast subset
    x_train_clean = x_train[:10000].astype("float32") / 255.0
    x_test_clean = x_test[:2000].astype("float32") / 255.0

    # Add Gaussian noise
    noise_factor = 0.35
    rng = np.random.default_rng(42)
    x_train_noisy = np.clip(x_train_clean + rng.normal(0, noise_factor, x_train_clean.shape), 0.0, 1.0)
    x_test_noisy = np.clip(x_test_clean + rng.normal(0, noise_factor, x_test_clean.shape), 0.0, 1.0)

    # Flatten for dense autoencoder
    x_train_noisy = x_train_noisy.reshape(-1, 784)
    x_train_clean = x_train_clean.reshape(-1, 784)
    x_test_noisy = x_test_noisy.reshape(-1, 784)
    x_test_clean = x_test_clean.reshape(-1, 784)

    # Denoising Autoencoder Architecture
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu", name="latent_space"),
        layers.Dense(256, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    print("\nTraining Denoising Autoencoder (Noisy -> Clean)...")
    model.fit(x_train_noisy, x_train_clean, epochs=4, batch_size=256, validation_split=0.1, verbose=1)

    # Evaluate reconstruction error
    denoised_predictions = model.predict(x_test_noisy, batch_size=256, verbose=0)
    mse = np.mean((x_test_clean - denoised_predictions) ** 2)
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))

    print("\n" + "=" * 65)
    print("DENOISING RECONSTRUCTION RESULTS:")
    print("-" * 65)
    print(f"Reconstruction Mean Squared Error (MSE): {mse:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR):       {psnr:.2f} dB")
    print("=" * 65)


if __name__ == "__main__":
    experiment_17_denoising_autoencoder()
