"""
Practical 11: Generative Adversarial Network (GAN)
Builds and trains a Generator and Discriminator on MNIST digits.
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


def experiment_11_gan():
    print("=" * 65)
    print("PRACTICAL 11: GENERATIVE ADVERSARIAL NETWORK (GAN)")
    print("Generator and Discriminator on MNIST")
    print("=" * 65)

    latent_dim = 64

    # 1. Generator Model
    generator = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
    ], name="Generator")

    # 2. Discriminator Model
    discriminator = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="Discriminator")
    discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 3. Combined GAN (Freeze discriminator during generator training)
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    fake_img = generator(gan_input)
    gan_output = discriminator(fake_img)
    gan = keras.Model(gan_input, gan_output, name="GAN")
    gan.compile(optimizer="adam", loss="binary_crossentropy")

    print("\nModel Architectures:")
    generator.summary()
    discriminator.summary()

    # Load small subset of MNIST for quick demo
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train[:1000].astype("float32").reshape(-1, 784) / 255.0
    batch_size = 64

    # Simple training loop for 5 steps
    print("\nTraining GAN for 5 iterations...")
    for step in range(1, 6):
        # 1. Train Discriminator on real + fake images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        real_labels = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise, verbose=0)
        fake_labels = np.zeros((batch_size, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)[0]
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)[0]
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # 2. Train Generator via GAN (aim to fool discriminator -> label 1)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        trick_labels = np.ones((batch_size, 1))
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, trick_labels)

        print(f"Step {step}/5: Discriminator Loss = {d_loss:.4f}, Generator Loss = {g_loss:.4f}")

    # Generate test sample
    test_noise = np.random.normal(0, 1, (1, latent_dim))
    generated_img = generator.predict(test_noise, verbose=0)
    print(f"\nGenerated synthetic image tensor shape: {generated_img.shape}")
    print(f"Generated pixel values range: [{generated_img.min():.2f}, {generated_img.max():.2f}]")
    print("=" * 65)


if __name__ == "__main__":
    experiment_11_gan()
