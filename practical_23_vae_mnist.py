"""
Practical 23: Variational Autoencoder (VAE)
Learns a continuous 2D latent space of MNIST digits; generates new digits by sampling.
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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z via reparameterization trick."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}


def experiment_23_vae():
    print("=" * 65)
    print("PRACTICAL 23: VARIATIONAL AUTOENCODER (VAE)")
    print("Latent Space Modeling & Synthetic Digit Generation on MNIST")
    print("=" * 65)

    latent_dim = 2

    # 1. Encoder
    enc_inputs = keras.Input(shape=(784,))
    x = layers.Dense(128, activation="relu")(enc_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(enc_inputs, [z_mean, z_log_var, z], name="Encoder")

    # 2. Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(latent_inputs)
    dec_outputs = layers.Dense(784, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, dec_outputs, name="Decoder")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))

    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train[:5000].astype("float32").reshape(-1, 784) / 255.0

    print("\nTraining Variational Autoencoder...")
    vae.fit(x_train, epochs=3, batch_size=128, verbose=1)

    # 3. Generate New Digits by Sampling from Latent Space
    random_latent_point = np.array([[0.5, -0.8]])  # Sample z in 2D Gaussian space
    generated_digit = decoder.predict(random_latent_point, verbose=0)

    print("\n" + "=" * 65)
    print("LATENT SAMPLING & GENERATION:")
    print("-" * 65)
    print(f"Sampled Latent Vector (z): {random_latent_point.tolist()[0]}")
    print(f"Generated Digit Tensor Shape: {generated_digit.shape} (784 pixels)")
    print(f"Pixel Intensity Range: [{generated_digit.min():.3f}, {generated_digit.max():.3f}]")
    print("=" * 65)


if __name__ == "__main__":
    experiment_23_vae()
