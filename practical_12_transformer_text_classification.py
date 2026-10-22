"""
Practical 12: Transformer-Based Text Classification
Compares a Self-Attention Transformer block vs an LSTM for 4-class text classification.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)


class TransformerBlock(layers.Layer):
    """Custom Transformer Encoder Layer with Multi-Head Self-Attention."""
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


def experiment_12_transformer():
    print("=" * 65)
    print("PRACTICAL 12: TRANSFORMER-BASED TEXT CLASSIFICATION")
    print("Comparing Transformer (Self-Attention) vs LSTM on 4 Classes")
    print("=" * 65)

    vocab_size = 1000
    max_len = 50
    embed_dim = 32
    num_classes = 4

    # Synthetic text data for 4-class news categorization
    x_train = np.random.randint(1, vocab_size, (500, max_len))
    y_train = np.random.randint(0, num_classes, (500,))
    x_test = np.random.randint(1, vocab_size, (100, max_len))
    y_test = np.random.randint(0, num_classes, (100,))

    # 1. Build Transformer Model
    inputs = keras.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim=embed_dim, num_heads=2, ff_dim=64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    transformer_model = keras.Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # 2. Build LSTM Model
    lstm_model = keras.Sequential([
        keras.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, embed_dim),
        layers.LSTM(32),
        layers.Dense(num_classes, activation="softmax"),
    ])
    lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train Transformer
    print("\n1. Training Transformer Model...")
    t0 = time.time()
    transformer_model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    t_time = time.time() - t0
    t_loss, t_acc = transformer_model.evaluate(x_test, y_test, verbose=0)

    # Train LSTM
    print("2. Training LSTM Model...")
    t0 = time.time()
    lstm_model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    l_time = time.time() - t0
    l_loss, l_acc = lstm_model.evaluate(x_test, y_test, verbose=0)

    print("\n" + "=" * 65)
    print("PERFORMANCE COMPARISON:")
    print("-" * 65)
    print(f"{'Model':<15} | {'Test Accuracy':<15} | {'Training Time'}")
    print("-" * 65)
    print(f"{'Transformer':<15} | {t_acc * 100:>13.2f}% | {t_time:.2f}s")
    print(f"{'LSTM':<15} | {l_acc * 100:>13.2f}% | {l_time:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    experiment_12_transformer()
