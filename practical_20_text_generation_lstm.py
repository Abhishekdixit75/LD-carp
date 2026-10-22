"""
Practical 20: Text Generation Using LSTM
Character-level LSTM language model with temperature-based sampling.
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


def sample_with_temperature(preds, temperature=1.0):
    """Helper function to sample an index from a probability array using temperature."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def experiment_20_text_generation():
    print("=" * 65)
    print("PRACTICAL 20: TEXT GENERATION USING LSTM")
    print("Character-Level LSTM Language Model & Temperature Sampling")
    print("=" * 65)

    corpus = "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune."
    chars = sorted(list(set(corpus)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    seq_length = 15

    # Prepare character sequences
    sentences = []
    next_chars = []
    for i in range(len(corpus) - seq_length):
        sentences.append([char_to_idx[c] for c in corpus[i : i + seq_length]])
        next_chars.append(char_to_idx[corpus[i + seq_length]])

    x_data = np.array(sentences)
    y_data = np.array(next_chars)

    # Build LSTM Language Model
    model = keras.Sequential([
        keras.Input(shape=(seq_length,)),
        layers.Embedding(vocab_size, 32),
        layers.LSTM(64),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.summary()

    print("\nTraining Character-Level LSTM on Text Corpus...")
    model.fit(x_data, y_data, epochs=30, batch_size=16, verbose=0)

    # Generation across different temperatures
    seed_phrase = "To be, or not t"
    print(f"\nSeed Prompt: \"{seed_phrase}\"\n")

    for temp in [0.2, 0.7, 1.2]:
        generated = seed_phrase
        curr_seq = [char_to_idx[c] for c in seed_phrase]

        for _ in range(30):
            x_input = np.array([curr_seq[-seq_length:]])
            preds = model.predict(x_input, verbose=0)[0]
            next_idx = sample_with_temperature(preds, temperature=temp)
            generated += idx_to_char[next_idx]
            curr_seq.append(next_idx)

        print(f"Temperature {temp:.1f}: \"{generated}\"")

    print("\n" + "=" * 65)
    print("Summary: Low temperature (0.2) yields predictable, conservative text;")
    print("High temperature (1.2) yields diverse, creative, but riskier output.")
    print("=" * 65)


if __name__ == "__main__":
    experiment_20_text_generation()
