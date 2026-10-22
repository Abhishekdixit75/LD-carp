"""
Practical 21: Named Entity Recognition (NER) Using Bidirectional LSTM
Sequence labeling for entity recognition (Person, Org, Location); reports Precision, Recall, and F1.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)


def experiment_21_ner():
    print("=" * 65)
    print("PRACTICAL 21: NAMED ENTITY RECOGNITION (NER)")
    print("Bidirectional LSTM for Token Classification & BIO Tagging")
    print("=" * 65)

    tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    tag2idx = {t: i for i, t in enumerate(tags)}
    num_tags = len(tags)
    vocab_size = 500
    max_len = 12

    # Synthetic sentences and corresponding token tags
    num_samples = 400
    x_data = np.random.randint(1, vocab_size, (num_samples, max_len))
    # Synthetic BIO tags (mostly 'O' with sparse entity tags)
    y_data = np.random.choice(num_tags, size=(num_samples, max_len), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    split = int(num_samples * 0.8)
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    # Build Bidirectional LSTM Sequence Labeling Model
    model = keras.Sequential([
        keras.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, 32, mask_zero=True),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Dense(num_tags, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("\nTraining BiLSTM Sequence Labeling Model...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    # Evaluate sequence tags
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=-1).ravel()
    y_true = y_test.ravel()

    print("\n" + "=" * 65)
    print("SEQUENCE LABELING EVALUATION REPORT:")
    print("-" * 65)
    print(classification_report(y_true, predictions, target_names=tags, zero_division=0))
    print("=" * 65)


if __name__ == "__main__":
    experiment_21_ner()
