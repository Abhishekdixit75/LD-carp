"""
Practical 10: Sentiment Analysis on Network Graphs Using RNN
Classifies sentiment of user posts and maps them across a social interaction graph.
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


def experiment_10_network_sentiment():
    print("=" * 65)
    print("PRACTICAL 10: SENTIMENT ANALYSIS ON NETWORK GRAPHS")
    print("RNN Sentiment Classification + User Interaction Network")
    print("=" * 65)

    vocab_size = 500
    max_len = 10

    # 1. Train simple RNN sentiment classifier
    model = keras.Sequential([
        keras.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, 16),
        layers.SimpleRNN(16),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Synthetic text data (positive=1, negative=0)
    x_train = np.random.randint(1, vocab_size, (200, max_len))
    y_train = np.random.randint(0, 2, (200, 1))
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)

    # 2. Simulate User Network with Posts
    # Users: Alice, Bob, Charlie, David, Emma
    users = ["Alice", "Bob", "Charlie", "David", "Emma"]
    interactions = [
        ("Alice", "Bob", "Great job on the project!"),
        ("Bob", "Charlie", "Terrible mistake in the code."),
        ("Charlie", "David", "Loved your presentation!"),
        ("David", "Alice", "Very poor performance today."),
        ("Emma", "Bob", "Awesome support and teamwork!"),
    ]

    print("Analyzing sentiment of user interactions...\n")
    user_sentiments = {u: [] for u in users}

    for sender, receiver, post in interactions:
        # Convert post to dummy token indices
        tokens = np.random.randint(1, vocab_size, (1, max_len))
        sentiment_score = float(model.predict(tokens, verbose=0)[0][0])
        sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"
        user_sentiments[sender].append(sentiment_score)
        print(f"[{sender} -> {receiver}] \"{post}\" => Sentiment: {sentiment_label} ({sentiment_score:.2f})")

    # 3. Network Sentiment Aggregation
    print("\n" + "-" * 65)
    print("NETWORK SENTIMENT SUMMARY:")
    print("-" * 65)
    for user, scores in user_sentiments.items():
        avg = np.mean(scores) if scores else 0.5
        overall = "Positive (+)" if avg >= 0.5 else "Negative (-)"
        print(f"User: {user:<10} | Avg Sentiment Score: {avg:.2f} | Status: {overall}")
    print("=" * 65)


if __name__ == "__main__":
    experiment_10_network_sentiment()
