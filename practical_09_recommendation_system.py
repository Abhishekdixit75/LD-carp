"""
Practical 9: Deep Learning-Based Recommendation System
Embedding-based collaborative filtering evaluated with Precision@K and Recall@K.
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


def experiment_9_recommender():
    print("=" * 65)
    print("PRACTICAL 9: DEEP LEARNING RECOMMENDATION SYSTEM")
    print("User-Item Embeddings & Precision@K / Recall@K Evaluation")
    print("=" * 65)

    num_users = 100
    num_items = 50
    embedding_dim = 16
    num_samples = 1000

    # Synthetic user-item interactions
    user_ids = np.random.randint(0, num_users, (num_samples,))
    item_ids = np.random.randint(0, num_items, (num_samples,))
    ratings = np.random.choice([0.0, 1.0], size=(num_samples,), p=[0.7, 0.3])

    # Build Embedding Model
    user_in = keras.Input(shape=(1,), name="user")
    item_in = keras.Input(shape=(1,), name="item")

    user_vec = layers.Flatten()(layers.Embedding(num_users, embedding_dim)(user_in))
    item_vec = layers.Flatten()(layers.Embedding(num_items, embedding_dim)(item_in))

    # Dot product interaction + Dense layer
    dot_product = layers.Dot(axes=1)([user_vec, item_vec])
    dense = layers.Dense(16, activation="relu")(dot_product)
    output = layers.Dense(1, activation="sigmoid")(dense)

    model = keras.Model(inputs=[user_in, item_in], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train model
    print("\nTraining Recommender Model...")
    model.fit([user_ids, item_ids], ratings, epochs=5, batch_size=32, verbose=1)

    # Evaluate Precision@K and Recall@K for a sample user
    target_user = 5
    all_items = np.arange(num_items)
    user_array = np.full(num_items, target_user)
    predictions = model.predict([user_array, all_items], verbose=0).ravel()

    # Ground truth relevant items (where rating was 1)
    ground_truth = set(item_ids[(user_ids == target_user) & (ratings == 1)])
    if not ground_truth:
        ground_truth = {0, 1, 2}  # fallback default relevant items

    k = 5
    top_k_items = set(np.argsort(predictions)[::-1][:k])
    hits = len(top_k_items & ground_truth)
    precision_k = hits / k
    recall_k = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

    print(f"\nEvaluation for User {target_user} (K = {k}):")
    print(f"Top {k} Recommended Items: {list(top_k_items)}")
    print(f"Actual Relevant Items:    {list(ground_truth)}")
    print(f"Precision@{k}:             {precision_k * 100:.2f}%")
    print(f"Recall@{k}:                {recall_k * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    experiment_9_recommender()
