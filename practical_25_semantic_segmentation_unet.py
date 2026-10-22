"""
Practical 25: Semantic Segmentation Using CNN (U-Net)
Pixel-level semantic segmentation with encoder-decoder and skip connections; evaluates with Mean IoU.
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


def build_simple_unet(input_shape=(64, 64, 3), num_classes=2):
    """Simple U-Net Architecture with Encoder, Bottleneck, and Decoder with Skip Connections."""
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

    # Decoder with Skip Connections
    u1 = layers.UpSampling2D((2, 2))(b)
    concat1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat1)

    u2 = layers.UpSampling2D((2, 2))(c3)
    concat2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat2)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c4)
    return keras.Model(inputs=inputs, outputs=outputs, name="Simple_UNet")


def experiment_25_semantic_segmentation():
    print("=" * 65)
    print("PRACTICAL 25: SEMANTIC SEGMENTATION (U-NET)")
    print("Pixel-Level Mask Classification & Mean Intersection over Union (IoU)")
    print("=" * 65)

    img_size = 64
    num_classes = 2  # Background (0) and Foreground/Pet (1)
    num_samples = 200

    # Synthetic images and ground truth masks
    x_images = np.zeros((num_samples, img_size, img_size, 3), dtype="float32")
    y_masks = np.zeros((num_samples, img_size, img_size), dtype="int32")
    center = img_size // 2
    for i in range(num_samples):
        y, x = np.ogrid[:img_size, :img_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        radius = np.random.randint(14, 22)
        mask = dist <= radius
        y_masks[i, mask] = 1
        # Foreground object is bright colored, background is dark with noise
        x_images[i] = np.random.uniform(0.1, 0.3, (img_size, img_size, 3))
        x_images[i, mask] = np.random.uniform(0.7, 1.0, (mask.sum(), 3))

    model = build_simple_unet(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\nTraining U-Net Semantic Segmentation Model...")
    model.fit(x_images, y_masks, epochs=4, batch_size=16, validation_split=0.1, verbose=1)

    # Predict segmentation masks on test images
    pred_probs = model.predict(x_images[:10], verbose=0)
    pred_masks = np.argmax(pred_probs, axis=-1)

    # Calculate Mean IoU
    ious = []
    for i in range(10):
        intersection = np.logical_and(y_masks[i] == 1, pred_masks[i] == 1).sum()
        union = np.logical_or(y_masks[i] == 1, pred_masks[i] == 1).sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou)

    mean_iou = np.mean(ious)

    print("\n" + "=" * 65)
    print("SEGMENTATION EVALUATION RESULTS:")
    print("-" * 65)
    print(f"Mean Intersection over Union (mIoU): {mean_iou * 100:.2f}%")
    print(f"Foreground Pixels (Ground Truth):  {(y_masks[0] == 1).sum()}")
    print(f"Foreground Pixels (Predicted Mask): {(pred_masks[0] == 1).sum()}")
    print("=" * 65)


if __name__ == "__main__":
    experiment_25_semantic_segmentation()
