"""
Practical 19: Object Detection Using Transfer Learning
Locates and labels objects with bounding boxes; evaluates with Intersection over Union (IoU).
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


def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) of two bounding boxes [y1, x1, y2, x2]."""
    y1_A, x1_A, y2_A, x2_A = min(boxA[0], boxA[2]), min(boxA[1], boxA[3]), max(boxA[0], boxA[2]), max(boxA[1], boxA[3])
    y1_B, x1_B, y2_B, x2_B = min(boxB[0], boxB[2]), min(boxB[1], boxB[3]), max(boxB[0], boxB[2]), max(boxB[1], boxB[3])

    inter_y1 = max(y1_A, y1_B)
    inter_x1 = max(x1_A, x1_B)
    inter_y2 = min(y2_A, y2_B)
    inter_x2 = min(x2_A, x2_B)

    inter_area = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)
    area_A = (y2_A - y1_A) * (x2_A - x1_A)
    area_B = (y2_B - y1_B) * (x2_B - x1_B)

    iou = inter_area / float(area_A + area_B - inter_area + 1e-6)
    return float(np.clip(iou, 0.0, 1.0))


def experiment_19_object_detection():
    print("=" * 65)
    print("PRACTICAL 19: OBJECT DETECTION & BOUNDING BOX REGRESSION")
    print("Transfer Learning Backbone + Classification & BBox Regression Heads")
    print("=" * 65)

    num_samples = 300
    img_shape = (64, 64, 3)

    # Synthetic images with targets: class label (3 classes) + bbox [ymin, xmin, ymax, xmax]
    x_images = np.random.uniform(0.1, 0.25, (num_samples, *img_shape)).astype("float32")
    y_classes = np.random.randint(0, 3, (num_samples,))
    y_boxes = np.zeros((num_samples, 4), dtype="float32")
    for i in range(num_samples):
        r1, c1 = np.random.randint(5, 25, 2)
        h, w = np.random.randint(18, 28, 2)
        r2, c2 = r1 + h, c1 + w
        x_images[i, r1:r2, c1:c2] = np.random.uniform(0.75, 1.0, (h, w, 3))
        y_boxes[i] = [r1 / 64.0, c1 / 64.0, r2 / 64.0, c2 / 64.0]

    # Feature Extractor Backbone (MobileNetV2)
    inputs = keras.Input(shape=img_shape)
    base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights=None)
    x = base_model(inputs)

    # Dual Output Heads
    # 1. Bounding Box Regressor (Preserves spatial features via Flatten)
    bbox_feat = layers.Flatten()(x)
    bbox_feat = layers.Dense(64, activation="relu")(bbox_feat)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_head")(bbox_feat)

    # 2. Class Label Predictor (3 classes)
    cls_feat = layers.GlobalAveragePooling2D()(x)
    class_output = layers.Dense(3, activation="softmax", name="class_head")(cls_feat)

    model = keras.Model(inputs=inputs, outputs=[class_output, bbox_output])
    model.compile(
        optimizer=keras.optimizers.Adam(0.003),
        loss={"class_head": "sparse_categorical_crossentropy", "bbox_head": "mse"},
        loss_weights={"class_head": 1.0, "bbox_head": 25.0},
        metrics={"class_head": "accuracy", "bbox_head": "mae"},
    )
    model.summary()

    print("\nTraining Object Detector (BBox Regression + Classification)...")
    model.fit(
        x_images,
        {"class_head": y_classes, "bbox_head": y_boxes},
        epochs=4,
        batch_size=32,
        verbose=1,
    )

    # Evaluate IoU on test samples
    pred_classes, pred_boxes = model.predict(x_images[:10], verbose=0)
    ious = [calculate_iou(y_boxes[i], pred_boxes[i]) for i in range(10)]
    mean_iou = np.mean(ious)

    print("\n" + "=" * 65)
    print("DETECTION & LOCALIZATION EVALUATION:")
    print("-" * 65)
    print(f"Mean Intersection over Union (mIoU): {mean_iou * 100:.2f}%\n")
    print("Sample Prediction (Test Image 0):")
    print(f"Ground Truth BBox:  {[round(v, 2) for v in y_boxes[0]]}")
    print(f"Predicted BBox:     {[round(v, 2) for v in pred_boxes[0]]}")
    print(f"Sample IoU Score:   {ious[0] * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    experiment_19_object_detection()
