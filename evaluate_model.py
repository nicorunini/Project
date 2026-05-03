import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


DATASET_DIR = Path("datasets")
CLASS_DIRS = ["CLOSED", "OPEN", "PARTIALLY CLOSED"]
DISPLAY_LABELS = ["Closed", "Open", "Partially Closed"]
IMAGE_SIZE = 128


def expected_channels(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    channels = input_shape[-1]
    return int(channels) if channels in (1, 3) else 3


def model_has_rescaling(model):
    return any(layer.__class__.__name__ == "Rescaling" for layer in model.layers)


def load_dataset(image_size: int, batch_size: int, color_mode: str):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_DIRS,
        shuffle=False,
        color_mode=color_mode,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )


def maybe_normalize(dataset, normalize_input: bool):
    if not normalize_input:
        return dataset
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def print_confusion_matrix(actual, predicted):
    matrix = tf.math.confusion_matrix(
        actual,
        predicted,
        num_classes=len(DISPLAY_LABELS),
    ).numpy()

    print("\nConfusion matrix")
    print("Rows = actual, columns = predicted")
    print(" " * 20 + " ".join(f"{label[:8]:>8}" for label in DISPLAY_LABELS))
    for label, row in zip(DISPLAY_LABELS, matrix):
        print(f"{label[:18]:<18}  " + " ".join(f"{value:8d}" for value in row))

    print("\nPer-class metrics")
    for index, label in enumerate(DISPLAY_LABELS):
        true_positive = matrix[index, index]
        actual_total = np.sum(matrix[index, :])
        predicted_total = np.sum(matrix[:, index])
        recall = true_positive / actual_total if actual_total else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        print(f"{label}: precision {precision:.2%}, recall {recall:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved eye-state model.")
    parser.add_argument("model_path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--normalize",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use auto unless you know whether the model already rescales inputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    channels = expected_channels(model)
    color_mode = "grayscale" if channels == 1 else "rgb"

    if args.normalize == "auto":
        normalize_input = not model_has_rescaling(model)
    else:
        normalize_input = args.normalize == "on"

    raw_ds = load_dataset(args.image_size, args.batch_size, color_mode)
    eval_ds = maybe_normalize(raw_ds, normalize_input).prefetch(tf.data.AUTOTUNE)

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    loss, accuracy = model.evaluate(eval_ds, verbose=0)
    probabilities = model.predict(eval_ds, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in raw_ds])

    print(f"Model: {args.model_path}")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Class order: {', '.join(DISPLAY_LABELS)}")
    print(f"Input: {args.image_size}x{args.image_size} {color_mode}")
    print(f"Input normalization: {'on' if normalize_input else 'off'}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print_confusion_matrix(actual, predicted)


if __name__ == "__main__":
    main()
