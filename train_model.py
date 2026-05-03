import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = Path("datasets")
DISPLAY_LABELS = ["Closed", "Open", "Partially Closed"]
IMAGE_SIZE = 128


def build_model(image_size: int, class_count: int):
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(image_size, image_size, 3),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(class_count, activation="softmax"))
    return model


def create_generators(args):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.validation_split,
    )

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )
    validation_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )
    return train_data, validation_data


def display_labels_from_generator(generator):
    ordered = sorted(generator.class_indices.items(), key=lambda item: item[1])
    return [label.replace("_", " ").title() for label, _ in ordered]


def save_labels(path: Path, labels):
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def print_confusion_matrix(model, validation_data, labels):
    validation_data.reset()
    probabilities = model.predict(validation_data, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = validation_data.classes
    matrix = tf.math.confusion_matrix(
        actual,
        predicted,
        num_classes=len(labels),
    ).numpy()

    print("\nConfusion matrix")
    print("Rows = actual, columns = predicted")
    print(" " * 20 + " ".join(f"{label[:8]:>8}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label[:18]:<18}  " + " ".join(f"{value:8d}" for value in row))

    print("\nPer-class metrics")
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        actual_total = np.sum(matrix[index, :])
        predicted_total = np.sum(matrix[:, index])
        recall = true_positive / actual_total if actual_total else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        print(f"{label}: precision {precision:.2%}, recall {recall:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an eye-state CNN classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="cnn_classifier.h5")
    parser.add_argument("--labels-path", default="labels.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    train_data, validation_data = create_generators(args)
    labels = display_labels_from_generator(train_data)

    model = build_model(args.image_size, train_data.num_classes)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=validation_data,
    )

    loss, accuracy = model.evaluate(validation_data, verbose=0)
    model.save(args.model_path)
    save_labels(Path(args.labels_path), labels)

    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print(f"Saved model to {args.model_path}")
    print(f"Saved labels to {args.labels_path}")
    print(f"Final validation loss: {loss:.4f}")
    print(f"Final validation accuracy: {accuracy:.2%}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print("Class order:", ", ".join(labels))
    print(f"Preprocessing: RGB, resized to {args.image_size}x{args.image_size}, rescaled to 0-1")
    print_confusion_matrix(model, validation_data, labels)


if __name__ == "__main__":
    main()
