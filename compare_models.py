import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from evaluate_model import (
    CLASS_DIRS,
    DATASET_DIR,
    DISPLAY_LABELS,
    IMAGE_SIZE,
    expected_channels,
    load_dataset,
    maybe_normalize,
    model_has_rescaling,
)


DEFAULT_BASELINE = "cnn_classifier.h5"
MODEL_PATTERNS = ("*.keras", "*.h5")


def discover_models():
    models = []
    for pattern in MODEL_PATTERNS:
        models.extend(
            path
            for path in Path(".").glob(pattern)
            if ".broken-" not in path.name.lower()
        )
    return sorted(set(models), key=lambda path: path.name.lower())


def resolve_model_paths(args):
    if args.models:
        return [Path(model_path) for model_path in args.models]

    discovered = discover_models()
    baseline = Path(args.baseline)
    if baseline.exists() and baseline not in discovered:
        discovered.insert(0, baseline)
    return discovered


def normalize_for_model(model, normalize_setting):
    if normalize_setting == "auto":
        return not model_has_rescaling(model)
    return normalize_setting == "on"


def class_metrics(actual, predicted, class_count):
    matrix = tf.math.confusion_matrix(
        actual,
        predicted,
        num_classes=class_count,
    ).numpy()

    precisions = []
    recalls = []
    for index in range(class_count):
        true_positive = matrix[index, index]
        actual_total = np.sum(matrix[index, :])
        predicted_total = np.sum(matrix[:, index])
        recalls.append(true_positive / actual_total if actual_total else 0.0)
        precisions.append(true_positive / predicted_total if predicted_total else 0.0)

    return matrix, precisions, recalls


def benchmark_model(model, dataset, warmup_batches, benchmark_batches):
    batches = []
    for images, _ in dataset.take(warmup_batches + benchmark_batches):
        batches.append(images)

    if not batches:
        return 0.0, 0.0, 0

    warmup = batches[:warmup_batches]
    measured = batches[warmup_batches:] or batches

    for images in warmup:
        model.predict(images, verbose=0)

    total_images = sum(int(images.shape[0]) for images in measured)
    start = time.perf_counter()
    for images in measured:
        model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start

    images_per_second = total_images / elapsed if elapsed else 0.0
    latency_ms = (elapsed / total_images * 1000.0) if total_images else 0.0
    return latency_ms, images_per_second, total_images


def evaluate_one_model(model_path, args):
    model = tf.keras.models.load_model(model_path, compile=False)
    channels = expected_channels(model)
    color_mode = "grayscale" if channels == 1 else "rgb"
    normalize_input = normalize_for_model(model, args.normalize)

    raw_ds = load_dataset(args.image_size, args.batch_size, color_mode)
    eval_ds = maybe_normalize(raw_ds, normalize_input).prefetch(tf.data.AUTOTUNE)

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    loss, accuracy = model.evaluate(eval_ds, verbose=0)
    probabilities = model.predict(eval_ds, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in raw_ds]
    )

    matrix, precisions, recalls = class_metrics(
        actual,
        predicted,
        len(DISPLAY_LABELS),
    )
    latency_ms, images_per_second, benchmark_images = benchmark_model(
        model,
        eval_ds,
        args.warmup_batches,
        args.benchmark_batches,
    )

    return {
        "path": Path(model_path),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "latency_ms": latency_ms,
        "images_per_second": images_per_second,
        "benchmark_images": benchmark_images,
        "params": int(model.count_params()),
        "size_mb": Path(model_path).stat().st_size / (1024 * 1024),
        "image_size": args.image_size,
        "color_mode": color_mode,
        "normalize_input": normalize_input,
        "matrix": matrix,
        "precisions": precisions,
        "recalls": recalls,
    }


def print_summary(results, baseline_name):
    baseline = next(
        (result for result in results if result["path"].name == Path(baseline_name).name),
        results[0] if results else None,
    )

    print("\nModel comparison")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Class order: {', '.join(DISPLAY_LABELS)}")
    print(
        "Model".ljust(32)
        + "Acc".rjust(9)
        + "Delta".rjust(9)
        + "Loss".rjust(9)
        + "Macro P".rjust(10)
        + "Macro R".rjust(10)
        + "ms/img".rjust(10)
        + "img/s".rjust(10)
        + "Params".rjust(12)
        + "MB".rjust(8)
    )
    print("-" * 120)

    for result in sorted(results, key=lambda item: item["accuracy"], reverse=True):
        delta = result["accuracy"] - baseline["accuracy"] if baseline else 0.0
        print(
            result["path"].name[:31].ljust(32)
            + f"{result['accuracy']:.2%}".rjust(9)
            + f"{delta:+.2%}".rjust(9)
            + f"{result['loss']:.4f}".rjust(9)
            + f"{result['macro_precision']:.2%}".rjust(10)
            + f"{result['macro_recall']:.2%}".rjust(10)
            + f"{result['latency_ms']:.2f}".rjust(10)
            + f"{result['images_per_second']:.1f}".rjust(10)
            + f"{result['params']:,}".rjust(12)
            + f"{result['size_mb']:.1f}".rjust(8)
        )

    if baseline:
        print(f"\nBaseline: {baseline['path'].name}")


def print_details(results):
    for result in results:
        print(f"\n{result['path'].name}")
        print(
            "Input: "
            f"{result['image_size']}x{result['image_size']} {result['color_mode']}, "
            f"normalization {'on' if result['normalize_input'] else 'off'}"
        )
        print(
            f"Benchmark images: {result['benchmark_images']} "
            f"({result['latency_ms']:.2f} ms/image)"
        )

        print("Confusion matrix")
        print("Rows = actual, columns = predicted")
        print(" " * 20 + " ".join(f"{label[:8]:>8}" for label in DISPLAY_LABELS))
        for label, row in zip(DISPLAY_LABELS, result["matrix"]):
            print(f"{label[:18]:<18}  " + " ".join(f"{value:8d}" for value in row))

        print("Per-class metrics")
        for label, precision, recall in zip(
            DISPLAY_LABELS,
            result["precisions"],
            result["recalls"],
        ):
            print(f"{label}: precision {precision:.2%}, recall {recall:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved eye-state models for baseline metrics and speed."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Models to compare. Defaults to every .keras and .h5 model in this folder.",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--benchmark-batches", type=int, default=10)
    parser.add_argument(
        "--normalize",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use auto unless you know whether the model already rescales inputs.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print confusion matrices and per-class metrics for every model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_paths = resolve_model_paths(args)
    if not model_paths:
        print("No .keras or .h5 models found.")
        return

    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        print("Missing model file(s): " + ", ".join(missing))
        raise SystemExit(1)

    print(f"Comparing {len(model_paths)} model(s) against {len(CLASS_DIRS)} classes.")
    results = []
    for model_path in model_paths:
        print(f"Evaluating {model_path}...")
        try:
            results.append(evaluate_one_model(model_path, args))
        except Exception as exc:
            print(f"Skipped {model_path}: {exc}")

    if not results:
        raise SystemExit("No models could be evaluated.")

    print_summary(results, args.baseline)
    if args.details:
        print_details(results)


if __name__ == "__main__":
    main()
