import os
import threading
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="Eye State Model Tester", layout="centered")


DEFAULT_MODEL_PATH = "cnn_classifier.h5"
DEFAULT_LABELS_PATH = "labels.txt"
FALLBACK_LABELS = ["Closed", "Open", "Partially Closed"]
IMAGE_SIZE = (128, 128)
MODEL_LOCK = threading.Lock()


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def parse_labels(raw_labels: str):
    labels = [label.strip() for label in raw_labels.splitlines() if label.strip()]
    return labels or FALLBACK_LABELS


def load_labels(labels_path: str):
    path = Path(labels_path)
    if path.exists():
        return parse_labels(path.read_text(encoding="utf-8"))
    return FALLBACK_LABELS


def suggested_labels_path(model_path: str):
    model_stem = Path(model_path).stem
    model_labels_path = Path(f"{model_stem}_labels.txt")

    if model_labels_path.exists():
        return str(model_labels_path)
    return DEFAULT_LABELS_PATH


def model_choices():
    search_dirs = [Path("."), Path.cwd().parent]
    choices = sorted({
        str(path)
        for search_dir in search_dirs
        for pattern in ("*.keras", "*.h5")
        for path in search_dir.glob(pattern)
    })
    if DEFAULT_MODEL_PATH in choices:
        choices.remove(DEFAULT_MODEL_PATH)
    choices.insert(0, DEFAULT_MODEL_PATH)
    return choices


def expected_channels(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    channels = input_shape[-1]
    return int(channels) if channels in (1, 3) else 3


def model_has_rescaling(model):
    return any(layer.__class__.__name__ == "Rescaling" for layer in model.layers)


def prepare_image(image: Image.Image, normalize_input: bool, channels: int):
    image_mode = "L" if channels == 1 else "RGB"
    image = image.convert(image_mode).resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    if channels == 1:
        array = np.expand_dims(array, axis=-1)
    if normalize_input:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def as_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if scores.size == 1:
        positive = float(np.clip(scores[0], 0.0, 1.0))
        return np.array([1.0 - positive, positive], dtype=np.float32)

    if np.any(scores < 0.0) or not np.isclose(float(np.sum(scores)), 1.0, atol=0.05):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)

    return np.clip(scores, 0.0, 1.0)


def predict(model, image: Image.Image, normalize_input: bool):
    prepared = prepare_image(image, normalize_input, expected_channels(model))
    with MODEL_LOCK:
        prediction = model.predict(prepared, verbose=0)
    return as_probabilities(prediction)


def prediction_table(labels, scores):
    best_index = int(np.argmax(scores))
    best_label = labels[best_index]
    best_score = float(scores[best_index])

    st.subheader(f"Prediction: {best_label}")
    st.metric("Confidence", f"{best_score:.2%}")

    st.write("Scores")
    for label, score in zip(labels, scores):
        score = float(score)
        st.progress(score)
        st.write(f"{label}: {score:.2%}")


def draw_prediction(image: Image.Image, label: str, score: float):
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    text = f"{label}  {score:.1%}"
    padding = 8
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + (padding * 2)
    height = box[3] - box[1] + (padding * 2)
    draw.rectangle((8, 8, 8 + width, 8 + height), fill=(15, 23, 42))
    draw.text((8 + padding, 8 + padding), text, fill=(255, 255, 255), font=font)
    return frame


def run_live_camera(model, labels, normalize_input: bool):
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("Live webcam mode needs `streamlit-webrtc`.")
        st.code("pip install streamlit-webrtc", language="bash")
        return

    class EyeStateVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            image_array = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(image_array)
            scores = predict(model, image, normalize_input)
            frame_labels = labels
            if len(frame_labels) != len(scores):
                frame_labels = [f"Class {index}" for index in range(len(scores))]
            best_index = int(np.argmax(scores))
            annotated = draw_prediction(image, frame_labels[best_index], float(scores[best_index]))
            return av.VideoFrame.from_ndarray(np.asarray(annotated), format="rgb24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(
        key="eye-state-live-camera",
        video_processor_factory=EyeStateVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_configuration,
    )


st.title("Eye State Model Tester")

with st.sidebar:
    st.header("Model")
    choices = model_choices()
    selected_model = st.selectbox("Model file", choices, index=0)
    model_path = st.text_input("Model path", value=selected_model)
    labels_path = st.text_input(
        "Labels file",
        value=suggested_labels_path(model_path),
        key=f"labels_path_{Path(model_path).name}",
    )
    labels = parse_labels(
        st.text_area(
            "Class labels",
            value="\n".join(load_labels(labels_path)),
            height=120,
        )
    )
    normalization_mode = st.selectbox(
        "Input normalization",
        ["Auto", "On", "Off"],
        index=0,
        help="Use Auto unless you know whether the model already has a Rescaling layer.",
    )

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.info("Train a model first with `python train_model.py`, or put your saved model in this folder.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as exc:
    st.error("Streamlit found the model file, but TensorFlow could not load it.")
    st.exception(exc)
    st.stop()

channels = expected_channels(model)
if normalization_mode == "Auto":
    normalize_input = not model_has_rescaling(model)
else:
    normalize_input = normalization_mode == "On"

st.caption(f"Model input: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} with {'grayscale' if channels == 1 else 'RGB'} preprocessing")
st.caption(f"Input normalization: {'on' if normalize_input else 'off'}")

source = st.radio(
    "Image source",
    ["Upload image", "Camera snapshot", "Live webcam"],
    horizontal=True,
)

if source == "Live webcam":
    if len(labels) != 3:
        st.warning("Live webcam mode expects three labels for this eye-state model.")
    run_live_camera(model, labels, normalize_input)
    st.stop()

image = None
if source == "Upload image":
    uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a photo")
    if camera_file:
        image = Image.open(camera_file)

if image is None:
    st.info("Choose an image to get a prediction.")
    st.stop()

st.image(image, caption="Input image", width="stretch")

scores = predict(model, image, normalize_input)
if len(labels) != len(scores):
    labels = [f"Class {index}" for index in range(len(scores))]
    st.warning("The number of labels does not match the model output, so generic labels are shown.")

prediction_table(labels, scores)
