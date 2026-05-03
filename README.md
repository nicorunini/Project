# Eye State Streamlit App

This project can train and test a three-class eye-state model:

- Open
- Partially Closed
- Closed

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

The trainer reads images from `datasets/OPEN`, `datasets/PARTIALLY CLOSED`, and `datasets/CLOSED`.
By default, images are loaded as RGB, resized to `128x128`, and rescaled from `0-255` to `0-1`.
The class order uses alphabetical folder order: `Closed`, `Open`, `Partially Closed`.
It saves:

- `cnn_classifier.h5`
- `labels.txt`

At the end of training, the script prints a confusion matrix so you can see which labels still need cleanup.

## Run Streamlit

```bash
streamlit run app.py
```

The default model is `cnn_classifier.h5`, with class labels from `labels.txt`.
The app supports image upload, camera snapshot, and live webcam mode. If you use a different saved model, set its path in the sidebar.
The app automatically uses grayscale preprocessing for grayscale models and RGB preprocessing for older RGB models.

## Windows shortcuts

You can also run these files from the project folder:

- `run_accuracy.bat` checks model accuracy in the terminal
- `run_streamlit.bat` opens the Streamlit app
- `train_original.bat` trains the original CNN and shows epochs
