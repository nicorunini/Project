@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" evaluate_model.py "cnn_classifier.h5"
pause
