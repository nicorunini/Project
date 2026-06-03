@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" compare_models.py --details
pause
