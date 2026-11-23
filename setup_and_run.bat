@echo off
REM Windows: creates the virtual environment (first run only), installs packages, starts the app.
cd /d "%~dp0"
set ENV_NAME=electricity_forecaster_env
if not exist "%ENV_NAME%\Scripts\activate.bat" (
    echo Creating virtual environment: %ENV_NAME%
    python -m venv %ENV_NAME%
    if errorlevel 1 (
        echo Python not found. Install Python 3.10+ from python.org and tick "Add Python to PATH".
        pause
        exit /b 1
    )
    call %ENV_NAME%\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call %ENV_NAME%\Scripts\activate.bat
)
echo Starting app at http://localhost:8501  (press Ctrl+C to stop)
streamlit run app.py
pause
