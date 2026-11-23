#!/usr/bin/env bash
# macOS / Linux: creates the virtual environment (first run only), installs packages, starts the app.
set -e
cd "$(dirname "$0")"
ENV_NAME="electricity_forecaster_env"
PY=$(command -v python3 || command -v python)
if [ ! -d "$ENV_NAME" ]; then
  echo "Creating virtual environment: $ENV_NAME"
  "$PY" -m venv "$ENV_NAME"
  source "$ENV_NAME/bin/activate"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$ENV_NAME/bin/activate"
fi
echo "Starting app at http://localhost:8501  (press Ctrl+C to stop)"
streamlit run app.py
