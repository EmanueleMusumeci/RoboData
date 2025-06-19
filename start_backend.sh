#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")"

# Check if virtual environment exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating and setting up virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # Activate existing virtual environment
    source .venv/bin/activate
fi

# Install required packages if not already installed
REQUIRED_PACKAGES=("uvicorn[standard]" "fastapi" "python-multipart" "python-jose[cryptography]" "passlib[bcrypt]")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show ${pkg%%[<>=]*} &> /dev/null; then
        echo "Installing $pkg..."
        pip install "$pkg"
    fi
done

# Set PYTHONPATH to include the project root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Start the FastAPI server
echo "Starting backend server..."
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir=.
