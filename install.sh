#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to print error messages
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# 1. Check for existing Ollama installation
if ! command -v ollama &> /dev/null
then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh || error_exit "Ollama installation failed."
    echo "Ollama installed successfully."
else
    echo "Ollama is already installed."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "pip is not installed. Please install it using your system's package manager."
    echo "For example, on Debian/Ubuntu, you can use: sudo apt install python3-pip"
    error_exit "pip not found."
fi

# Try pip3 first, then pip
PIP_COMMAND=""
if command -v pip3 &> /dev/null; then
    PIP_COMMAND="pip3"
elif command -v pip &> /dev/null; then
    PIP_COMMAND="pip"
else
    # This case should ideally be caught by the check above, but as a fallback:
    echo "pip is not installed. Please install it using your system's package manager."
    echo "For example, on Debian/Ubuntu, you can use: sudo apt install python3-pip"
    error_exit "pip not found."
fi

if [ -f "requirements.txt" ]; then
    $PIP_COMMAND install -r requirements.txt || error_exit "Failed to install Python dependencies from requirements.txt."
    echo "Python dependencies installed successfully."
else
    echo "requirements.txt not found. Skipping Python dependency installation."
fi

# 3. Set Environment Variables
echo "Setting environment variables in .env file..."
ENV_FILE=".env"
OLLAMA_API_URL="OLLAMA_API_URL=http://localhost:11434"
MODEL_NAME="${MODEL_NAME:-llama3.1:8b}"
EMBEDDINGS_NAME="${EMBEDDINGS_NAME:-nomic-embed-text:latest}"
MODEL="MODEL=$MODEL_NAME"
EMBEDDINGS_MODEL="EMBEDDINGS_MODEL=$EMBEDDINGS_NAME"

# Create .env file if it doesn't exist
touch $ENV_FILE

# Update or add OLLAMA_API_URL
if grep -q "^OLLAMA_API_URL=" "$ENV_FILE"; then
    sed -i "s|^OLLAMA_API_URL=.*|$OLLAMA_API_URL|" "$ENV_FILE"
else
    echo "$OLLAMA_API_URL" >> "$ENV_FILE"
fi

# Update or add MODEL
if grep -q "^MODEL=" "$ENV_FILE"; then
    sed -i "s|^MODEL=.*|$MODEL|" "$ENV_FILE"
else
    echo "$MODEL" >> "$ENV_FILE"
fi
# Update or add EMBEDDINGS_MODEL
if grep -q "^EMBEDDINGS_MODEL=" "$ENV_FILE"; then
    sed -i "s|^EMBEDDINGS_MODEL=.*|$EMBEDDINGS_MODEL|" "$ENV_FILE"
else
    echo "$EMBEDDINGS_MODEL" >> "$ENV_FILE"
fi
echo ".env file updated successfully."

# 4. Pull the Ollama models (chat model + embedding model — both are required)
if command -v ollama &> /dev/null; then
    echo "Pulling Ollama model: $MODEL_NAME..."
    ollama pull "$MODEL_NAME" || error_exit "Failed to pull Ollama model $MODEL_NAME."
    echo "Pulling embedding model: $EMBEDDINGS_NAME..."
    ollama pull "$EMBEDDINGS_NAME" || error_exit "Failed to pull embedding model $EMBEDDINGS_NAME."
    echo "Ollama models pulled successfully."
else
    error_exit "Ollama command not found. Cannot pull model."
fi

# 5. Provide Execution Instructions
echo ""
echo "Installation and setup complete!"
echo "To activate the environment variables, run:"
echo "  source .env"
echo ""
echo "Then, you can run the application, for example:"
echo "  streamlit run app.py"
echo ""

# 6. Make the script executable (This step will be done by the user or a separate command)
# However, we can inform the user.
echo "Note: If you haven't already, make this script executable by running:"
echo "  chmod +x install.sh"

exit 0
