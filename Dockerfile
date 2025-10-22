# ==========================
# Dockerfile
# ==========================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow \
    tqdm \
    opencv-python \
    torchvision \
    pillow \
    matplotlib \
    fastapi \
    uvicorn \
    numpy

# Optional: make MLflow accessible inside container
ENV MLFLOW_TRACKING_URI=file:/workspace/mlruns

# Expose MLflow port
EXPOSE 5000
