# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# avoid interactive dialogs
ENV DEBIAN_FRONTEND=noninteractive

# create app dir
WORKDIR /workspace

# Copy training code early to leverage docker cache
COPY train/ ./train/

# Install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r train/requirements.txt

# Create data and mlflow mounts
VOLUME ["/workspace/data", "/workspace/mlflow"]

# default command for interactive run (override in docker-compose)
CMD ["bash"]
