# Optimized Dockerfile for the ETA Challenge
# Target total image size: ≤ 2.5 GB.
# This image supports training, testing, and grading.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libgomp1: required for XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# We install the CPU version of torch to keep the image size well under the 2.5GB limit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy core logic and model
# Using the new root-level structure
COPY predict.py grade.py train.py model.pkl ./
COPY tests/ ./tests/

# Default entrypoint is for grading (as required by the submission spec)
# To run training: docker run --rm -v $(pwd)/data:/app/data <image_name> train.py
# To run tests:    docker run --rm <image_name> -m pytest tests/
ENTRYPOINT ["python"]
CMD ["grade.py"]
