# Production-Grade Dockerfile for Free Cloud Deployment (Hugging Face Spaces / Render / Railway)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install system dependencies for OpenCV and Astropy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic

# Copy project files
COPY . .

# Generate sample scenarios & pre-train model checkpoint
RUN python generate_sample_data.py && python train.py

# Expose port
EXPOSE 8000

# Run FastAPI production server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
