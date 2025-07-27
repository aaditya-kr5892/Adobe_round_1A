FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt


# Copy all source files
COPY src/*.py ./
COPY config/config.yaml ./config/
COPY main.py ./

# Copy your trained models
COPY data/models/ ./models/

# Set the entry point
CMD ["python", "main.py"]
