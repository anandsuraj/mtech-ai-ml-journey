# Dockerfile for Hybrid RAG System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize threading for FAISS/OpenBLAS
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV TOKENIZERS_PARALLELISM=false

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Copy project files
COPY . .

# Create necessary directories and ensure they are writable
RUN mkdir -p data reports .cache/huggingface && chmod -R 777 data reports .cache

# Expose port for Flask UI
EXPOSE 5000

# Default command runs the UI
CMD ["python", "ui/app.py"]
