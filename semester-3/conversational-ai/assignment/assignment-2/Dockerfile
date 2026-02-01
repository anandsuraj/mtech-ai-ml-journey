# Dockerfile for Hybrid RAG System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data reports

# Expose port for Flask UI
EXPOSE 5000

# Default command
CMD ["python", "ui/app.py"]
