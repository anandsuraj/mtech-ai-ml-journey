# =========================
# Base Image
# =========================
FROM python:3.10-slim

# =========================
# Environment Variables
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# Working Directory
# =========================
WORKDIR /app

# =========================
# System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python Dependencies
# =========================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Application Code
# =========================
COPY src/ src/
COPY models/ models/
COPY configs/ configs/
COPY . .
# =========================
# Expose Port
# =========================
EXPOSE 8000

# =========================
# Start Inference API
# =========================
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]