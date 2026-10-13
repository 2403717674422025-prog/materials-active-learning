
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

# Copy the backend package into /app/backend
COPY . /app/backend

# Create directories used by the application
RUN mkdir -p /app/uploads /app/models /app/data/processed /app/data/raw

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
