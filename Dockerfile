FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gradio_app.py .
COPY agent/ ./agent/

# Ports will be exposed by docker-compose
# CMD will be set by docker-compose 