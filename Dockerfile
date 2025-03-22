FROM python:3.10-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variable for Python path
ENV PYTHONPATH=.

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD uvicorn agent.trip_agent_server:app --host 0.0.0.0 --port ${PORT:-8000} 