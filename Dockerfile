FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH="/app"
# Install system dependencies: redis-server and supervisor
RUN apt-get update && apt-get install -y --no-install-recommends redis-server supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gradio_app.py .
COPY agent/ ./agent/

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the new startup script and make it executable
COPY start_frontend.sh /app/
RUN chmod +x /app/start_frontend.sh

# DEBUG: List contents of /app and /app/agent immediately after COPY
RUN echo "Listing /app immediately after COPY agent/ :" && ls -lha /app
RUN echo "Listing /app/agent immediately after COPY agent/ :" && ls -lha /app/agent
# ... other COPY commands ...
# Expose the Gradio port
EXPOSE 7860

# Run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
#CMD ["python", "-c", "import sys; print(sys.path); import agent; print('Successfully imported agent')"]
# CMD ["python", "debug_script.py"]
