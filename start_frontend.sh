#!/bin/bash

# This script reads the PORT environment variable set by Cloud Run
# and passes it as an argument to the Gradio application.

# Cloud Run injects the port your application should listen on
# via the PORT environment variable.
CLOUD_RUN_PORT=${PORT:-7860} # Default to 7860 if PORT is not set (e.g., local testing)

echo "Starting Gradio frontend on port: $CLOUD_RUN_PORT"

# Execute the Gradio application, passing the Cloud Run port
# Use exec to replace the script process with the python process
exec python gradio_app.py --server-name 0.0.0.0 --server-port $CLOUD_RUN_PORT

