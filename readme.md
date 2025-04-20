# Trip Agent API

A demo API service for trip planning powered by OpenAI and Google Places API.

## Deployment Options

### Option 1: Heroku Deployment

#### Prerequisites

- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
- OpenAI API key
- Google Places API key

#### One-Click Deployment

[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

#### Easy Deployment with Script

Run our deployment script:
```
./deploy_to_heroku.sh
```

The script will guide you through the deployment process.

#### Manual Deployment Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/trip_agent.git
   cd trip_agent
   ```

2. Login to Heroku:
   ```
   heroku login
   ```

3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

4. Configure environment variables:
   ```
   heroku config:set OPENAI_API_KEY=your_openai_api_key
   heroku config:set GOOGLE_PLACE_API_KEY=your_google_api_key
   heroku config:set GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   heroku config:set PYTHONPATH=.
   ```

5. Deploy to Heroku:
   ```
   git push heroku main
   ```

6. Open the app:
   ```
   heroku open
   ```

### Option 2: Docker Deployment

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- OpenAI API key
- Google Places API key

#### Easy Deployment with Script

Run our Docker deployment script:
```
./deploy_with_docker.sh
```

The script will guide you through the deployment process.

#### Manual Docker Deployment

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/trip_agent.git
   cd trip_agent
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_PLACE_API_KEY=your_google_api_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   ```

3. Build and run with Docker Compose:
   ```
   docker-compose up --build
   ```

4. Access the API at http://localhost:8001/api/py/docs and the Gradio UI at http://localhost:7860

#### Redis Configuration in Docker

The application uses Redis for session management and caching. When running in Docker:

- Redis runs as a separate service defined in `docker-compose.yml`
- The backend and frontend connect to Redis using the service name (`redis://redis:6379`)
- SSL for Redis is disabled by default using the `REDIS_SSL_ENABLED=false` environment variable

If you need to enable SSL for Redis (for external Redis instances):

1. Set `REDIS_SSL_ENABLED=true` in your environment or docker-compose.yml
2. Configure your Redis server to use SSL
3. Update the Redis URL if needed to point to your SSL-enabled Redis server

For local development with Docker, the default configuration works without any changes.

#### Shutting Down Docker Containers

To properly shut down the Docker containers and free up resources:

1. If you started with `docker-compose up` (attached mode), press `Ctrl+C` in the terminal where it's running

2. If you started with `docker-compose up -d` (detached mode), run:
   ```
   docker-compose down
   ```

3. To completely clean up (removes containers, networks, and volumes):
   ```
   docker-compose down -v
   ```

4. To remove built images as well:
   ```
   docker-compose down -v --rmi all
   ```

#### Troubleshooting Redis Connection Issues

If you encounter Redis connection errors like:
```
TypeError: AbstractConnection.__init__() got an unexpected keyword argument 'ssl_cert_reqs'
```

This is typically related to SSL configuration for Redis. The solution is:

1. Ensure `REDIS_SSL_ENABLED=false` is set in the environment for both backend and frontend services when using the Docker Redis service
2. Check that the Redis URL format is correct for your environment
3. For external Redis instances that require SSL, make sure your Redis server is properly configured for SSL connections

The application is configured to automatically handle Redis connections based on the `REDIS_SSL_ENABLED` environment variable, which defaults to `false`.

## API Endpoints

- `GET /api/py/docs`: API documentation (Swagger UI)
- `GET /api/py/helloFastApi`: Test endpoint
- `POST /api/py/categories`: Get categories for a city
- `POST /api/py/recommendations`: Get attraction recommendations
- `POST /api/py/greedy-itinerary`: Get optimized itinerary using greedy scheduler
- `POST /api/py/itinerary-details-conversation`: Converse about itinerary details

## Local Development

1. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys

4. Run the server:
   ```
   uvicorn agent.trip_agent_server:app --reload
   ```

5. Access the API at http://localhost:8000/api/py/docs

# AI Trip Planner - Gradio Frontend

This repository contains a Gradio-based frontend for the AI Trip Planner application. The frontend provides a user-friendly interface for interacting with the trip planning backend.

## Running Locally

### Prerequisites
- Python 3.10+
- pip

### Setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Gradio app:
```bash
python gradio_app.py
```

The Gradio app will be available at http://127.0.0.1:7860.

## Configuration

The Gradio app can be configured using environment variables or command-line arguments:

- `BACKEND_URL`: URL of the backend API (default: `http://localhost:8001`)
- `--server-name`: Server name/host (default: `127.0.0.1`, use `0.0.0.0` for Docker)
- `--server-port`: Server port (default: `7860`)

## Features

- Interactive trip planning form to collect initial trip details
- Conversational interface for refining trip plans
- Visual display of the generated itinerary
- Feedback mechanism for users to rate responses and the itinerary
