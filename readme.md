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
   docker-compose up -d
   ```

4. Access the API at http://localhost:8000/api/py/docs

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

## Running with Docker

### Prerequisites
- Docker
- Docker Compose

### Running the App with Docker Compose

The easiest way to run the app is using Docker Compose, which will start both the Gradio frontend and a mock backend for testing:

```bash
docker compose up
```

The Gradio app will be available at http://localhost:7860.

### Building and Running Just the Gradio App

If you want to run just the Gradio app without the mock backend:

```bash
docker build -t ai-trip-planner-frontend .
docker run -p 7860:7860 -e BACKEND_URL=http://your-backend-url ai-trip-planner-frontend
```

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
