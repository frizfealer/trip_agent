#!/bin/bash

echo "Trip Agent API - Docker Deployment Script"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install it first:"
    echo "https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Please install it first:"
    echo "https://docs.docker.com/compose/install/"
    exit 1
fi

# Setup environment variables
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Ask for API keys
read -p "Enter your OpenAI API key: " OPENAI_API_KEY
read -p "Enter your Google Places API key: " GOOGLE_PLACES_API_KEY

# Check if keys are already in .env, and update or add them
if grep -q "OPENAI_API_KEY" .env; then
    sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_API_KEY/" .env
else
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env
fi

if grep -q "GOOGLE_PLACE_API_KEY" .env; then
    sed -i '' "s/GOOGLE_PLACE_API_KEY=.*/GOOGLE_PLACE_API_KEY=$GOOGLE_PLACES_API_KEY/" .env
else
    echo "GOOGLE_PLACE_API_KEY=$GOOGLE_PLACES_API_KEY" >> .env
fi

# Build and run the Docker container
echo "Building and starting Docker containers..."
docker-compose up -d --build

echo "Deployment complete!"
echo "Your frontend app is available at: http://localhost:7860"
echo "API documentation: http://localhost:8001/api/py/docs" 