#!/bin/bash

echo "Trip Agent API - Heroku Deployment Script"
echo "=========================================="

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "Heroku CLI not found. Please install it first:"
    echo "https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if user is logged in
heroku whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "You need to log in to Heroku first"
    heroku login
fi

# Get app name from user
read -p "Enter your app name (or press enter to let Heroku generate one): " APP_NAME

# Create the app
if [ -z "$APP_NAME" ]; then
    echo "Creating a new Heroku app with a generated name..."
    heroku create
    APP_NAME=$(heroku apps:info | grep "=== " | cut -d' ' -f2)
else
    echo "Creating a new Heroku app named $APP_NAME..."
    heroku create $APP_NAME
fi

# Set environment variables
echo "Setting up environment variables..."
read -p "Enter your OpenAI API key: " OPENAI_API_KEY
read -p "Enter your Google Places API key: " GOOGLE_PLACES_API_KEY
read -p "Enter your Google Maps API key: " GOOGLE_MAPS_API_KEY

heroku config:set OPENAI_API_KEY=$OPENAI_API_KEY --app $APP_NAME
heroku config:set GOOGLE_PLACE_API_KEY=$GOOGLE_PLACES_API_KEY --app $APP_NAME
heroku config:set PYTHONPATH=. --app $APP_NAME
heroku config:set GOOGLE_MAPS_API_KEY=$GOOGLE_MAPS_API_KEY --app $APP_NAME

# Deploy the app
echo "Deploying to Heroku..."
git push heroku main

# Display app URL
APP_URL=$(heroku apps:info --app $APP_NAME | grep "Web URL" | cut -d' ' -f3)
echo "Deployment complete!"
echo "Your app is available at: $APP_URL"
echo "API documentation: ${APP_URL}api/py/docs" 