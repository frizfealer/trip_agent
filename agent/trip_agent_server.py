import base64
import logging
import os
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the necessary classes from our existing codebase
from agent.chat_openai_factory import ChatOpenAIFactory
from agent.trip_agent import RefinedAttraction, TripAgent
from agent.utils.google_place_api import GooglePlaceAPI

load_dotenv()  # Load environment variables from .env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trip Planner API",
    description="An API to get detailed attraction recommendations based on trip preferences.",
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
)

# Update the CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    # "http://localhost:3001",  # Add your frontend's actual port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add a Pydantic model for the categories request
class CategoriesRequest(BaseModel):
    city: str


# for the input of the recommendations endpoint
class RecommendationRequest(BaseModel):
    city: str
    interests: List[str]
    excluded_recommendations: List[str]


# for the output of the recommendations endpoint
class Experience(BaseModel):
    title: str
    imageUrl: str
    duration: str
    price: float
    city: str
    category: str


class ItineraryRequest(BaseModel):
    recommendations: List[RefinedAttraction]
    budget: int
    start_day: str
    num_days: int
    travel_type: str
    itinerary_description: str


class ImageProxyRequest(BaseModel):
    url: str


# Initialize the agent at startup
def get_trip_agent():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_PLACE_API_KEY")

    if not openai_api_key or not google_api_key:
        raise HTTPException(status_code=500, detail="OpenAI or Google API key not configured.")

    chat_factory = ChatOpenAIFactory(openai_api_key=openai_api_key)
    google_api = GooglePlaceAPI(api_key=google_api_key)
    return TripAgent(chat_factory, google_api)


# Create a single instance of TripAgent
trip_agent = get_trip_agent()


@app.post("/api/py/recommendations", response_model=List[Experience])
async def get_trip_recommendations(payload: RecommendationRequest):
    """
    Endpoint to get a list of refined attraction recommendations.
    """
    print(f"Received request with payload: {payload}")

    try:
        recommendations = await trip_agent.get_recommendations(
            payload.city, len(payload.interests), payload.interests, payload.excluded_recommendations
        )

        # Convert RefinedAttraction to Experience
        experiences = []
        for rec in recommendations:
            experience = Experience(
                title=rec.name,
                imageUrl=rec.photos[0] if rec.photos else "",  # Use first photo URL if available
                duration=f"{rec.duration} hours",
                price=float(rec.cost),  # Convert cost to float for price
                city=payload.city,
                category=rec.category,
            )
            experiences.append(experience)

        return experiences
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/py/greedy-itinerary")
async def get_greedy_itinerary(payload: ItineraryRequest):
    """
    Endpoint to get a detailed itinerary using the greedy scheduler algorithm.
    This endpoint optimizes the schedule based on opening hours, costs, and time preferences.
    """
    try:
        # The recommendations are already RefinedAttraction objects thanks to FastAPI's automatic deserialization
        # Get the optimized itinerary using the greedy scheduler
        itinerary_str = await trip_agent.get_itinerary_with_greedy_scheduler(
            recommendations=payload.recommendations,
            trip_days=payload.num_days,
            budget=payload.budget,
            start_day=payload.start_day,
            travel_type=payload.travel_type,
            itinerary_description=payload.itinerary_description,
        )

        return {"itinerary": itinerary_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add the hello endpoint to the main app
@app.get("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI"}


@app.post("/api/py/categories", response_model=List[str])
async def get_categories(payload: CategoriesRequest):
    """
    Endpoint to get available categories for trip planning.
    Takes a city parameter to get location-specific categories.
    """
    try:
        categories = await trip_agent.get_categories(payload.city)
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/py/proxy-image")
async def proxy_image(request: ImageProxyRequest):
    """
    Proxy endpoint for Google Places API images.
    Fetches the image and returns it as a base64 encoded string.
    """
    try:
        # Get the Google Places API key from environment
        api_key = os.getenv("GOOGLE_PLACE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Google Places API key not configured")

        # Add the API key to the URL if it's a Google Places API URL
        if "places.googleapis.com" in request.url:
            separator = "&" if "?" in request.url else "?"
            url = f"{request.url}{separator}key={api_key}"
        else:
            url = request.url
        logger.info(f"Fetching image from {url}")
        # Fetch the image
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image")

        # Convert to base64
        image_base64 = base64.b64encode(response.content).decode()

        # Return with appropriate mime type
        content_type = response.headers.get("content-type", "image/jpeg")
        return {"imageUrl": f"data:{content_type};base64,{image_base64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # run uvicorn agent.trip_agent_server:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
