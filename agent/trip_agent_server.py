import os
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env

# Import the necessary classes from our existing codebase
from agent.chat_openai_factory import ChatOpenAIFactory
from agent.google_place_api import GooglePlaceAPI
from agent.trip_agent import RefinedAttraction, TripAgent
from agent.trip_preference import TripPreference

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


# Pydantic model for the request payload
class RecommendationRequest(BaseModel):
    city: str
    n_recommendations: int
    people_count: int
    budget: int
    interests: str  # expected as comma-separated interests


# Initialize the agent at startup
def get_trip_agent():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_PLACE_API_KEY")

    if not openai_api_key or not google_api_key:
        raise HTTPException(
            status_code=500, detail="OpenAI or Google API key not configured."
        )

    chat_factory = ChatOpenAIFactory(openai_api_key=openai_api_key)
    google_api = GooglePlaceAPI(api_key=google_api_key)
    return TripAgent(chat_factory, google_api)

# Create a single instance of TripAgent
trip_agent = get_trip_agent()

@app.post("/api/py/recommendations", response_model=List[RefinedAttraction])
async def get_trip_recommendations(payload: RecommendationRequest):
    """
    Endpoint to get a list of refined attraction recommendations.
    """
    print(f"Received request with payload: {payload}")

    # Process the interests string (comma-separated) into a list of interests
    interests_list = [
        interest.strip()
        for interest in payload.interests.split(",")
        if interest.strip()
    ]

    # Build trip preferences (using a default trip_days value, e.g., 7)
    preferences = TripPreference(
        location=payload.city,
        people_count=payload.people_count,
        budget=payload.budget,
        interests=interests_list,
        trip_days=7,
    )

    try:
        recommendations = await trip_agent.get_recommendations(
            payload.n_recommendations, preferences
        )

        # Clean the recommendations data before returning
        cleaned_recommendations = []
        for rec in recommendations:
            rec_dict = rec.model_dump()
            # Replace NaN values with None
            for key, value in rec_dict.items():
                if isinstance(value, float) and (value != value):  # Check for NaN
                    rec_dict[key] = None
            cleaned_recommendations.append(rec_dict)

        return cleaned_recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add the hello endpoint to the main app
@app.get("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI"}


# Add a Pydantic model for the categories request
class CategoriesRequest(BaseModel):
    city: str

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


if __name__ == "__main__":
    import uvicorn
    # run uvicorn agent.trip_agent_server:app --reload --port 8001   
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
