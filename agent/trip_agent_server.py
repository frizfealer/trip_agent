import base64
import logging
import os
import traceback
from typing import Any, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the necessary classes from our existing codebase
from agent.trip_agent import RefinedAttraction, TripAgent
from agent.utils.google_place_api import GooglePlaceAPI
from agent.utils.session_manager import SessionManager

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


class ItineraryDetailsConversationRequest(BaseModel):
    """Request model for the itinerary details conversation endpoint"""

    messages: Optional[List[Any]] = Field(
        default=None, description="Previous conversation messages. If None, starts a new conversation."
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for continuing a conversation. If None, creates a new session."
    )


class ItineraryDetailsConversationResponse(BaseModel):
    """Response model for the itinerary details conversation endpoint"""

    response: str = Field(description="The AI assistant's response")
    users_itinerary_details: List[dict] = Field(default_factory=list, description="The user's itinerary details")
    itinerary: Any = Field(default_factory=dict, description="The itinerary according to the user's requirements")
    session_id: str = Field(description="Session ID for continuing the conversation")


# Initialize the agent and session manager at startup
def get_trip_agent():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_PLACE_API_KEY")

    if not openai_api_key or not google_api_key:
        raise HTTPException(status_code=500, detail="OpenAI or Google API key not configured.")

    google_api = GooglePlaceAPI(api_key=google_api_key)
    return TripAgent(google_api)


# Create a single instance of TripAgent and SessionManager
trip_agent = get_trip_agent()
session_manager = SessionManager()


# Helper function for detailed error handling
def handle_exception(e: Exception, context: str = "API operation"):
    """
    Centralized error handler that provides detailed error information

    Args:
        e: The exception that was raised
        context: A description of what operation was being performed

    Raises:
        HTTPException with detailed error information
    """
    error_traceback = traceback.format_exc()
    error_details = {"error": str(e), "traceback": error_traceback, "context": context}
    logger.error(f"Error in {context}: {str(e)}\n{error_traceback}")
    raise HTTPException(status_code=500, detail=error_details)


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
        handle_exception(e, f"getting trip recommendations for {payload.city}")


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
        handle_exception(e, f"creating greedy itinerary for {payload.num_days} days")


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
        handle_exception(e, f"getting categories for {payload.city}")


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
        handle_exception(e, f"proxying image from {request.url}")


# Session management endpoints
@app.post("/api/py/sessions", status_code=201)
async def create_session():
    """
    Create a new conversation session.

    Returns:
        The session ID for the new session
    """
    try:
        session_id = session_manager.create_session()
        return {"session_id": session_id}
    except Exception as e:
        handle_exception(e, "creating new session")


@app.get("/api/py/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get the details of a specific session.

    Args:
        session_id: The ID of the session to retrieve

    Returns:
        The session details or a 404 if not found
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")

    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "last_accessed": session["last_accessed"],
        "message_count": len(session["messages"]),
        "has_itinerary_details": bool(session.get("users_itinerary_details")),
    }


@app.delete("/api/py/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a specific session.

    Args:
        session_id: The ID of the session to delete

    Returns:
        Success message or a 404 if not found
    """
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"status": "success", "message": f"Session {session_id} deleted"}


@app.post("/api/py/itinerary-details-conversation", response_model=ItineraryDetailsConversationResponse)
async def handle_itinerary_details_conversation(payload: ItineraryDetailsConversationRequest):
    """
    Handle multi-turn conversation for gathering trip details from users.

    Uses server-side session management to store conversation history.

    - If session_id is None, starts a new conversation and creates a new session
    - If session_id is provided, continues the existing conversation from that session

    Returns the assistant's response and updated session information.
    """
    try:
        # Handle session management
        session_id = payload.session_id
        session = None

        if session_id:
            # Try to retrieve existing session
            session = session_manager.get_session(session_id)
            if not session:
                # Session expired or not found, create a new one
                session_id = session_manager.create_session()
                session = session_manager.get_session(session_id)
        else:
            # No session ID provided, create a new session
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)

        # Initialize result with a default structure, ensuring it always has a session_id
        result = {
            "response": "",
            "messages": [],
            "users_itinerary_details": session.get("users_itinerary_details", []),
            "itinerary": session.get("itinerary", {}),
            "session_id": session_id,
        }

        if (
            session.get("users_itinerary_details") == []
            or "city" not in session.get("users_itinerary_details")[0]
            or "days" not in session.get("users_itinerary_details")[0]
        ):
            # Determine the messages to use
            if session.get("messages"):
                # Use messages from the session if available
                messages = session["messages"]
                messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in payload.messages or []])
            else:
                messages = []
            # Get response from the trip agent
            inquiry_result = await trip_agent.get_itinerary_inquiry(messages)
            # Process the response to ensure all items in messages are dictionaries
            session_manager.update_session(
                session_id,
                messages=inquiry_result["messages"],
                users_itinerary_details=inquiry_result.get("users_itinerary_details"),
            )

            # Update the result with the inquiry results
            result["response"] = inquiry_result["response"]
            result["messages"] = inquiry_result["messages"]
            result["users_itinerary_details"] = inquiry_result.get("users_itinerary_details", [])

        if (
            len(session.get("users_itinerary_details", [])) == 1
            and "city" in session.get("users_itinerary_details")[0]
            and "days" in session.get("users_itinerary_details")[0]
        ):
            # first time we get the itinerary details
            messages = []
            if payload.messages:
                messages = payload.messages

            draft_result = await trip_agent.get_itinerary_draft(
                itinerary_requirements=session.get("users_itinerary_details")[0],
                itinerary=session.get("itinerary", {}),
                messages=messages,
            )

            # Ensure itinerary is a dictionary format
            itinerary_data = draft_result.get("itinerary", {})
            if isinstance(itinerary_data, list):
                itinerary_data = {"days": itinerary_data}

            session_manager.update_session(session_id, itinerary=itinerary_data)

            # Update the result with the draft results
            result["response"] = draft_result.get("response", "")
            result["itinerary"] = itinerary_data

        # Run periodic cleanup of expired sessions (not waiting for result)
        session_manager.cleanup_expired_sessions()
        logger.info(f"session_id: {session_id}, Session data: {session}")
        return result
    except Exception as e:
        handle_exception(e, f"handling itinerary conversation with session {payload.session_id}")


if __name__ == "__main__":
    import uvicorn

    # run uvicorn agent.trip_agent_server:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
