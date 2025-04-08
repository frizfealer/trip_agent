import base64
import logging
import os
import traceback
from datetime import UTC, datetime
from typing import Any, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the AppException from our exceptions module
from agent.exceptions import AppException

# Import the necessary classes from our existing codebase
from agent.trip_agent import RefinedAttraction, TripAgent
from agent.utils.google_place_api import GooglePlaceAPI
from agent.utils.session_manager import SessionManager

load_dotenv()  # Load environment variables from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trip Planner API",
    description="An API to get detailed attraction recommendations based on trip preferences.",
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
)


# --- Exception Handlers ---
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    # Use logger.exception to include traceback automatically
    logger.exception(
        f"[AppException] {exc.message} "
        f"URL={request.url.path} "
        f"Method={request.method} "
        f"Client={request.client.host if request.client else 'unknown'}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.status_code,
                "error_code": exc.error_code,
                "timestamp": datetime.now(UTC).isoformat(),
                "path": str(request.url.path),
            },
            "meta": {"status": "error", "timestamp": datetime.now(UTC).isoformat(), "version": "1.0"},
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"[Unhandled Exception] {type(exc).__name__} at {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal Server Error",
                "path": str(request.url.path),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.info(f"[Validation Error] at {request.method} {request.url.path} — {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Validation Error",
                "details": exc.errors(),
                "path": str(request.url.path),
            }
        },
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


class TripRequirements(BaseModel):
    city: str
    days: int
    start_date: str
    end_date: Optional[str] = None
    budget: Optional[float] = None
    session_id: Optional[str] = None


class ItineraryDraftConversationRequest(BaseModel):
    """Request model for the itinerary details conversation endpoint"""

    messages: Optional[List[Any]] = Field(
        default=None, description="Previous conversation messages. If None, starts a new conversation."
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for continuing a conversation. If None, creates a new session."
    )


class ItineraryDraftConversationResponse(BaseModel):
    """Response model for the itinerary details conversation endpoint"""

    response: str = Field(description="The AI assistant's response")
    trip_requirements: dict = Field(default_factory=dict, description="The user's trip requirements")
    itinerary: dict = Field(default_factory=dict, description="The itinerary according to the user's requirements")
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
session_manager = SessionManager(prefix="trip_agent_session:")


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


def _get_session(session_id: Optional[str]):
    # Handle session management
    session = None
    if session_id:
        # Try to retrieve existing session
        session = session_manager.get_session(session_id)
    if not session_id or not session:
        # No session ID provided, create a new session
        session_id = session_manager.create_session(
            {
                "messages": [],
                "trip_requirements": {},
                "itinerary": {},
            }
        )
        session = session_manager.get_session(session_id)

    return session, session_id


@app.post("/api/py/post-trip-requirements")
async def handle_trip_requirements(payload: TripRequirements):
    """
    Endpoint to post trip requirements (city, days, starting_date, etc.).

    Required parameters:
    - city: The destination city
    - days: Number of days for the trip
    - starting_date: Start date of the trip

    Optional parameters:
    - end_date: End date of the trip
    - budget: Budget for the trip
    - session_id: Session ID to associate with this request

    Returns:
        JSON response with standardized format:
        {
            "data": {
                "session_id": str,
                "trip_requirements": dict
            }
        }

    Raises:
        RequestValidationError: For invalid input format (handled by FastAPI)
        AppException: For business logic validation errors or server errors
    """
    try:
        _, session_id = _get_session(payload.session_id)

        # Extract trip requirements data
        trip_data = {"city": payload.city, "days": payload.days, "start_date": payload.start_date}

        # Add optional fields if provided
        if payload.end_date:
            trip_data["end_date"] = payload.end_date
        if payload.budget:
            trip_data["budget"] = payload.budget

        logger.info(f"Updating session with trip requirements: {trip_data}")
        # Update session with trip requirements
        session_manager.update_session(session_id, **{"trip_requirements": trip_data})

        return {"session_id": session_id}
    except AppException:
        # Re-raise AppExceptions as they already have the correct format
        raise
    except Exception as e:
        # Log the original error with traceback first
        logger.exception(f"Original error processing trip requirements for session {payload.session_id or 'new'}")
        # For unexpected server-side errors, wrap in AppException for consistent API response
        raise AppException(
            message="Failed to process trip requirements",
            status_code=500,
            error_code="TRIP_REQUIREMENTS_ERROR",
        )


@app.post("/api/py/itinerary-draft-conversation", response_model=ItineraryDraftConversationResponse)
async def handle_itinerary_draft_conversation(payload: ItineraryDraftConversationRequest):
    """
    Handle multi-turn conversation for gathering trip details from users.

    Uses server-side session management to store conversation history.

    - If session_id is None, starts a new conversation and creates a new session
    - If session_id is provided, continues the existing conversation from that session

    Returns the assistant's response and updated session information.
    """
    try:
        # Handle session management using the updated _get_session function
        session, session_id = _get_session(payload.session_id)
        logger.info(f"session_id: {session_id}, Session data: {session}")

        # Initialize result with a default structure, ensuring it always has a session_id
        result = {
            "response": "",
            "messages": [],
            "trip_requirements": session.get("trip_requirements", {}),
            "itinerary": session.get("itinerary", {}),
            "session_id": session_id,
        }

        # if the trip requirements are not set, we need to get the itinerary details
        if not session.get("trip_requirements"):
            result["data"]["response"] = "Please provide your trip requirements in the form."
            return result

        messages = []
        if payload.messages:
            messages = payload.messages
        draft_result = await trip_agent.get_itinerary_draft(
            trip_requirements=session.get("trip_requirements"),
            itinerary=session.get("itinerary", {}),
            messages=messages,
        )
        # Ensure itinerary is a dictionary format
        if draft_result.get("itinerary_updated", False):
            itinerary_draft = {"days": draft_result.get("itinerary", [])}
            session_manager.update_session(session_id, **{"itinerary": itinerary_draft})
            result["itinerary"] = itinerary_draft
        # Update the result with the draft results
        result["response"] = draft_result.get("response", "")
        # Run periodic cleanup of expired sessions (not waiting for result)
        session_manager.cleanup_expired_sessions()
        return result
    except AppException:
        # Re-raise AppExceptions as they already have the correct format
        raise
    except Exception as e:
        # For unexpected errors, wrap in AppException with appropriate context
        raise AppException(
            message=f"Failed to process itinerary conversation: {str(e)}",
            status_code=500,
            error_code="ITINERARY_CONVERSATION_ERROR",
        )


if __name__ == "__main__":
    import uvicorn

    # run uvicorn agent.trip_agent_server:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
