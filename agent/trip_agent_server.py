import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta

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

class ItineraryRequest(BaseModel):
    recommendations: List[RefinedAttraction]
    budget: int
    start_day: str
    num_days: int
    travel_type: str
    itinerary_description: str

class ChatSession:
    def __init__(self):
        self.messages = []
        self.weights_history = []
        self.itineraries = []
        self.current_recommendations: Optional[List[RefinedAttraction]] = None
        self.current_preferences: Optional[Dict] = None
        self.last_activity: datetime = datetime.now()
        self.favorite_itineraries: List[Dict] = []
        self.session_stats = {
            "total_itineraries": 0,
            "total_messages": 0,
            "start_time": datetime.now(),
            "weight_adjustments": [],
        }
        
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
        self.session_stats["total_messages"] += 1
        
    def add_favorite(self, itinerary_index: int) -> bool:
        """Add an itinerary to favorites."""
        if 0 <= itinerary_index < len(self.itineraries):
            self.favorite_itineraries.append(self.itineraries[itinerary_index])
            return True
        return False
        
    def get_session_stats(self) -> Dict:
        """Get detailed statistics about the session."""
        duration = datetime.now() - self.session_stats["start_time"]
        return {
            **self.session_stats,
            "session_duration": str(duration),
            "avg_iterations_per_itinerary": len(self.weights_history) / max(1, len(self.itineraries)),
            "favorite_count": len(self.favorite_itineraries),
            "last_active": self.last_activity.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
    def revert_to_itinerary(self, index: int) -> Optional[Dict]:
        """Revert to a previous itinerary."""
        if 0 <= index < len(self.itineraries):
            return self.itineraries[index]
        return None
        
    def compare_itineraries(self, n: int = 2) -> str:
        """Compare the n most recent itineraries and their optimizations."""
        if len(self.itineraries) < 2:
            return "Not enough itineraries to compare."
            
        recent_itineraries = self.itineraries[-n:]
        comparison = ["Here's a comparison of the recent itineraries:"]
        
        for idx, itin in enumerate(recent_itineraries, 1):
            comparison.append(f"\nItinerary {idx}:")
            comparison.append(f"Generated at: {itin['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            if 'weights_explanation' in itin:
                comparison.append(f"Optimization: {itin['weights_explanation']}")
            
            # Add detailed differences if we have a previous itinerary to compare with
            if idx > 1:
                prev_itin = recent_itineraries[idx-2]
                differences = self._compare_itinerary_details(prev_itin, itin)
                if differences:
                    comparison.append("Changes from previous version:")
                    comparison.extend(differences)
                
        return "\n".join(comparison)
        
    def _compare_itinerary_details(self, prev_itin: Dict, curr_itin: Dict) -> List[str]:
        """Compare detailed differences between two itineraries."""
        differences = []
        
        # Compare weights if available
        if 'weights' in prev_itin and 'weights' in curr_itin:
            for key, curr_val in curr_itin['weights'].items():
                prev_val = prev_itin['weights'].get(key, 0)
                if abs(curr_val - prev_val) > 0.1:  # Significant change threshold
                    if curr_val > prev_val:
                        differences.append(f"- Increased emphasis on {key[2:]}")
                    else:
                        differences.append(f"- Decreased emphasis on {key[2:]}")
        
        return differences
        
    def adjust_weights_from_feedback(self, feedback: str) -> Dict[str, float]:
        """Adjust weights based on user feedback using keyword analysis."""
        base_weights = self.weights_history[-1]['weights'].copy() if self.weights_history else {
            "w_xp": 0.5, "w_count": 0.5, "w_cost": 0.5,
            "w_dur": 0.5, "w_travel": 0.5, "w_time": 0.5
        }
        
        # Analyze feedback for keywords and adjust weights
        feedback_lower = feedback.lower()
        adjustments = []
        
        if any(word in feedback_lower for word in ['rushed', 'tight', 'busy', 'packed']):
            base_weights['w_count'] *= 0.7
            base_weights['w_dur'] *= 1.3
            adjustments.append("relaxed pace")
            
        if any(word in feedback_lower for word in ['cheap', 'expensive', 'cost', 'budget']):
            base_weights['w_cost'] *= 1.3
            adjustments.append("cost optimization")
            
        if any(word in feedback_lower for word in ['quality', 'best', 'top', 'rated']):
            base_weights['w_xp'] *= 1.3
            adjustments.append("quality focus")
            
        if any(word in feedback_lower for word in ['close', 'near', 'travel', 'distance']):
            base_weights['w_travel'] *= 1.3
            adjustments.append("travel optimization")
            
        # Normalize weights to ensure they stay in reasonable range
        max_weight = max(base_weights.values())
        if max_weight > 1.0:
            scale = 1.0 / max_weight
            base_weights = {k: v * scale for k, v in base_weights.items()}
        
        # Record the adjustment
        self.session_stats["weight_adjustments"].append({
            "timestamp": datetime.now(),
            "feedback": feedback,
            "adjustments": adjustments,
            "weights": base_weights
        })
        
        return base_weights
        
    def is_inactive(self, timeout_minutes: int = 60) -> bool:
        """Check if the session has been inactive for longer than the timeout."""
        return (datetime.now() - self.last_activity) > timedelta(minutes=timeout_minutes)
        
    def reset(self):
        """Reset the session while keeping recommendations and preferences."""
        self.messages = []
        self.weights_history = []
        self.itineraries = []
        self.update_activity()
        
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now()})
        
    def add_weights(self, weights: Dict[str, float]):
        self.weights_history.append({
            "weights": weights,
            "timestamp": datetime.now(),
            "explanation": self._get_weights_explanation(weights)
        })
    
    def _get_weights_explanation(self, weights: Dict[str, float]) -> str:
        """Generate a human-readable explanation of what the weights mean for the itinerary."""
        explanations = []
        if weights["w_xp"] > 0.6:
            explanations.append("prioritizing highly-rated attractions")
        if weights["w_count"] > 0.6:
            explanations.append("fitting in more activities")
        if weights["w_cost"] > 0.6:
            explanations.append("keeping costs low")
        if weights["w_dur"] > 0.6:
            explanations.append("preferring shorter activities")
        if weights["w_travel"] > 0.6:
            explanations.append("minimizing travel time between locations")
        if weights["w_time"] > 0.6:
            explanations.append("optimizing for time slots")
            
        if not explanations:
            return "balanced optimization across all factors"
        return "optimization focused on: " + ", ".join(explanations)
        
    def add_itinerary(self, itinerary: str, weights: Optional[Dict[str, float]] = None):
        entry = {
            "itinerary": itinerary,
            "timestamp": datetime.now(),
        }
        if weights:
            entry["weights"] = weights
            entry["weights_explanation"] = self._get_weights_explanation(weights)
        self.itineraries.append(entry)
        
    def get_history(self):
        return {
            "messages": self.messages,
            "weights_history": self.weights_history,
            "itineraries": self.itineraries
        }

class ChatMessage(BaseModel):
    session_id: str
    message: str
    recommendations: Optional[List[RefinedAttraction]] = None
    preferences: Optional[Dict] = None

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

# In-memory storage for chat sessions
chat_sessions: Dict[str, ChatSession] = {}

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
            itinerary_description=payload.itinerary_description
        )
        
        return {"itinerary": itinerary_str}
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

@app.post("/api/py/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    Endpoint for chat-based itinerary planning.
    Maintains conversation history and allows for iterative refinement of itineraries.
    """
    # Cleanup inactive sessions
    await cleanup_inactive_sessions()
    
    # Create new session if it doesn't exist
    if chat_message.session_id not in chat_sessions:
        chat_sessions[chat_message.session_id] = ChatSession()
    
    session = chat_sessions[chat_message.session_id]
    session.update_activity()
    
    # Store recommendations and preferences if provided
    if chat_message.recommendations:
        session.current_recommendations = chat_message.recommendations
    if chat_message.preferences:
        session.current_preferences = chat_message.preferences
    
    # Add user message to history
    session.add_message("user", chat_message.message)
    
    try:
        # If we have recommendations, generate new itinerary based on feedback
        if session.current_recommendations and session.current_preferences:
            try:
                # Adjust weights based on user feedback
                adjusted_weights = session.adjust_weights_from_feedback(chat_message.message)
                
                itinerary_str = await trip_agent.get_itinerary_with_greedy_scheduler(
                    recommendations=session.current_recommendations,
                    trip_days=session.current_preferences.get("num_days", 7),
                    budget=session.current_preferences.get("budget", 1000),
                    start_day=session.current_preferences.get("start_day", "Monday"),
                    travel_type=session.current_preferences.get("travel_type", "driving"),
                    itinerary_description=chat_message.message
                )
                
                # Update session statistics
                session.session_stats["total_itineraries"] += 1
                
                # Store the new itinerary with its associated weights
                session.add_itinerary(itinerary_str, adjusted_weights)
                
                # Prepare response message
                response_message = "I've generated a new itinerary based on your feedback."
                if adjusted_weights:
                    weights_explanation = session._get_weights_explanation(adjusted_weights)
                    response_message += f"\n\nThis itinerary was generated with {weights_explanation}."
                
                # Add comparison with previous itinerary if available
                if len(session.itineraries) > 1:
                    response_message += "\n\nCompared to the previous itinerary, this version "
                    if adjusted_weights:
                        response_message += f"focuses more on {weights_explanation}."
                    
                response_message += f"\n\n{itinerary_str}"
                
                # Add system response
                session.add_message("assistant", response_message)
                
                return {
                    "response": response_message,
                    "history": session.get_history(),
                    "latest_weights": adjusted_weights,
                    "success": True
                }
            except Exception as e:
                error_message = f"Error generating itinerary: {str(e)}"
                session.add_message("assistant", error_message)
                return {
                    "response": error_message,
                    "history": session.get_history(),
                    "latest_weights": None,
                    "success": False
                }
        else:
            message = "Please provide recommendations and preferences to generate an itinerary."
            session.add_message("assistant", message)
            return {
                "response": message,
                "history": session.get_history(),
                "latest_weights": None,
                "success": False
            }
            
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        session.add_message("assistant", error_message)
        return {
            "response": error_message,
            "history": session.get_history(),
            "latest_weights": None,
            "success": False
        }

@app.get("/api/py/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    Endpoint to retrieve chat history for a specific session.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id].get_history()

@app.post("/api/py/chat/reset/{session_id}")
async def reset_chat_session(session_id: str):
    """Reset a chat session while keeping recommendations and preferences."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_sessions[session_id].reset()
    return {"message": "Session reset successfully"}

@app.get("/api/py/chat/compare/{session_id}")
async def compare_itineraries(session_id: str, n: int = 2):
    """Compare the n most recent itineraries in a session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    comparison = chat_sessions[session_id].compare_itineraries(n)
    return {"comparison": comparison}

@app.post("/api/py/chat/favorite/{session_id}/{itinerary_index}")
async def add_favorite_itinerary(session_id: str, itinerary_index: int):
    """Add an itinerary to favorites."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = chat_sessions[session_id].add_favorite(itinerary_index)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid itinerary index")
    
    return {"message": "Itinerary added to favorites"}

@app.get("/api/py/chat/stats/{session_id}")
async def get_session_statistics(session_id: str):
    """Get detailed statistics about a chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id].get_session_stats()

@app.post("/api/py/chat/revert/{session_id}/{itinerary_index}")
async def revert_to_itinerary(session_id: str, itinerary_index: int):
    """Revert to a previous itinerary."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    itinerary = chat_sessions[session_id].revert_to_itinerary(itinerary_index)
    if not itinerary:
        raise HTTPException(status_code=400, detail="Invalid itinerary index")
    
    return {"message": "Reverted to previous itinerary", "itinerary": itinerary}

async def cleanup_inactive_sessions():
    """Remove inactive sessions to free up memory."""
    inactive_sessions = [
        session_id for session_id, session in chat_sessions.items()
        if session.is_inactive()
    ]
    for session_id in inactive_sessions:
        del chat_sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    # run uvicorn agent.trip_agent_server:app --reload --port 8001   
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
