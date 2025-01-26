import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field
from trip_agent import TripAgent  # Assuming you have the TripAgent class available

from agent.chat_openai_factory import ChatOpenAIFactory  # Assuming this factory exists
from agent.google_place_api import GooglePlaceAPI  # Assuming this API exists


# Define TripPreference data structure
class TripPreference(BaseModel):
    """Trip preference of a trip specified by the user."""

    trip_days: int = Field(..., description="Duration of the trip in days")
    people_count: int = Field(..., description="Number of people in the group")
    location: str = Field(..., description="Location of the trip")
    budget: float = Field(..., description="Budget for the trip in USD")
    interests: Optional[List[str]] = Field(
        None, description="List of interests for the trip"
    )


# Initialize the Flask app
app = Flask(__name__)

# Create an instance of TripAgent
load_dotenv()
chat_openai_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
google_place_api = GooglePlaceAPI(api_key=os.getenv("GOOGLE_PLACE_API_KEY"))
trip_agent = TripAgent(chat_openai_factory, google_place_api)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_categories", methods=["POST"])
def get_categories():
    data = request.json
    city = data.get("city")
    if not city:
        return jsonify({"error": "City is required"}), 400

    # Use asyncio to call the async method
    categories = asyncio.run(trip_agent.get_categories(city))

    return jsonify({"categories": categories})


@app.route("/submit_categories", methods=["POST"])
def submit_categories():
    try:
        data = request.json
        print("Incoming data:", data)  # Log the incoming data

        # Extract fields
        selected_categories = data.get("categories")
        people_count = int(data.get("peopleCount"))  # Convert to integer
        trip_days = int(data.get("tripDays"))  # Convert to integer
        budget = float(data.get("budget"))  # Convert to float
        location = data.get("location") or data.get(
            "city"
        )  # Get location or fallback to city

        if (
            not selected_categories
            or not people_count
            or not trip_days
            or not budget
            or not location
        ):
            print("Error: Missing required fields")
            return jsonify({"error": "All fields are required"}), 400

        # Construct TripPreference object
        trip_preference = TripPreference(
            trip_days=trip_days,
            people_count=people_count,
            location=location,
            budget=budget,
            interests=selected_categories,
        )

        # Use trip_agent to get recommendations
        recommendations = asyncio.run(
            trip_agent.get_recommendations(
                n_recommendations=10, trip_preference=trip_preference
            )
        )
        print([rec.model_dump() for rec in recommendations])

        # Return recommendations to the frontend
        return jsonify(
            {
                "message": "Recommendations fetched successfully",
                "recommendations": [rec.model_dump() for rec in recommendations],
            }
        )

    except ValueError as e:
        print("ValueError:", str(e))  # Log the ValueError
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        print("Exception:", str(e))  # Log any unexpected exceptions
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/submit_recommendations", methods=["POST"])
def submit_recommendations():
    data = request.json
    selected_recommendations = data.get("selectedRecommendations", [])
    if not selected_recommendations:
        return jsonify({"error": "No recommendations selected"}), 400

    # Process selected recommendations
    # Call trip_agent.get_itinerary_with_reflection()
    # itinerary = trip_agent.get_itinerary_with_reflection(selected_recommendations)
    print(selected_recommendations)
    # return jsonify({"itinerary": itinerary})


if __name__ == "__main__":
    app.run(debug=True)
