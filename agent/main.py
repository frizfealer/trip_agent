import asyncio
import os

import requests
from dotenv import load_dotenv

from agent.chat_openai_factory import ChatOpenAIFactory
from agent.trip_agent import TripAgent
from agent.trip_preference import TripPreference


# Step 1: Set up API keys
def fetch_tripadvisor_attractions(location, category="attractions", limit=10):
    """
    Fetch attractions from TripAdvisor based on location and category.
    """
    url = "https://api.tripadvisor.com/v2/locations/search"
    headers = {"Authorization": f"Bearer {os.getenv("TRIPADVISOR_API_KEY")}"}
    params = {
        "query": location,
        "category": category,
        "limit": limit,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error fetching from TripAdvisor: {response.status_code}")
        return []


async def main():
    load_dotenv()
    trip_preference = TripPreference(
        trip_days=5,
        people_count=2,
        location="Taipei",
        budget=10,
    )
    chat_openai_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
    trip_agent = TripAgent(chat_openai_factory)
    categories = await trip_agent.get_categories(location=trip_preference.location)
    print(categories)
    trip_preference.interests = categories
    recommendations = await trip_agent.get_recommendations(
        n_recommendations=17,
        trip_preference=trip_preference,
    )
    for idx, r in enumerate(recommendations):
        print(f"{idx + 1}. {r}")
        print("------------------------------------------------------------")
    itinerary = await trip_agent.get_itinerary(
        recommendations, trip_preference=trip_preference
    )


if __name__ == "__main__":
    asyncio.run(main())
