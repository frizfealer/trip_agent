from agent.trip_agent import TripAgent
from agent.chat_openai_factory import ChatOpenAIFactory
from dotenv import load_dotenv
import os
import asyncio

import requests
import openai


# Step 1: Set up API keys
def fetch_tripadvisor_attractions(location, category="attractions", limit=10):
    """
    Fetch attractions from TripAdvisor based on location and category.
    """
    url = f"https://api.tripadvisor.com/v2/locations/search"
    headers = {"Authorization": f"Bearer {os.getenv("TRIPADVISOR_API_KEY")}"}
    params = {
        "query": location,
        "category": category,
        "limit": limit,
    }
    response = requests.get(url, headers=headers, params=params)
    import pdb

    pdb.set_trace()
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error fetching from TripAdvisor: {response.status_code}")
        return []


async def main():
    load_dotenv()
    chat_openai_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
    trip_agent = TripAgent(chat_openai_factory)
    categories = await trip_agent.get_categories("Taipei")
    print(categories)
    recommendations = await trip_agent.get_recommendations(
        n_recommendations=6,
        trip_days=5,
        people_count=2,
        # locations="Taipei",
        location="Taipei",
        budget=2000,
        interests=", ".join(categories[:]),
    )
    for idx, r in enumerate(recommendations):
        print(f"{idx + 1}. {r}")
        print("------------------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
