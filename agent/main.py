import asyncio
import os

from dotenv import load_dotenv

from agent.chat_openai_factory import ChatOpenAIFactory
from agent.google_place_api import GooglePlaceAPI
from agent.trip_agent import TripAgent
from agent.trip_preference import TripPreference


async def main():
    load_dotenv()
    trip_preference = TripPreference(
        trip_days=5,
        people_count=2,
        location="Taipei",
        budget=10,
    )
    chat_openai_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
    google_place_api = GooglePlaceAPI(api_key=os.getenv("GOOGLE_PLACE_API_KEY"))
    trip_agent = TripAgent(chat_openai_factory, google_place_api)

    categories = await trip_agent.get_categories(location=trip_preference.location)
    print(categories)
    trip_preference.interests = categories
    recommendations = await trip_agent.get_recommendations(
        n_recommendations=5,
        trip_preference=trip_preference,
    )
    for idx, r in enumerate(recommendations):
        print(f"{idx + 1}. {r}")
        print("------------------------------------------------------------")
    with open("recommendations.pkl", "wb") as f:
        import pickle

        pickle.dump(recommendations, f)
    # part two generate itinerary
    with open("recommendations.pkl", "rb") as f:
        import pickle

        recommendations = pickle.load(f)

    itinerary_it = await trip_agent.get_itinerary_with_reflection(
        recommendations, trip_preference, reflection_num=2
    )
    import pdb

    pdb.set_trace()
    async for chunk in itinerary_it:
        print(chunk["messages"][-1].pretty_print())


if __name__ == "__main__":
    asyncio.run(main())
