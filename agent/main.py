from agent.trip_agent import TripAgent
from agent.chat_openai_factory import ChatOpenAIFactory
from dotenv import load_dotenv
import os
import asyncio


async def main():
    load_dotenv()
    chat_openai_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
    trip_agent = TripAgent(chat_openai_factory)
    categories = await trip_agent.get_categories("Taipei")
    print(categories)
    recommendations = await trip_agent.get_recommendations(
        n_recommendations=7,
        trip_days=5,
        people_count=2,
        # locations="Taipei",
        location="Taipei",
        budget=2000,
        interests=", ".join(categories),
    )
    for idx, r in enumerate(recommendations):
        print(f"{idx + 1}. {r}")
        print("------------------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
