from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.chat_openai_factory import ChatOpenAIFactory
from agent.trip_preference import TripPreference

DEFAULT_CATEGORIES = "Must-visit"


class Categories(BaseModel):
    """A list of categories of interests for visitors to a location."""

    categories: List[str]


class Attraction(BaseModel):
    """A single attraction/activity recommendation to tell users."""

    name: str = Field(description="Name of the attraction/activity")
    location: str = Field(description="Specific location (address or district)")
    duration: int = Field(
        description="Recommended duration, in hours. If two days, put 48 hours."
    )
    peopleCount: int = Field(description="Suitable group size.")
    budget: int = Field(description="Estimated budget, in USD dollars.")
    category: str = Field(
        description="Category of interests. Can be one or more categories."
    )
    agenda: str = Field(
        description="Agenda highlights (key activities, pro tips, or unique insights)"
    )
    time: str = Field(description="Best time to visit (e.g. morning, evening)")


class TripRecommendations(BaseModel):
    """A list of attraction/activity recommendations for a trip."""

    recommendations: List[Attraction]


SYSTEM_MESSAGE_TEMPLATE = (
    "From now on, you are an excellent tour guide of "
    "{location}, living in "
    "{location} for 20 years."
)

RECOMMENDATION_PROMPT = """
**Trip Planning Request**

I am planning a trip and need detailed recommendations based on the following preferences:

### Trip Preferences
- **Duration**: {trip_days} days
- **Group Size**: {people_count} people
- **Location**: {location}
- **Budget**: {budget} in USD
- **Interests**: {interests}

---

### Recommendations
Please provide:
1. {n_recommendations} aligned recommendations** matching my interests. X depends on the overall duration and budget of my trip. 
    The overall duration and budget should be smaller or equals peferences duration and budget.
2. Proritize must-visits categories and order the recommendations based on their relevance to my interests and their popularities.
3. Make sure each recommendation's budget is within my budget.
4. Make sure each recommendation's duration is within my trip duration.
5. Make sure each recommendation's peopleCount is suitable for my group size.
"""

ITINERARY_PROMPT = """
I am planning a trip and need detailed recommendations based on the following preferences:

### Trip Preferences
- **Duration**: {trip_days} days
- **Group Size**: {people_count} people
- **Location**: {location}
- **Budget**: {budget} in USD
- **Interests**: {interests}

Given the following recommendations, enhance these recommendations by organizing them into an itinerary and adding
    unique insights, local tips, and any hidden gems that complement the listed places.
{recommendations}
"""


class TripAgent:
    def __init__(self, chat_openai_factory: ChatOpenAIFactory):
        self.chat_openai_factory = chat_openai_factory

    async def get_categories(self, location: str) -> Categories:
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                (
                    "human",
                    f"List 10 most interesting categories of activities for visitors to {location} (e.g. food, sight-seeing). Provide only concise category names without details.",
                ),
            ]
        )
        recommender = self.chat_openai_factory.create(
            top_p=0.9,
            model_name="gpt-4o",
        ).with_structured_output(Categories)
        chain = prompt_template | recommender
        categoreis = await chain.ainvoke(
            {
                "location": location,
            }
        )
        return categoreis.categories

    async def get_recommendations(
        self,
        n_recommendations: int,
        trip_preference: TripPreference,
    ) -> List[Attraction]:
        interests = f"{DEFAULT_CATEGORIES}, {", ".join(trip_preference.interests)}"
        n_recommendations = max(n_recommendations, 3)
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", RECOMMENDATION_PROMPT),
            ],
        )
        recommender = self.chat_openai_factory.create(
            top_p=0.95,
            model_name="gpt-4o",
        ).with_structured_output(TripRecommendations)
        chain = prompt_template | recommender
        recommendations = await chain.ainvoke(
            input={
                "n_recommendations": n_recommendations,
                "trip_days": trip_preference.trip_days,
                "people_count": trip_preference.people_count,
                "location": trip_preference.location,
                "budget": trip_preference.budget,
                "interests": interests,
            }
        )
        return recommendations.recommendations

    async def get_itinerary(
        self,
        recommendations: List[Attraction],
        trip_preference: TripPreference,
    ) -> str:
        """Generate a detailed itinerary based on the recommendations."""
        interests = f"{DEFAULT_CATEGORIES}, {", ".join(trip_preference.interests)}"
        recommendations_str = ""
        for rec in recommendations:
            recommendations_str += str(rec) + "\n"
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", ITINERARY_PROMPT),
            ],
        )
        planner = self.chat_openai_factory.create(
            top_p=0.95,
            model_name="gpt-4o",
        )
        chain = prompt_template | planner
        itinerary = await chain.ainvoke(
            input={
                "recommendations": recommendations_str,
                "trip_days": trip_preference.trip_days,
                "people_count": trip_preference.people_count,
                "location": trip_preference.location,
                "budget": trip_preference.budget,
                "interests": interests,
            }
        )
        return itinerary.content
