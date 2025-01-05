from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from agent.chat_openai_factory import ChatOpenAIFactory
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_functions_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


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
    peopleCount: str = Field(description="Suitable group size.")
    budget: str = Field(description="Estimated budget, in USD dollars.")
    category: str = Field(
        description="Category of interests. Can be one or more categories."
    )
    agenda: str = Field(
        description="Agenda highlights (key activities, tips, or unique insights)"
    )


class TripRecommendations(BaseModel):
    """A list of attraction/activity recommendations for a trip."""

    recommendations: List[Attraction]


SYSTEM_MESSAGE_TEMPLATE = """From now on, \
    you are an excellent tour guide of {recommender_country}, living in {recommender_country} for 20 years.
"""

RECOMMENDATION_PROMPT = """
**Trip Planning Request**

I am planning a trip and need detailed recommendations based on the following preferences:

### Trip Preferences
- **Duration**: {trip_days} days
- **Group Size**: {people_count} people
- **Location**: {location}
- **Budget**: {budget}
- **Interests**: {interests}

---

### Recommendations
Please provide:
1. **{n_recommendations} aligned recommendations** matching my interests.
2. **1-3 exploratory recommendations** beyond my stated interests but still fun and enriching. 
   If the recommendations are not aligned with my interests, 
   still provide their categories of interest, annotated with "Exploratory", e.g. "Exploratory: Hot spring, nature".
For each recommendation, include:
1. **Name of the attraction/activity**
2. **Specific location (address or district)**
3. **Recommended duration**
4. **Suitable group size**
5. **Estimated budget**
6. **Category of interests. Can be one or more categories.**
7. **Agenda highlights** (key activities, tips, or unique insights).

"""


class TripAgent:
    def __init__(self, chat_openai_factory: ChatOpenAIFactory):
        self.chat_openai_factory = chat_openai_factory
        self.country = None

    async def set_country(self, location: str, reset: bool = False):
        if self.country is None or reset:
            knowledge_fn = self.chat_openai_factory.create(
                temperature=0, model_name="gpt-4o"
            )
            self.country = await knowledge_fn.pipe(StrOutputParser()).ainvoke(
                [
                    (
                        "human",
                        f"what are the country of the following location, answering just the country name, don't show redundant country: {location}",
                    )
                ]
            )

    async def get_categories(self, location: str) -> Categories:
        await self.set_country(location)
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
                "recommender_country": self.country,
                "location": location,
            }
        )
        return categoreis.categories + ["Hidden Gems", "Must-visit"]

    async def get_recommendations(
        self,
        n_recommendations: int,
        trip_days: int,
        people_count: int,
        location: str,
        budget: Optional[float] = None,
        interests: Optional[str] = None,
    ) -> List[Dict]:
        n_recommendations = min(n_recommendations, 3)
        await self.set_country(location)
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", RECOMMENDATION_PROMPT),
            ]
        )
        recommender = self.chat_openai_factory.create(
            top_p=0.9,
            model_name="gpt-4o",
        ).with_structured_output(TripRecommendations)
        chain = prompt_template | recommender
        recommendations = await chain.ainvoke(
            {
                "recommender_country": self.country,
                "n_recommendations": n_recommendations,
                "trip_days": trip_days,
                "people_count": people_count,
                "location": location,
                "budget": budget,
                "interests": interests,
            }
        )
        return recommendations.recommendations
