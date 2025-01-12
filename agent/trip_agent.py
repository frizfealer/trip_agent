from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from agent.chat_openai_factory import ChatOpenAIFactory
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_functions_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

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
    "{recommender_country}, living in "
    "{recommender_country} for 20 years."
)

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
1. **{n_recommendations}** aligned recommendations** matching my interests.
2. **1-3 exploratory recommendations** beyond my stated interests but still fun and enriching. 
   If the recommendations are not aligned with my interests, 
   still provide their categories of interest, annotated with "Exploratory", e.g. "Exploratory: Hot spring, nature".
   If you cannot find any exploratory recommendations, you can skip this part.
3. Proritize must-visits categories and order the recommendations based on their relevance to my interests and their popularities.
4. For each recommendation, include the properties:
    1. name: Name of the attraction/activity.
    2. location: Specific location (address or district).
    3. duration: Recommended duration, in hours. If two days, put 48 hours.
    4. peopleCount: Suitable group size.
    5. budget: Estimated budget, in USD dollars.
    6. category: Category of interests. Can be one or more categories.
5. Each recommendation should include only the specified properties. Avoid adding extra information or descriptions.
"""

RECOMMENDATION_VERIFICATION_PROMPT = """
provided interests: {interests}
Proposed recommendations:
{firstpass_recommendations}

Please verify the recommendations using the following steps to ensure they meet the provided criteria. 
1. Aligned Recommendations Check (applied for not-`Exploratory` recommendations):
    Verify at least one categories of the recommendation match at least one of the provided interests.
2. Exploratory Recommendations Check (applied for `Exploratory` recommendations):
    Verify **ALL** categories of the recommendation are outside of the provided interests.
Below are some examples
Provided interests: Food, Sight-seeing, Cultural Experiences, Shopping, Night Markets
Recommendation 1:
    - ... (other properties)
    - **category**: Exploratory: Hot spring, Nature
    - status: pass
-> This recommendation passes the second check because it does not include any of the provided interests.
Recommendation 2:
    - ... (other properties)
    - **category**: Exploratory: Sight-seeing, Nature
-> This recommendation fails the second check because it includes Sight-seeing, which is one of the provided interests.
The output should be replaced with another recommendation that passes the checks. e.g.:
Recommendation 2:
    - ... (other properties)
    - **category**: Exploratory: Food, Nature
    - status: replaced
Recommendation 3:
    - ... (other properties)
    - **category**: Food, Nature
    - status: pass
-> This recommendation passes the first check because it includes Food, which is one of the provided interests.
Recommendation 4:
    - ... (other properties)
    - **category**: Hot spring, Nature
-> This recommendation fails the first check because either Hot spring or Nature is not one of the provided interests.
    The output should be replaced with another recommendation that passes the checks. e.g.:
Recommendation 4:
    - ... (other properties)
    - **category**: Sight-seeing, Nature
    - status: replaced
"""

VERFIED_RECOMMENDATION_EXTRACT_PROMPT = """
Please extract the from the belowing recommendations that are either stats = pass or status = replaced,
and output them with the properties that defined in the function `Attraction`
{verifed_recommendations}
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
        return categoreis.categories

    async def get_recommendations(
        self,
        n_recommendations: int,
        trip_days: int,
        people_count: int,
        location: str,
        budget: Optional[float] = None,
        interests: Optional[str] = None,
    ) -> List[Dict]:
        interests = f"{DEFAULT_CATEGORIES}, {interests}"
        n_recommendations = max(n_recommendations, 3)
        await self.set_country(location)
        # Generate the first-pass recommendations.
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", RECOMMENDATION_PROMPT),
            ],
        )
        recommender = self.chat_openai_factory.create(
            top_p=0.95,
            model_name="gpt-4o",
        )
        chain = prompt_template | recommender
        firstpass_recommendations = await chain.ainvoke(
            input={
                "recommender_country": self.country,
                "n_recommendations": n_recommendations,
                "trip_days": trip_days,
                "people_count": people_count,
                "location": location,
                "budget": budget,
                "interests": interests,
            }
        )
        # Verify the recommendations follow the criteria.
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", RECOMMENDATION_VERIFICATION_PROMPT),
            ]
        )
        knowledge_fn = self.chat_openai_factory.create(
            temperature=0, model_name="gpt-4o"
        )
        verifed_recommendations = await prompt_template.pipe(knowledge_fn).ainvoke(
            input={
                "recommender_country": self.country,
                "firstpass_recommendations": firstpass_recommendations.content,
                "interests": interests,
            }
        )
        # Extract the verfied recoomendations with all properties we want.
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                ("human", VERFIED_RECOMMENDATION_EXTRACT_PROMPT),
            ]
        )
        knowledge_fn = self.chat_openai_factory.create(
            model_name="gpt-4o"
        ).with_structured_output(TripRecommendations)
        recommendations = await prompt_template.pipe(knowledge_fn).ainvoke(
            input={
                "recommender_country": self.country,
                "verifed_recommendations": verifed_recommendations.content,
            }
        )
        return recommendations
