import copy
import json
import logging
from typing import Annotated, List, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.chat_openai_factory import ChatOpenAIFactory
from agent.google_place_api import GooglePlaceAPI
from agent.scheduler.event import (
    Day,
    Event,
    convert_hours_to_slots,
    convert_minutes_to_slots,
    parse_regular_opening_hours,
)
from agent.scheduler.itinerary import (
    generate_travel_cost_matrix_from_travel_time_matrix,
    postprocess_travel_time_matrix,
)
from agent.scheduler.itinerary_scheduler import ItineraryScheduler
from agent.trip_preference import TripPreference
from agent.utils.travel_time import get_travel_time_matrix

client = OpenAI()
DEFAULT_CATEGORIES = "Must-visit"


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


class Categories(BaseModel):
    """A list of categories of interests for visitors to a location."""

    categories: List[str]


class ProposedAttraction(BaseModel):
    """A single attraction/activity recommendation proposed by the LLM."""

    name: str = Field(description="Name of the attraction/activity")
    duration: float = Field(description="Recommended duration, in hours. If two days, put 48 hours.")
    category: str = Field(
        description="Category of interests. Can be one or more categories, at most three categories."
    )
    # agenda: str = Field(
    #     description="Agenda highlights in one sentence."
    # )
    cost: int = Field(description="Price of the attraction/activity in USD")
    time: str = Field(description="Best time to visit (e.g. morning, evening)")


class RefinedAttraction(ProposedAttraction):
    """A single attraction/activity recommendation to tell users, after verification."""

    rating: Union[float, None] = Field(default=None, description="Rating of the attraction from Google reviews.")
    regular_opening_hours: str = Field(default="", description="Regular opening hours of the attraction.")
    formatted_address: str = Field(default="", description="The address of the attraction.")
    website_uri: str = Field(default="", description="The website of the attraction.")
    editorial_summary: str = Field(default="", description="The editorial summary of the attraction.")
    photos: List[str] = Field(default_factory=list, description="A list of photos for the attraction.")


class TripRecommendations(BaseModel):
    """A list of attraction/activity recommendations for a trip."""

    proposed_attraction: List[ProposedAttraction]


class ItineraryWeightParameters(BaseModel):
    w_xp: float = Field(description="Weight for experience points (higher = prefer high-XP events)")
    w_count: float = Field(description="Weight for event count bonus (higher = prefer more events)")
    w_cost: float = Field(description="Weight for event cost penalty (higher = avoid expensive events)")
    w_dur: float = Field(description="Weight for event duration penalty (higher = prefer shorter events)")
    w_travel: float = Field(description="Weight for travel cost penalty (higher = penalize expensive travel cost)")
    w_time: float = Field(description="Weight for travel time penalty (higher = penalize longer travel times)")
    w_gap: float = Field(
        description="Weight for gap between events (higher = prefer more gap between events, less events, more relaxed itinerary)"
    )


class ActivityDetail(BaseModel):
    duration: str = Field(description="Duration of the activity, example: '3 hours'")
    agenda: str = Field(description="Description of the activity")
    budget: str = Field(description="Estimated budget for the activity, example: '$0'")


class Activity(BaseModel):
    time_of_day: str = Field(description="Time of day for the activity, example: 'Morning', 'Afternoon', 'Evening'")
    title: str = Field(description="Title of the activity, example: 'Chiang Kai-shek Memorial Hall'")
    location: str = Field(description="Location of the activity, example: 'Zhongzheng District'")
    details: ActivityDetail


class TripItinerary(BaseModel):
    trip_title: str = Field(description="Title of the trip, example: '7-Day Taiwan Adventure'")
    start_date: str = Field(description="Start date of the trip in 'YYYY-MM-DD' format")
    end_date: str = Field(description="End date of the trip in 'YYYY-MM-DD' format")
    total_budget: str = Field(
        description="Total budget for the trip, by adding up the budgets of all activities, e.g. $10."
    )
    additional_tips: str = Field(
        description="Additional tips for the trip, such as packing advice or transportation details"
    )
    days: List[Day] = Field(description="List of days with activities planned")


# Function to convert TripItinerary object back to string with original formatting
def itinerary_to_string(itinerary: TripItinerary) -> str:
    result = []
    result.append(f"### {itinerary.trip_title}")
    result.append(f"Start Date: {itinerary.start_date}")
    result.append(f"End Date: {itinerary.end_date}")
    if itinerary.total_budget:
        result.append(f"Total Budget: {itinerary.total_budget}")
    if itinerary.additional_tips:
        result.append(f"Additional Tips: {itinerary.additional_tips}")
    result.append("")

    for day in itinerary.days:
        result.append(f"### {day.day_title}")
        for activity in day.activities:
            result.append(f"- **{activity.time_of_day}**")
            result.append(f"  - **{activity.title}** ({activity.location})")
            result.append(f"    - Duration: {activity.details.duration}")
            result.append(f"    - Agenda: {activity.details.agenda}")
            result.append(f"    - Budget: {activity.details.budget}")
        result.append("")

    return "\n".join(result).strip()


SYSTEM_MESSAGE_TEMPLATE = (
    "From now on, you are an excellent tour guide of" " {location}, living in {location} for 20 years."
)

RECOMMENDATION_PROMPT = """
**Trip Attraction Recommendations Request**

I am planning a trip and need detailed attraction recommendations based on the following preferences:

### Trip Preferences
- **Location**: {location}
- **Group Size**: {people_count} people
- **Budget**: {budget} in USD
- **Interests**: {interests}

---

### Recommendations
Please provide:
1. {n_recommendations} aligned recommendations** matching my interests.
2. Prioritize must-visits categories and order the recommendations based on their relevance to my interests and their popularities.
3. Make sure each recommendation's budget is within my budget.
4. Make sure each recommendation's duration is within my trip duration.
5. Make sure each recommendation's peopleCount is suitable for my group size.
6. Mare sure each recommendation is a concrete attraction place, not a general activity.
"""

ITINERARY_PROMPT = """
Given my trip preferences and the attraction candidates, please provide a detailed itinerary for my trip.
### Respond using the {function_name} function.
### Trip Preferences
- **Duration**: {trip_days} days
- **Group Size**: {people_count} people
- **Location**: {location}
- **Budget**: {budget} in USD
### Attractions Candidates
{recommendations}
"""


tools = [
    {
        "type": "function",
        "function": {
            "name": "schedule_events",
            "description": "Schedule events using a greedy algorithm that maximizes the weighted score of the itinerary.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["score_fn_weights"],
                "properties": {
                    "score_fn_weights": {
                        "type": "object",
                        "required": ["w_xp", "w_count", "w_cost", "w_dur", "w_travel_time", "w_gap"],
                        "properties": {
                            "w_xp": {
                                "type": "number",
                                "description": "Weight for experience value. Higher weight prioritizes events with higher experiential value.",
                            },
                            "w_count": {
                                "type": "number",
                                "description": "Weight for number of events. Higher weight prioritizes more events.",
                            },
                            "w_cost": {
                                "type": "number",
                                "description": "Weight for costs (negative for penalties). Higher weight encourages higher expenses.",
                            },
                            "w_dur": {
                                "type": "number",
                                "description": "Weight for duration (negative for penalties). Higher weight places importance on longer events.",
                            },
                            "w_travel_time": {
                                "type": "number",
                                "description": "Weight for travel time (negative for penalties). Higher weight encourages more travel time.",
                            },
                            "w_gap": {
                                "type": "number",
                                "description": "Weight for gap time (negative for penalties). Higher weight increases the idle time.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]


class State(TypedDict):
    messages: Annotated[list, add_messages]


class TripAgent:

    def __init__(self, chat_openai_factory: ChatOpenAIFactory, google_place_api: GooglePlaceAPI):
        self.chat_openai_factory = chat_openai_factory
        self.google_place_api = google_place_api

    async def get_categories(self, location: str) -> List[str]:
        """Get a list of categories of activities for visitors to a location."""
        prompt_template = ChatPromptTemplate(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                (
                    "human",
                    # Provide only concise category names without details.",
                    f"List 10 most interesting categories of activities for visitors to {location} (e.g. food, sight-seeing).",
                ),
            ]
        )
        recommender = self.chat_openai_factory.create(
            top_p=0.95,
            model_name="gpt-3.5-turbo",
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
    ) -> List[RefinedAttraction]:
        n_recommendations = max(n_recommendations, 3)

        system_message = SYSTEM_MESSAGE_TEMPLATE + (
            " Provide the best possible attractions recommendations for the user based on their preferences."
        )

        prompt_template = ChatPromptTemplate(
            [
                ("system", system_message),
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
                "people_count": trip_preference.people_count,
                "location": trip_preference.location,
                "budget": trip_preference.budget,
                "interests": {", ".join(trip_preference.interests)},
            }
        )
        # verify and augmented with Places API
        refined_attractions = []
        print(f"proposed_attractions: {[i.name for i in recommendations.proposed_attraction]}")
        for proposed_attraction in recommendations.proposed_attraction:
            search_res = self.google_place_api.text_search(f"{proposed_attraction.name} in {trip_preference.location}")
            print(search_res["rating"])
            if search_res:
                search_res = {k: search_res[k] for k in list(RefinedAttraction.model_fields.keys())[5:]}
                refined_attractions.append(RefinedAttraction(**(proposed_attraction.dict() | search_res)))
        return refined_attractions

    async def get_itinerary_with_greedy_scheduler(
        self,
        recommendations: List[RefinedAttraction],
        trip_days: int,
        budget: int,
        start_day: str = "Monday",
        travel_type: str = "driving",
        itinerary_description: str = "",
    ) -> str:
        """Generate a detailed itinerary based on the recommendations."""
        # first convert the recommendations to events
        events = []
        for rec in recommendations:
            opening_hours = parse_regular_opening_hours(rec.regular_opening_hours)
            events.append(
                Event(
                    name=rec.name,
                    id=rec.name,
                    # Convert hours to N slots
                    duration=convert_hours_to_slots(rec.duration),
                    opening_hours=opening_hours,
                    cost=rec.cost,
                    base_exp=rec.rating if rec.rating else 3.0,
                )
            )

        # Build travel time matrix using actual travel times
        locations = [event.name for event in events]
        travel_cost_matrix = {}
        travel_time_matrix = {}
        try:
            raw_travel_time_matrix = get_travel_time_matrix(locations, mode=travel_type)
            # Convert minutes to 30-min slots, rounding to nearest slot
            for (origin, destination), minutes in raw_travel_time_matrix.items():
                travel_time_matrix[(origin, destination)] = convert_minutes_to_slots(minutes)
        except Exception as e:
            logging.warning(f"Could not get travel time matrix: {e}")
        travel_time_matrix = postprocess_travel_time_matrix(events, travel_time_matrix)
        travel_cost_matrix = generate_travel_cost_matrix_from_travel_time_matrix(travel_time_matrix, travel_type)
        travel_time_cost_str = ""
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                travel_time_cost_str += f"{locations[i]} to {locations[j]}: {travel_time_matrix[(locations[i], locations[j])]:.0f} mins, ${travel_cost_matrix[(locations[i], locations[j])]:.0f} \n"

        system_message = """
You are an itinerary expert tasked with optimizing trip itineraries by assigning appropriate weights to various factors affecting the itinerary's score. 
Given a user’s trip description, your goal is to provide weights to the `schedule_events` function.

# Output Format
Provide a set of weights for the `schedule_events` function, ensuring they are tailored to the budget request and trip description provided.
            """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": itinerary_description},
        ]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        itinerary_summary = ""
        weights_dict_hist = []
        for _ in range(3):
            if completion.choices[0].message.tool_calls is None:
                break
            tool_call = completion.choices[0].message.tool_calls[0]
            weights_dict = json.loads(tool_call.function.arguments)
            weights_dict_hist.append(weights_dict)

            scheduler = ItineraryScheduler(
                events=copy.deepcopy(events),
                start_day=Day[start_day.upper()],
                num_days=trip_days,
                total_budget=budget,
                travel_cost_matrix=travel_cost_matrix,
                travel_time_matrix=travel_time_matrix,
                allow_partial_attendance=True,
                largest_waiting_time=convert_hours_to_slots(2),
            )
            itinerary = scheduler.greedy_schedule(**weights_dict)
            event_cost, travel_cost = itinerary.calculate_total_cost()
            travel_time, gap_time = itinerary.calculate_total_travel_time_and_gap_time()
            itinerary_summary = (
                f"Event Cost: {event_cost}, travel Cost: {travel_cost}\n"
                f"Travel Time: {travel_time}, gap Time: {gap_time}"
            )
            messages.append(completion.choices[0].message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": itinerary_summary,
                }  # append result message
            )
            user_feedback_prompt = ""
            empty_days = [
                sum([1 for slot in day if slot is not None]) < itinerary.day_resolution * 2 / 3 * 1 / 4
                for day in itinerary.days
            ]
            if any(empty_days):
                user_feedback_prompt = (
                    "You scheduled too few events."
                    "Please schedule more events by increasing w_xp, w_count, or decrease w_cost, w_dur, w_travel_time.\n"
                )
            if event_cost < sum([event.cost for event in events]) / 2:
                user_feedback_prompt = (
                    "You spent too little money."
                    "Please schedule more events by making w_cost larger; potentially increasing w_xp or w_count might help.\n"
                )

            messages.append(
                {
                    "role": "user",
                    "content": f"Here is the summary of the itinerary you generated:\n{itinerary_summary}\n"
                    f"Here is the user feedback (can be empty):\n{user_feedback_prompt}"
                    + "Given the itinerary summary and user feedback, "
                    + "if needed,please generate a new set of weights for the `schedule_events` function.",
                }
            )

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            import pdb

            pdb.set_trace()
        return str(itinerary)

    async def get_itinerary_with_reflection(
        self,
        recommendations: Union[List[RefinedAttraction] | str],
        trip_preference: TripPreference,
        reflection_num: int = 3,
    ) -> str:
        """Generate a detailed itinerary based on the recommendations."""

        system_message = SYSTEM_MESSAGE_TEMPLATE.format(location=trip_preference.location) + (
            " Generate a detailed itinerary based on the follwing preferences and recommended attractions provided by users."
            " Dont use the attractions outside the recommended attractions unless there is not enough attractions."
            " If the user provides feedback on how to modify the trip, respond with a revised version of your previous attempts."
        )
        if not isinstance(recommendations, str):
            recommendations_str = ""
            for rec in recommendations:
                recommendations_str += str(rec) + "\n"
        else:
            recommendations_str = recommendations
        prompt_template = ChatPromptTemplate(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
            ],
        )
        llm = self.chat_openai_factory.create(
            top_p=0.95,
            model_name="gpt-4o",
        ).with_structured_output(TripItinerary)
        generate = prompt_template | llm

        system_message = SYSTEM_MESSAGE_TEMPLATE.format(location=trip_preference.location) + (
            f" You are assessing an itineray of a trip to {trip_preference.location}."
            " **Checking**"
            " 1. Check any attractions in the itinerary are inside the user's attraction candidates."
            " In addtion, check if the properties of the attractions in the itinerary are the same as them in the user's attraction candidates."
            " Compare each properties side-by-side to ensure they are the same."
            " 2. Check if the total buduget is within the user's preferences."
            " Add the budget of each attraction in the itinerary and check if the total budget is within the user's preferences."
            " 3. Check if the group size is within the user's preferences."
            " 4. Check if the duration is within the user's perferences."
            " 5. Check if the itinerary if it is too packed or too loose."
            " 6. Check if the itinerary is feasible: if one attraction is too far from another, if the user has enough time to visit all attractions."
            " After executing all checkings, generate an overvall critique and recommendations for the user's itinerary."
        )

        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_message,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        llm = self.chat_openai_factory.create(
            temperature=0,
            model_name="gpt-4o",
        )
        reflect = reflection_prompt | llm

        async def generation_node(state: State) -> State:
            messages = await generate.ainvoke(state["messages"])
            return {"messages": [itinerary_to_string(messages)]}

        async def reflection_node(state: State) -> State:
            # Other messages we need to adjust
            cls_map = {"ai": HumanMessage, "human": AIMessage}
            # First message is the original user request. We hold it the same for all nodes
            translated = [state["messages"][0]] + [
                cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
            ]
            res = await reflect.ainvoke(translated)
            # We treat the output of this as human feedback for the generator
            return {"messages": [HumanMessage(content=res.content)]}

        def should_continue(state: State):
            # Larger than 6 because we have one user's request at the beginning.
            if len(state["messages"]) > reflection_num * 2:
                # End after 3 iterations
                return END
            return "reflect"

        builder = StateGraph(State)
        builder.add_node("generate", generation_node)
        builder.add_node("reflect", reflection_node)
        builder.add_edge(START, "generate")
        builder.add_conditional_edges("generate", should_continue)
        builder.add_edge("reflect", "generate")
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        itinerary_request = ITINERARY_PROMPT.format(
            trip_days=trip_preference.trip_days,
            people_count=trip_preference.people_count,
            location=trip_preference.location,
            budget=trip_preference.budget,
            recommendations=recommendations_str,
            function_name="TripItinerary",
        )
        itinerary = await graph.ainvoke(
            input={
                "messages": [HumanMessage(content=itinerary_request)],
            },
            config={"configurable": {"thread_id": "1"}},
        )
        return itinerary["messages"][-1].content
