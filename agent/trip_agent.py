import copy
import json
import logging
import time
from typing import Annotated, List, Optional, Union

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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
from agent.utils.google_place_api import GooglePlaceAPI
from agent.utils.travel_time import get_travel_time_matrix

client = AsyncOpenAI()
DEFAULT_CATEGORIES = "Must-visit"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    cost: int = Field(description="Price of the attraction/activity in USD")


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
- **Interests**: {interests}
---
### Recommendations
Please provide:
1. {n_recommendations} aligned recommendations** matching my interests.
2. Prioritize must-visits categories and order the recommendations based on their relevance to my interests and their popularities.
3. Mare sure each recommendation is a concrete attraction place, not a general activity.
4. DO NOT recommend any attractions that are in the following list: {excluded_recommendations}
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
                        "required": ["w_xp", "w_count", "w_cost", "w_dur", "w_travel_time", "w_gap", "max_gap_time"],
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
                            "max_gap_time": {
                                "type": "number",
                                "description": "Maximum gap time allowed between events (in hours). Higher value allows more flexibility. Default is 2 hours.",
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


TRIP_INQUIRY_PROMPT = """
StateAct as an itinerary agent and gather detailed information from customers to help construct their itineraries.

Politely and friendly inquire about the following details for the itinerary:
- City they want to visit
- Total number of days for the itinerary
- Starting day of the itinerary (e.g., Monday)
- Budget
- Number of people
- Any additional information or specific requirements

Encourage the customers to provide as much detail as possible. If customers are reluctant, focus on obtaining at least the city and the number of days for the itinerary. Respect their choice if they prefer not to provide details.

# Steps

1. Greet the customer warmly and introduce yourself as the itinerary agent.
2. Ask about the city they are interested in visiting.
3. Inquire about the total number of days they plan for the itinerary.
4. Request information about the itinerary's starting day.
5. Ask about their budget for the trip.
6. Determine the number of people traveling.
7. Encourage them to share any additional requirements or preferences.
8. If the customer is hesitant, gently remind them that these details help in creating a more tailored itinerary.
9. Respect their decision if they choose not to share information.
11. Once the requirements are gathered, use the extract_itinerary_details` function to generate the itinerary.
"""

TOOLS_FOR_TRIP_INQUIRY = [
    {
        "type": "function",
        "name": "extract_trip_details",
        "description": "Extract the user's trip's details",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City the user wants to visit"},
                "days": {"type": "number", "description": "Total number of days for the trip"},
                "starting_day": {
                    "type": "string",
                    "description": "Starting day of the trip (e.g. Monday, 2025-01-01, This Memorial Day.)",
                },
                "people_count": {"type": "number", "description": "Number of people going on the trip"},
                "budget": {
                    "type": "number",
                    "description": "Total budget for the trip, can be None if not provided",
                },
                "additional_requirements": {
                    "type": "string",
                    "description": "Any additional requirements for the trip; 'None' means no additional requirements",
                },
            },
            "additionalProperties": False,
        },
    }
]

TRIP_ITINERARY_PROMPT = """
Create a short itinerary based on the user's provided requirements. 

user requirements: {user_requirements}
previous itinerary proposed: {previous_itinerary}
Ensure the itinerary format follows the specified day-to-day structure and time slots. 
Consider each requirement carefully to tailor the itinerary specifically to the user's preferences and constraints. 
Apply your 20 years of experience as a trip agent to create an engaging, practical, and enjoyable travel plan. 
Ensure that all travel times between events are verified using `calculate_travel_time` and default values are provided when necessary.

# Steps

1. **Understand Requirements**: Carefully review the user's requirements about the trip. 
2a. **Update the existing itinerary**: If the user has provided an existing itinerary, update the itinerary based on the user's requirements.
2b. **Generate a new itinerary with default Values**: Generate a new itinerary if no previous itinerary is provided. 
    If users do not specified, set travel type to driving. Schedule breakfast, lunch, and dinner at 7am, 12:30pm, and 6:30pm, respectively. 
    Assume each meal duration is one hour if not specified. Assume wake-up time at 6:30am and sleep time at 11:00pm. 
    Don't assume a hotel location. You should ask the user if you need hotel location information.
3. **Plan Daily Activities**: Develop a day-by-day schedule that includes suggested attractions, activities, and sites to visit that align with the user's inputs and fit within their specified budget. 
4. **Verify travel times**: Verify travel times to ensure feasibility by calling the `calculate_travel_time()` function.
5. **Include travel events**: Include travel event in the itinerary, this should include the type of travel and travel time.
5. **Finalize the Itinerary**: Ensure the itinerary is coherent, seamless, and offers a balance of activities and rest time suitable for the group size and composition. 


# Notes

- When considering budget constraints, suggest a mix of free activities (like parks) alongside paid attractions.
- If you don't have any information about the travel time, dont show travel time as NA (not available).
- Always check for any special events, holidays, or closures that might affect the availability of certain attractions or activities. 
  Adjust the itinerary accordingly.

# Examples Output:
Day 1: Arrival and Explore Hollywood and Griffith Park

- **6:30am-7:30am**: Breakfast at a local cafe
- **7:30am-8:30am**: Travel to Griffith Observatory. Type: Driving. Travel time: NA because there is no hotel location
- **8:30am-10:00am**: Visit the Hollywood Sign. 
- **10:00am-10:10am**: Travel to Griffith Observatory. Type: Driving. Travel time is ~ 2 minutes to Griffith Observatory
- **10:10am-1:00pm**: Explore Griffith Observatory and enjoy the view of the cityscape.
- **1:30pm-2:30pm**: Lunch near Griffith Park.
- **2:30pm-3:10pm**: Travel to The Getty Center. Type: Driving. Travel time is ~ 35 minutes to The Getty Center
- **3:10pm-6:00pm**: Visit The Getty Center.
- **6:30pm-7:30pm**: Dinner at a nearby restaurant.
- **7:30pm-8:00pm**: Return to hotel/rest. Type: Driving. Travel time: NA because there is no hotel location
"""

TOOLS_FOR_TRIP_ITINERARY = [
    {
        "type": "function",
        "name": "get_travel_times",
        "description": "Get a matrix of travel times between all pairs of locations. Processes locations in batches to respect API limits. Rate limited to 100 elements per second. Location should be specific, e.g. Empire Buidling in New York City.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["locations", "default_time", "mode"],
            "properties": {
                "locations": {
                    "type": "array",
                    "description": "List of location strings (addresses or place names)",
                    "items": {
                        "type": "string",
                        "description": "A location string representing an address or place name",
                    },
                },
                "default_time": {
                    "type": "number",
                    "description": "Default travel time in minutes to use when travel time cannot be determined",
                },
                "mode": {
                    "type": "string",
                    "description": "Travel mode - must be one of 'driving', 'walking', 'bicycling', or 'transit'",
                    "enum": ["driving", "walking", "bicycling", "transit"],
                },
            },
            "additionalProperties": False,
        },
    }
]


class TripDetails(BaseModel):
    """Trip details extracted from user conversation"""

    city: str
    num_days: int
    start_date: Optional[str] = None
    budget: Optional[int] = None
    num_people: Optional[int] = None
    interests: Optional[List[str]] = None
    additional_requirements: Optional[str] = None


class TripAgent:

    def __init__(self, google_place_api: GooglePlaceAPI):
        self.google_place_api = google_place_api

    async def get_categories(self, location: str) -> List[str]:
        """Get a list of categories of activities for visitors to a location."""

        completion = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE_TEMPLATE.format(location=location)},
                {
                    "role": "user",
                    "content": f"List 10 most interesting categories of activities for visitors to {location} (e.g. food, sight-seeing).",
                },
            ],
            top_p=0.95,
            response_format=Categories,
        )
        categories = completion.choices[0].message.parsed
        return categories.categories

    async def get_recommendations(
        self,
        location: str,
        n_recommendations: int,
        trip_preference: TripPreference,
        excluded_recommendations: List[str],
    ) -> List[RefinedAttraction]:
        start_time = time.time()
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE_TEMPLATE.format(location=location)},
                {
                    "role": "user",
                    "content": RECOMMENDATION_PROMPT.format(
                        n_recommendations=n_recommendations,
                        interests=trip_preference,
                        excluded_recommendations=excluded_recommendations,
                    ),
                },
            ],
            top_p=0.95,
            response_format=TripRecommendations,
        )
        end_time = time.time()
        logger.info(f"Time taken to generate recommendations: {end_time - start_time} seconds")
        recommendations = completion.choices[0].message.parsed
        # verify and augmented with Places API
        refined_attractions = []
        queries = [
            f"{proposed_attraction.name} in {location}" for proposed_attraction in recommendations.proposed_attraction
        ]
        start_time = time.time()
        all_search_res = await self.google_place_api.batch_text_search(queries)
        end_time = time.time()
        logger.info(f"Time taken to refine attractions: {end_time - start_time} seconds")
        for proposed_attraction, query in zip(recommendations.proposed_attraction, queries):
            if query in all_search_res and len(all_search_res[query]) > 0:
                search_res = all_search_res[query][0]
                search_res = {k: search_res[k] for k in list(RefinedAttraction.model_fields.keys())[4:]}
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
Given a user's trip description, your goal is to provide weights to the `schedule_events` function.

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
            max_gap_time = weights_dict["score_fn_weights"]["max_gap_time"]
            weights_dict["score_fn_weights"].pop("max_gap_time")
            scheduler = ItineraryScheduler(
                events=copy.deepcopy(events),
                start_day=Day[start_day.upper()],
                num_days=trip_days,
                total_budget=budget,
                travel_cost_matrix=travel_cost_matrix,
                travel_time_matrix=travel_time_matrix,
                allow_partial_attendance=True,
                largest_waiting_time=max_gap_time,
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
            logger.info(weights_dict)
            logger.info(itinerary_summary)
            logger.info(user_feedback_prompt)
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
        logger.info(weights_dict_hist)
        return str(itinerary)

    async def get_itinerary_inquiry(self, messages=[]):
        """
        Interact with the user through multiple rounds to gather trip details.

        Args:
            messages: List of previous conversation messages. If None, starts a new conversation.

        Returns:
            Dict containing both the assistant's response and the updated messages list
        """
        if messages == []:
            # Start a new conversation
            messages = [
                {"role": "system", "content": TRIP_INQUIRY_PROMPT},
                {
                    "role": "assistant",
                    "content": "Hello! I'm your travel itinerary assistant. To help create your perfect trip plan, I'll need some details. Where would you like to travel to?",
                },
            ]
            return {"response": messages[-1]["content"], "messages": messages}

        # Continue an existing conversation
        # Get the assistant's response
        response = await client.responses.create(
            model="gpt-4o",
            input=messages,
            tools=TOOLS_FOR_TRIP_INQUIRY,
        )

        # Update the messages with the assistant's response
        updated_messages = messages.copy()
        assistant_message = ""
        users_itinerary_details = {}

        if response.output[0].type == "function_call":
            # Extract the function call details
            func_call = response.output[0]
            users_itinerary_details = func_call.arguments

            # Add a dictionary representation of the function call
            updated_messages.append(func_call)
            # Add the function result
            updated_messages.append(
                {  # append result message
                    "type": "function_call_output",
                    "call_id": func_call.call_id,
                    "output": json.dumps({"status": "success"}),
                }
            )

            # Get the response after function call
            response_after_function_call = await client.responses.create(
                model="gpt-4o",
                input=updated_messages,
                tools=TOOLS_FOR_TRIP_INQUIRY,
            )

            assistant_message = response_after_function_call.output[0].content[0].text
            updated_messages.append({"role": "assistant", "content": assistant_message})

        elif response.output[0].type == "message":
            assistant_message = response.output[0].content[0].text
            updated_messages.append({"role": "assistant", "content": assistant_message})

        # Ensure we have valid itinerary details or an empty placeholder
        try:
            parsed_details = json.loads(users_itinerary_details) if users_itinerary_details else {}
        except json.JSONDecodeError:
            parsed_details = {}
        return {
            "response": assistant_message,
            "messages": updated_messages,
            "users_itinerary_details": [parsed_details] if parsed_details else [],
        }

    async def get_itinerary_draft(self, itinerary_requirements: dict, itinerary: dict = {}, messages=[]):
        """
        Genearte itineary draft based on itineary requirements and previous itinerary if provided.
        """
        if itinerary_requirements["city"] is None or itinerary_requirements["days"] is None:
            return {"status": "error", "message": "Please provide the city and days of the trip."}
        else:
            output_format = {
                "format": {
                    "type": "json_schema",
                    "name": "itinerary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "The response to the user's request related to the proposed itinerary.",
                            },
                            # "text": {
                            #     "type": "string",
                            #     "description": "The input text containing the itinerary to be extracted.",
                            # },
                            "itinerary": {
                                "type": "array",
                                "description": "The proposed itinerary items for each day",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Day": {
                                            "type": "number",
                                            "description": "The day number of the itinerary",
                                        },
                                        "day-description": {
                                            "type": "string",
                                            "description": "Short description of the day-itinerary",
                                        },
                                        "day-itinerary": {
                                            "type": "array",
                                            "description": "List of activities and events for the day",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "time": {
                                                        "type": "string",
                                                        "description": "Time period for the activity",
                                                    },
                                                    "title": {
                                                        "type": "string",
                                                        "description": "Title of the activity or event",
                                                    },
                                                    "type": {
                                                        "type": "string",
                                                        "description": "Type of the activity (e.g. event, commute:driving, comute:biking, commute:walking, commute:transit)",
                                                    },
                                                    "estimated_travel_time": {
                                                        "type": ["number", "null"],
                                                        "description": "Estimated travel time if the activity is a commute.",
                                                    },
                                                },
                                                "required": ["time", "title", "type", "estimated_travel_time"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["Day", "day-description", "day-itinerary"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["response", "itinerary"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            }

            messages_with_context = [
                {
                    "role": "system",
                    "content": TRIP_ITINERARY_PROMPT.format(
                        user_requirements=itinerary_requirements, previous_itinerary=itinerary
                    ),
                },
            ]
            messages_with_context.extend(messages)
            start_time = time.time()
            response = await client.responses.create(
                model="gpt-4o",
                input=messages_with_context,
                tools=TOOLS_FOR_TRIP_ITINERARY,
                text=output_format,
                temperature=0,
            )
            end_time = time.time()
            logger.info(f"Time taken to generate itinerary draft: {end_time - start_time} seconds")
        while response.output[0].type == "function_call":
            func_call = response.output[0]
            arguments = json.loads(func_call.arguments)
            logger.info(f"Function call: {func_call}")
            travel_time_matrix = get_travel_time_matrix(
                locations=arguments["locations"], default_time=arguments["default_time"], mode=arguments["mode"]
            )
            messages_with_context.append(func_call)
            # Add the function result
            messages_with_context.append(
                {  # append result message
                    "type": "function_call_output",
                    "call_id": func_call.call_id,
                    "output": str(travel_time_matrix),
                }
            )
            start_time = time.time()
            response = await client.responses.create(
                model="gpt-4o",
                input=messages_with_context,
                tools=TOOLS_FOR_TRIP_ITINERARY,
                text=output_format,
                temperature=0,
            )
            end_time = time.time()
            logger.info(f"Time taken to generate itinerary draft: {end_time - start_time} seconds")
        # response = json.loads(response.output[0].content[0].text)
        response = json.loads(response.output[0].content[0].text)
        return {"itinerary": response["itinerary"], "response": response["response"]}
