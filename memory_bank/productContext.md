# Product Context: AI Trip Planner

## Problem Space

Planning trips, especially multi-day itineraries in unfamiliar locations, is time-consuming and complex. Users struggle with:

* Finding relevant attractions matching their specific interests and budget.
* Optimizing the sequence of activities to minimize travel time and avoid backtracking.
* Accounting for real-world constraints like opening hours, travel duration, and budget limits.
* Visualizing the flow of the trip.
* Making adjustments to plans easily and providing feedback on suggestions.

Existing AI planners often provide generic suggestions without deeply considering these practical constraints or user preferences, leading to unrealistic or unsatisfying itineraries.

## Vision

To create an intelligent, interactive trip planning assistant that generates personalized, feasible, and enjoyable itineraries. The assistant should feel like collaborating with an experienced local tour guide who understands the user's needs, the practicalities of travel, and incorporates user feedback, presented through a **modern and polished user interface.** *(Updated)*

## Goals

*   **Personalization:** Generate itineraries tailored to user interests, budget, pace, and group size.
*   **Feasibility:** Ensure itineraries are realistic by incorporating travel times, opening hours, and activity durations.
*   **Efficiency:** Optimize routes and schedules to make the most of the user's time.
*   **Interactivity & Feedback:** Allow users to easily refine requirements, request alternative suggestions, modify the generated plan, and provide feedback on AI responses and itinerary items.
*   **Clarity:** Present the itinerary in an easy-to-understand format (initially text-based via API, planned visual UI via Gradio).
    *   **UI Polish:** Strive for a modern, intuitive, and visually appealing user interface. *(New)*
*   **Persistence:** Maintain user progress across sessions via persistent backend storage.

## Target User

Travelers planning trips ranging from a few days to a couple of weeks, who want a structured yet personalized itinerary without spending hours on manual research and planning. This includes solo travelers, couples, families, and small groups.

## How it Should Work (Primary Flow)

1.  **Initiation:** User interacts with the system (via API endpoint, eventually Gradio UI).
2.  **Session Management:** Backend retrieves existing session via ID from Redis or creates a new one if ID is missing/invalid. The session ID is managed by the frontend.
3.  **Requirement Gathering:** The system engages in a conversation (`get_itinerary_inquiry`) to collect key trip details (destination, dates, duration, budget, interests, etc.). Conversation history and extracted details are stored in the Redis session.
4.  **(Optional) Recommendation:** User can request specific attraction recommendations (`get_recommendations`). The system uses an LLM and Google Places API to provide relevant, detailed suggestions.
5.  **Itinerary Drafting:** Once minimum requirements (city, days) are met, the system generates/updates a draft itinerary (`get_itinerary_draft`). This involves:
    *   Using an LLM to structure the day-by-day plan based on requirements and conversation history from the session.
    *   Incorporating default meal times and potentially sleep/wake times.
    *   Using a tool (`get_travel_times`) to fetch estimated travel times between locations via an external API.
    *   Structuring the output with activities, locations, timings, and travel segments.
    *   Storing the draft itinerary in the Redis session.
6.  **Refinement & Feedback:** The user interacts with the itinerary (via Gradio UI). They can provide feedback on specific items or the overall plan, or ask for modifications. This feedback/request is sent to the backend.
7.  **Backend Update:** The backend receives the feedback/request, updates the conversation history/requirements in the session, potentially re-runs `get_itinerary_draft`, and saves the updated state to Redis.
8.  **Output:** The updated itinerary and AI response are presented to the user in the Gradio UI.
