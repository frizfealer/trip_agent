# AI Trip Planner

## Motivation

1.  Existing AI planners often ignore critical constraints like travel distance and time between locations.
2.  Many AI planners struggle to fully adhere to specific user requirements (budget, interests, pace).
3.  Text-only itinerary outputs are less intuitive than visual, calendar-style UIs.
4.  Users find it difficult to modify generated itineraries or request alternative recommendations easily.

## Solution

1.  **Constraint-Aware Itinerary Drafting:** Implement an itinerary generation process (`get_itinerary_draft` workflow) that explicitly considers travel time/distance, opening hours, and user-defined requirements.
2.  **Interactive Planning & Feedback:** Provide features for easy substitution of activities, generation of alternative recommendations, and user feedback on AI responses/itineraries within the UI.
3.  **Requirement Adherence:** Utilize LLMs with structured prompting and potentially validation steps to ensure user requirements are consistently met.
4.  **Persistent Session-Based Context:** Manage conversation history (user/AI messages) and user preferences effectively using server-side sessions stored persistently (Redis).

## Core Functionality

*   **Recommendation Generation:** Suggest attractions based on location and user interests, refined with real-world data via Google Places API.
*   **Conversational Requirement Gathering:** Engage users in multi-turn dialogue to collect trip details, storing context in persistent backend sessions.
*   **Itinerary Drafting:** Generate structured, day-by-day itineraries via LLM, incorporating activities, meals, and travel segments, verifying travel times using external APIs.
*   **Session Management:** Maintain user-specific context across interactions using Redis.

## Tech Stack

*   **Frontend:** Gradio (Planned)
*   **Backend Framework:** FastAPI (Python)
*   **Core Logic:** Python
*   **LLM:** OpenAI API (`gpt-4o` currently used), Gemini API (Planned for future integration)
*   **Session Storage:** Redis
*   **External APIs:**
    *   Google Places API (for attraction details, photos)
    *   Google Maps Distance Matrix API (likely, for travel times)
*   **Key Libraries:** `openai`, `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `requests`, `redis`.
