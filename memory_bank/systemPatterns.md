# System Patterns: AI Trip Planner

## Overall Architecture

*   **Client-Server:** A frontend client (Gradio, planned) interacts with a backend API server (FastAPI).
*   **API-Driven Backend:** The FastAPI server exposes endpoints for various functionalities (recommendations, itinerary generation, session management).
*   **Agent-Based Core Logic:** The `TripAgent` class encapsulates the primary intelligence, orchestrating LLM calls, external API interactions, and data processing.
*   **Persistent Session Management:** Server-side sessions (`SessionManager`) maintain state (conversation history, user requirements, current itinerary draft) across multiple requests using Redis for persistence. The frontend manages only the `session_id`.

## Key Workflows

1.  **Conversational Requirement Gathering & Itinerary Drafting (`/api/py/itinerary-details-conversation`):**
    *   Frontend sends request, including `session_id` (if available) and latest user message.
    *   Backend retrieves/creates session from Redis using `session_id`.
    *   Append user message to session's conversation history.
    *   Check if core requirements (city, days) are met in the session.
    *   **If requirements incomplete:**
        *   Call `TripAgent.get_itinerary_inquiry` with history.
        *   LLM (`gpt-4o`) processes conversation, potentially using `extract_trip_details` tool.
        *   Update session with new AI message and extracted details.
        *   Return assistant's response and session ID.
    *   **If requirements complete:**
        *   Call `TripAgent.get_itinerary_draft` with requirements, history, and current itinerary draft from session.
        *   LLM (`gpt-4o`) generates/updates itinerary structure.
        *   LLM uses `get_travel_times` tool, which now calls the cached function (`get_travel_time_matrix_cached`) to retrieve travel durations, checking Redis first and falling back to the external API for missing pairs.
        *   Update session with new AI message and updated itinerary draft.
        *   Return assistant's response, structured itinerary, and session ID.
    *   *(Potential Enhancement: Add endpoint to handle explicit user feedback on itinerary/responses).*

2.  **Recommendation Generation (`/api/py/recommendations`):**
    *   Receive request (city, interests, exclusions). (Stateless, doesn't require session).
    *   Call `TripAgent.get_recommendations`.
    *   LLM (`gpt-4o`) generates initial `ProposedAttraction` list.
    *   `GooglePlaceAPI.batch_text_search` fetches details.
    *   Combine LLM proposals with API details into `RefinedAttraction`.
    *   Map `RefinedAttraction` to frontend-friendly `Experience` model.
    *   Return list of `Experience`.

3.  **Session Handling:**
    *   Frontend stores `session_id` (e.g., in `gr.State`).
    *   Backend `SessionManager` uses `session_id` to interact with Redis (`get`, `setex`, `delete`).
    *   Sessions expire after a defined period (`DEFAULT_EXPIRY_TIME`).

4.  **Feedback Submission (`/api/py/feedback`):**
    *   Frontend sends request including `session_id`, `liked` (Optional[bool]), and `feedback_text` (Optional[str]).
    *   Backend retrieves session data (requirements, conversation, itinerary) from Redis using `session_id`.
    *   Backend initializes Supabase client (`agent/utils/supabase_client.py`).
    *   Backend constructs record matching the `feedback` table schema in Supabase.
    *   Backend inserts the record into the Supabase `feedback` table.
    *   Backend returns success/error status to frontend.

## LLM Interaction Patterns (Current Implementation)

*   **Direct OpenAI Client Usage:** The `TripAgent` currently instantiates and uses the `openai.AsyncOpenAI` client directly for all LLM interactions. There is no active abstraction layer in use.
*   **API Methods Used:** Primarily uses `client.beta.chat.completions.parse`, `client.chat.completions.create`, and `client.responses.create` methods from the `openai` library.
*   **Prompt Engineering:** Specific, detailed prompts guide the LLM for different tasks (recommendations, inquiry, drafting). Prompts incorporate session context (requirements, history).
*   **Tool Use / Function Calling:** LLM uses predefined tools (`extract_trip_details`, `get_travel_times`) defined within `TripAgent` and passed to the OpenAI API.
*   **Structured Output (JSON Schema):** LLM is constrained to produce output matching specific Pydantic models or JSON schemas, often using the `response_format` or `text` parameters in OpenAI API calls.
*   **Configuration:** Relies on the `OPENAI_API_KEY` environment variable for authentication.
*   **(Planned) LLM Provider Abstraction:** Code for an abstraction layer (Strategy Pattern using `LLMProvider` interface for OpenAI, Gemini, Anthropic) exists in `agent/llm_providers/` but is **not currently integrated or used**.

## Data Structures

*   **Pydantic Models:** Extensively used for API request/response validation and structuring data from LLM tools/parsing. **Backend models will need to be updated to support richer data for interactive itinerary elements.** *(New)*
*   **Session Data (Redis):** JSON strings stored in Redis, containing keys like `created_at`, `last_accessed`, `messages` (list of user/AI turns), `users_itinerary_details` (dict), `itinerary` (dict/list).
*   **Cache Data (Redis):** Individual travel times (`origin:destination:mode -> time`) stored as strings with a configurable TTL. Keys use the prefix `travel_time:`.
*   **Exception Handling:** Custom exception classes in `agent/exceptions.py`, with `AppException` as the base class for application-specific errors. FastAPI exception handlers map these to appropriate HTTP responses.

## External API Integration

*   **Google Places API:** Wrapped by `GooglePlaceAPI` class. Used for `batch_text_search` and image proxy.
*   **Travel Time API (likely Google Maps):** Accessed via `get_travel_time_matrix_cached` function (which wraps `get_travel_time_matrix`) and the `get_travel_times` LLM tool. Results are cached in Redis.
*   **LLM API:** Currently interacts only with the **OpenAI API** via the `openai` Python client library (`gpt-4o` model specified).
*   **Supabase API:** Accessed via `supabase-py` client library, initialized in `agent/utils/supabase_client.py`. Used for storing feedback data in the `feedback` table.

## Frontend-Backend Interaction

*   Gradio UI communicates with FastAPI backend via HTTP requests to defined API endpoints.
*   Frontend manages the `session_id` to maintain conversation state with the backend.
*   Backend returns structured data (JSON) for the frontend to display (e.g., AI response text, itinerary data, recommendations). **The structure of itinerary data will be enhanced to support richer UI elements.** *(New)*
*   Frontend includes components to display conversation, itinerary, and allow user input/feedback.
*   Frontend calls the `/api/py/feedback` endpoint to submit user feedback (like/dislike/comment).
*   **Frontend Styling:** Gradio UI will be styled using a combination of built-in themes and custom CSS (`frontend/custom.css`). *(New)*
*   **Frontend Components:** The plan includes using more advanced Gradio component structures (e.g., `gr.Accordion`) for displaying interactive itinerary details, moving beyond simple Markdown where beneficial. *(New)*
