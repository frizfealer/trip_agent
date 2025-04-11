# Tech Context: AI Trip Planner

## Backend

*   **Language:** Python 3.x
*   **Framework:** FastAPI
*   **Web Server:** Uvicorn
*   **LLM Interaction:** Direct usage of the `openai` library (v1.x+).
    *   Client: `openai.AsyncOpenAI` instantiated directly in `TripAgent`.
    *   Model Used: `gpt-4o` (explicitly mentioned in code).
    *   API Methods: Uses `client.beta.chat.completions.parse`, `client.chat.completions.create`, `client.responses.create`.
    *   Features Used: Chat completions, Tool/Function Calling, Structured Output (JSON Schema parsing via `response_format` or `text` parameter).
    *   **(Designed, Not Integrated):** Code for LLM provider abstraction (OpenAI, Gemini, Anthropic) exists in `agent/llm_providers/` but is not currently used.
*   **Session Management & Caching:** Redis is used for both persistent session storage (via `SessionManager`) and caching external API results (specifically, Google Maps travel times in `travel_time.py`). Requires Redis server connection (via `REDIS_URL` env var or default `redis://localhost:6379`). Cache TTL is configurable via `TRAVEL_TIME_CACHE_TTL_SECONDS` (defaults to 1 day).
*   **External API Clients:** `requests` library (for Google APIs, image proxy).
*   **Data Validation/Serialization:** Pydantic
*   **Concurrency:** `asyncio` (FastAPI is async, `AsyncOpenAI` client used).
*   **Dependency Management:** `requirements.txt` (includes `supabase-py`).
*   **Environment Variables:** `python-dotenv` (`.env` file) for API keys (OpenAI, Google Places), `REDIS_URL`, `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `TRAVEL_TIME_CACHE_TTL_SECONDS`.
*   **Logging:** Standard Python `logging` module.
*   **Feedback Storage:** Supabase (PostgreSQL) via `supabase-py`.

## Frontend (Planned)

*   **Framework:** Gradio
*   **Session Handling:** Manages a `session_id` (obtained from backend) using `gr.State` to maintain context across interactions with the backend API.
*   **Feedback Submission:** Calls the backend `/api/py/feedback` endpoint to submit user feedback.

## External Services & APIs

*   **OpenAI API:** Requires an API key (`OPENAI_API_KEY`). Used for core intelligence.
*   **Google Places API:** Requires an API key (`GOOGLE_PLACE_API_KEY`). Used for fetching attraction details and image proxy.
*   **Google Maps Distance Matrix API (Inferred):** Used by `utils/travel_time.py`. Requires an API key (likely the same `GOOGLE_PLACE_API_KEY`). Results are cached in Redis.
*   **Redis:** Required for session storage and travel time caching. Connection configured via `REDIS_URL`.
*   **Supabase:** Used for persistent storage of user feedback. Requires `SUPABASE_URL` and `SUPABASE_KEY`.

## Development Setup

*   Install dependencies: `pip install -r requirements.txt`
*   Set up Redis server (local or remote).
*   Set up a Supabase project and create the `feedback` table.
*   Create a `.env` file with `OPENAI_API_KEY`, `GOOGLE_PLACE_API_KEY`, `REDIS_URL`, `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `TRAVEL_TIME_CACHE_TTL_SECONDS`.
*   Run backend server: `uvicorn agent.trip_agent_server:app --reload --port 8001`.

## Deployment (Hints from Filesystem)

*   **Heroku:** `Procfile`, `app.json`, `runtime.txt`, `deploy_to_heroku.sh` suggest Heroku deployment is considered/used. Requires Redis add-on.
*   **Docker:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `deploy_with_docker.sh` suggest Docker containerization is an option. Requires Redis service.

## Key Libraries & Modules

*   `agent/trip_agent.py`: Core agent logic, LLM interactions (focus on `get_itinerary_draft`).
*   `agent/trip_agent_server.py`: FastAPI application, API endpoints.
*   `agent/utils/google_place_api.py`: Wrapper for Google Places API calls.
*   `agent/utils/travel_time.py`: Functions for fetching travel time matrices (`get_travel_time_matrix`) and a cached version (`get_travel_time_matrix_cached`) using Redis.
*   `agent/utils/session_manager.py`: Redis-backed session storage.
*   `agent/utils/supabase_client.py`: Utility for initializing the Supabase client.
*   `agent/scheduler/`: Modules related to custom itinerary scheduler (considered secondary/experimental).
*   `fastapi`: Web framework.
*   `openai`: OpenAI API client.
*   `pydantic`: Data validation and settings management.
*   `requests`: HTTP requests to external APIs.
*   `redis`: Redis client library (used for sessions and caching).
*   `supabase-py`: Supabase client library.
*   `uvicorn`: ASGI server.
*   `python-dotenv`: Loading environment variables.

## Potential Technical Issues/Considerations

*   **API Key Management:** Secure handling of OpenAI, Google, and Supabase keys/URLs.
*   **Redis Configuration:** Ensuring correct `REDIS_URL` and potentially authentication/SSL for remote Redis instances. Handling potential Redis connection errors gracefully.
*   **Supabase Configuration:** Ensuring correct `SUPABASE_URL` and `SUPABASE_KEY`. Handling potential Supabase connection errors or API failures.
*   **Rate Limiting:** Calls to external APIs (OpenAI, Google, Supabase) might need handling. Caching helps mitigate Google Maps rate limits.
*   **Error Handling:** Robust handling for external API/LLM failures, Redis connection issues, and Supabase operations.
*   **Scalability:** FastAPI and Redis generally scale well, but consider load balancing and Redis cluster/sentinel for very high load (both session and cache traffic). Redis pipelining is used for cache lookups.
*   **Cost:** OpenAI, Google Maps (reduced by caching), and potentially hosted Redis costs.
*   **Cache Management:** Cache uses TTL for expiration. Consider strategies for manual cache invalidation if underlying data changes significantly.
*   **LLM Reliability & Flexibility:** Ensuring consistent quality from OpenAI. Flexibility is currently limited due to direct integration; integrating the designed abstraction layer is needed for provider switching.
*   **Feedback Loop:** Backend API endpoint (`/api/py/feedback`) and Gradio frontend integration are implemented to capture user feedback (like/dislike/comment) and store it in Supabase. Further analysis or use of this feedback data is not yet implemented.
