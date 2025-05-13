# Active Context: AI Trip Planner - Frontend Modernization Planning

## Current Focus

*   **Supabase Feedback Integration:** Implementing Supabase integration to store user feedback (like/dislike/comments) on itineraries. *(Existing)*
*   **Memory Bank Update:** Updating documentation to reflect the new Supabase feedback implementation and frontend modernization plan. *(Updated)*
*   **Frontend UI/UX Modernization (Phase 1 - Gradio):** Planning and preparing for initial visual and layout improvements to the Gradio frontend. *(New)*

## Recent Changes (Implemented)

*   **LLM Provider Code Created:** A new `agent/llm_providers/` directory was created with base interfaces, provider implementations (OpenAI, Gemini, Anthropic), and a factory. **However, this code is NOT currently integrated into `TripAgent` or `TripAgentServer`.**
*   **Travel Time Caching:**
    *   Added `get_travel_time_matrix_cached` function to `agent/utils/travel_time.py`.
    *   This function checks Redis for cached travel times before calling the original `get_travel_time_matrix` for missing pairs.
    *   Uses Redis pipelining for efficient lookups.
    *   Cache TTL is configurable via `TRAVEL_TIME_CACHE_TTL_SECONDS` environment variable (defaults to 1 day).
    *   Updated `TripAgent` (`get_itinerary_draft` and `get_itinerary_with_greedy_scheduler`) to use the cached function.
*   **Direct OpenAI Usage:** `TripAgent` currently uses the `openai` library directly (`AsyncOpenAI` client).
*   **Gradio UI:** Basic UI exists with chat and itinerary display. Feedback buttons are now connected to the backend.
*   **Session Management:** Redis-based session management is functional via `SessionManager`.
*   **Core Endpoints:** Key FastAPI endpoints (`/itinerary-draft-conversation`, `/recommendations`, etc.) are implemented.
*   **Feedback Endpoint:** Added `/api/py/feedback` endpoint to receive user feedback and store it in Supabase.
*   **Supabase Integration:**
    *   Added `supabase-py` dependency.
    *   Created `agent/utils/supabase_client.py` for client initialization.
    *   Integrated feedback submission logic into the backend endpoint.
    *   Connected Gradio UI feedback buttons to the backend endpoint.
*   **Debugger Fix:** A breakpoint was removed from `gradio_app.py`.
*   **Memory Bank Updated:** Documentation (`techContext.md`, `systemPatterns.md`, `activeContext.md`, `progress.md`) updated to reflect the travel time caching and Supabase feedback implementations.

## Next Steps

1.  **Frontend Modernization - Phase 1 (Gradio):** *(New Section)*
    *   **Theme and Basic Styling:**
        *   Experiment with built-in Gradio themes (e.g., `gr.themes.Base()`).
        *   Create and link `frontend/custom.css` for custom fonts, colors, spacing.
    *   **Layout Improvements (Initial Form):**
        *   Use `gr.Row` for compact input layout.
        *   Enhance clarity with placeholders/descriptions.
    *   **Layout Improvements (Main View):**
        *   Improve visual hierarchy (chat vs. itinerary) using CSS.
        *   Refine Markdown table appearance for itinerary.
    *   **Enhanced User Feedback & Interactivity:**
        *   Standardize notification styles (`gr.Info`, `gr.Warning`, `gr.Error`).
        *   Consistent button styling (variants, text-based icons).
2.  **Frontend Modernization - Phase 1.5 (Interactive Itinerary Elements):** *(New Section)*
    *   **Backend API Modification:**
        *   Update `TripAgent.get_itinerary_draft` and Pydantic models to include richer activity data (description, image_url, location_details).
    *   **Frontend Implementation (Gradio):**
        *   Refactor itinerary display to use `gr.Accordion` for days and activities, showing detailed information on expansion.
        *   Update/replace `format_itinerary_md` to generate Gradio components.
3.  **Integrate LLM Provider Abstraction (High Priority):** Refactor `TripAgent` and `TripAgentServer` to utilize the existing `agent/llm_providers/` code (Strategy/Factory patterns). *(Existing)*
4.  **Configure Supabase Environment Variables:** Ensure `SUPABASE_URL` and `SUPABASE_KEY` are added to the `.env` file or environment. *(Existing)*
5.  **Install LLM Dependencies:** Install dependencies for Gemini (`google-generativeai`) and Anthropic (`anthropic`) *when* the abstraction is integrated. *(Existing)*
6.  **Configure LLM Environment Variables:** Set up environment variables for the desired LLM provider *after* integration. *(Existing)*
7.  **Test Integrated LLM Providers:** Test with actual API keys *after* integration. *(Existing)*
8.  **Enhance Error Handling:** Improve error handling for API failures and rate limits, especially for different providers and Supabase. *(Existing)*
9.  **(Optional) Refine Feedback Usage:** Consider how the collected feedback in Supabase might be used (e.g., analysis, future fine-tuning). *(Existing)*

## Active Decisions & Considerations (Current State)

*   **Frontend Enhancement Strategy:** Prioritizing Gradio-native improvements (Phase 1: theming, CSS, layout) followed by Gradio-based interactive elements (Phase 1.5: accordion display for activities, requiring backend data enrichment). *(New)*
*   **Direct OpenAI Integration:** The system currently directly integrates with the OpenAI API via the `openai` library. *(Existing)*
*   **LLM Abstraction (Designed, Not Active):** The Strategy Pattern and Factory for LLM providers are *designed* and code exists, but they are not currently wired into the application flow. *(Existing)*
*   **Session Management & Caching:** Redis is actively used for persistent session state and travel time caching. *(Existing)*
*   **Gradio Frontend:** Gradio provides the user interface, interacting with the FastAPI backend. *(Existing)*
*   **Feedback Mechanism (Implemented):** User feedback (like/dislike/comment) is captured via Gradio UI, sent to a dedicated backend endpoint, and stored persistently in Supabase. *(Existing)*

## Important Patterns & Preferences (Current Implementation)

*   **Direct API Client Usage:** `TripAgent` directly instantiates and uses `AsyncOpenAI`. *(Existing)*
*   **API-Driven Backend:** FastAPI serves as the backend API layer. *(Existing)*
*   **Redis for State & Cache:** Redis is leveraged for session persistence (`SessionManager`) and travel time caching (`travel_time.py`). *(Existing)*
*   **Supabase for Feedback Storage:** Supabase (PostgreSQL) is used via `supabase-py` to store user feedback records. *(Existing)*
*   **Pydantic for Validation:** Used extensively for API models and data structuring. *(Existing)*
*   **Environment-Based Configuration:** Used for API keys/URLs (`OPENAI_API_KEY`, `GOOGLE_PLACE_API_KEY`, `REDIS_URL`, `SUPABASE_URL`, `SUPABASE_KEY`), and `TRAVEL_TIME_CACHE_TTL_SECONDS`. *(Existing)*
*   **(Designed) Strategy/Factory:** Patterns for LLM abstraction exist in `llm_providers` but are not currently applied. *(Existing)*

## Learnings & Insights (From Code Review & Planning)

*   **Documentation Drift:** The Memory Bank documentation had drifted from the actual implementation regarding the LLM provider abstraction. Regular code review is crucial to keep documentation accurate. (Resolved for feedback loop). *(Existing)*
*   **Abstraction Requires Integration:** Creating abstraction layers (`llm_providers`) is only the first step; they must be actively integrated into the application to provide value. *(Existing)*
*   **Feedback Loop Implemented:** The user feedback mechanism (UI -> Backend API -> Supabase Storage) is now functional. *(Existing)*
*   **Direct API Usage:** The current direct use of the `openai` library creates a hard dependency, limiting flexibility. *(Existing)*
*   **Caching Strategy:** Implemented caching for `get_travel_time_matrix` by wrapping the original function and checking Redis first. *(Existing)*
*   **Frontend Modernization Plan:** A phased approach for Gradio UI improvements has been defined, starting with visual enhancements and leading to more interactive elements. *(New)*
