import argparse
import datetime
import json
import logging
import os

import gradio as gr
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - allow overriding with environment variables
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
TRIP_REQUIREMENTS_ENDPOINT = f"{BACKEND_URL}/api/py/post-trip-requirements"
ITINERARY_DRAFT_ENDPOINT = f"{BACKEND_URL}/api/py/itinerary-draft-conversation"
FEEDBACK_ENDPOINT = f"{BACKEND_URL}/api/py/feedback"  # Added feedback endpoint


def post_trip_requirements(city, days, start_date) -> str:
    """Helper function to post trip requirements to the backend."""
    payload = {
        "city": city,
        "days": days,
        "start_date": start_date,
    }
    try:
        response = requests.post(TRIP_REQUIREMENTS_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data.get("session_id")
    except requests.exceptions.RequestException as e:
        logger.error(f"[Backend] error calling /post-trip-requirements: {e}")
        return ""


def request_itinerary_draft_conversation(session_id=None, user_message=None):
    """Calls the backend API endpoint."""
    payload = {
        "session_id": session_id,
        # Start with an empty message list for the initial call
        "messages": [{"role": "user", "content": user_message}] if user_message else [],
    }
    try:
        response = requests.post(ITINERARY_DRAFT_ENDPOINT, json=payload, timeout=120)  # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        # Return a structured error message for Gradio
        logger.error(f"[Backend] error calling /itinerary-draft-conversation: {e}")
        # Mimic backend response structure for error handling in Gradio
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"[Backend] error calling /itinerary-draft-conversation: {e}")
        return {}


def format_itinerary_md(itinerary_data):
    """Formats the itinerary JSON into Markdown for display."""
    if not itinerary_data:
        return "No itinerary generated yet, or data is invalid."

    # Handle the nested days structure
    if isinstance(itinerary_data, dict):
        itinerary_data = itinerary_data.get("days", [])

    if not isinstance(itinerary_data, list) or not itinerary_data:
        return "No itinerary data available."

    md = "## Trip Itinerary\n\n"
    try:
        for day_data in itinerary_data:
            day_num = day_data.get("Day", "N/A")
            day_desc = day_data.get("day-description", "No description")
            md += f"### Day {day_num}: {day_desc}\n\n"
            md += "| Time          | Activity                     | Type        | Est. Travel Time |\n"
            md += "|---------------|------------------------------|-------------|------------------|\n"
            activities = day_data.get("day-itinerary", [])
            if not activities:
                md += "| *No activities planned* | | | |\n"
            else:
                for activity in activities:
                    time = activity.get("time", "N/A")
                    title = activity.get("title", "N/A")
                    act_type = activity.get("type", "N/A")
                    travel_time = activity.get("estimated_travel_time")
                    travel_time_str = f"{travel_time} min" if travel_time is not None else "N/A"
                    md += f"| {time:<13} | {title:<28} | {act_type:<11} | {travel_time_str:<16} |\n"
            md += "\n"
        return md
    except Exception as e:
        print(f"Error formatting itinerary markdown: {e}")  # Debug print
        return f"Error displaying itinerary: {e}\nRaw data: {itinerary_data}"


# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Trip Planner", css="frontend/custom.css") as demo:
    gr.Markdown("# 🗺️ AI Trip Planner", elem_id="app-main-title")

    # State variables
    session_id_state = gr.State(None)
    chat_history_state = gr.State([])  # To store tuples for gr.Chatbot

    # --- Initial Form View ---
    with gr.Column(visible=True) as form_view:
        gr.Markdown("### Plan Your Trip")
        with gr.Row():
            # TODO: Replace with actual city list if available from backend?
            city_input = gr.Dropdown(
                ["New York", "Tokyo", "Paris", "London", "Rome", "Taipei"], label="Destination City", scale=2
            )
            days_input = gr.Dropdown([1, 2, 3, 4, 5, 6, 7], label="Number of Days", scale=1)
            # Using Textbox for date, Gradio doesn't have a dedicated calendar input yet
            start_date_input = gr.Textbox(
                label="Start Date (YYYY-MM-DD)", placeholder=datetime.date.today().strftime("%Y-%m-%d"), scale=1
            )
        # Add other fields if needed (Budget, People Count - can be asked in chat)
        submit_form_button = gr.Button("Start Planning", variant="primary")

    # --- Main Interaction View (Chat + Itinerary) ---
    with gr.Column(visible=False, elem_id="main_view") as main_view:
        with gr.Row():
            # Chat Column
            with gr.Column(scale=1):
                # Chatbot display
                chatbot_display = gr.Chatbot(label="Conversation", height=600, type="messages")
                with gr.Group(visible=False) as chat_and_feedback_component:
                    chat_input = gr.Textbox(
                        label="Your Message", placeholder="Type your message or requirements here..."
                    )
                    with gr.Row():
                        send_button = gr.Button("➡️ Send", variant="primary")
                        chat_feedback_button = gr.Button("📝 Rate Last Response")  # Added icon
                    reset_button = gr.Button(
                        "🔄 Plan New Trip", variant="secondary"
                    )  # Changed variant for less emphasis

            # Itinerary Column
            with gr.Column(scale=1):
                itinerary_display = gr.Markdown(label="Generated Itinerary")
                with gr.Group(visible=False) as itinerary_feedback_component:
                    # Itinerary Feedback UI
                    with gr.Row():
                        like_button = gr.Button("👍 Like Itinerary")
                        dislike_button = gr.Button("👎 Dislike Itinerary")
                    feedback_text = gr.Textbox(
                        label="Itinerary Feedback/Comments", placeholder="Enter feedback here..."
                    )
                    submit_feedback_button = gr.Button("Submit Feedback")

    # --- Event Handlers ---

    def handle_form_submit(city, days, start_date):
        """
        Called when the initial form is submitted.
        Calls the backend to start a session and get the first response.
        Updates the UI to show the main view.
        """
        logger.info(f"Form submitted: City={city}, Days={days}, Start Date={start_date}")  # Debug print

        if not city or not days or not start_date:
            gr.Warning("Please fill in all fields.")
            # Need to return updates for all outputs of the click event
            return {
                form_view: gr.update(visible=True),
                main_view: gr.update(visible=False),
                session_id_state: None,
                chat_history_state: [],
                chatbot_display: None,
                itinerary_display: "",
            }

        # First update UI to show loading state
        yield {
            form_view: gr.update(visible=False),
            main_view: gr.update(visible=True),
            session_id_state: None,
            chat_history_state: [],
            chatbot_display: [{"role": "assistant", "content": "Generating your itinerary, please wait..."}],
            itinerary_display: "# ⏳ Generating your personalized itinerary...",
            itinerary_feedback_component: gr.update(visible=False),
            chat_and_feedback_component: gr.update(visible=False),
        }

        # get the session_id
        session_id = post_trip_requirements(city, days, start_date)
        logger.info(f"session_id: {session_id}")

        # Check if post_trip_requirements failed (returns empty string on error)
        if not session_id:
            gr.Error("Failed to create a session. Please try again.")
            yield {
                form_view: gr.update(visible=True),
                main_view: gr.update(visible=False),
                session_id_state: None,
                chat_history_state: [],
                chatbot_display: None,
                itinerary_display: "",
            }
            return

        # Get initial response from backend
        draft_response = request_itinerary_draft_conversation(session_id=session_id)

        # Check if the backend call failed
        if not draft_response:
            gr.Error("Failed to get itinerary. Please try again.")
            yield {
                form_view: gr.update(visible=True),
                main_view: gr.update(visible=False),
                session_id_state: None,
                chat_history_state: [],
                chatbot_display: None,
                itinerary_display: "",
            }
            return

        ai_response = draft_response.get("response", "")
        itinerary_md = format_itinerary_md(draft_response.get("itinerary"))
        # Use message dictionaries instead of tuples
        initial_chat_history = [
            {"role": "assistant", "content": ai_response},
        ]

        logger.info(f"Initial Session ID: {session_id}")  # Debug print
        logger.info(f"Initial AI Response: {ai_response}")  # Debug print
        logger.info(f"Initial Itinerary MD:\n{itinerary_md}")  # Debug print

        # Final yield with the actual data
        yield {
            form_view: gr.update(visible=False),
            main_view: gr.update(visible=True),
            session_id_state: session_id,
            chat_history_state: initial_chat_history,
            chatbot_display: initial_chat_history,
            itinerary_display: itinerary_md,
            itinerary_feedback_component: gr.update(visible=True),
            chat_and_feedback_component: gr.update(visible=True),
        }

    def handle_chat_submit(session_id, user_message, current_chat_history):
        """
        Called when the user sends a message in the chat.
        Calls the backend with the session_id and message.
        Updates the chat history and itinerary display.
        """
        print(f"Chat submitted: Session ID={session_id}, Message={user_message}")  # Debug print
        if not user_message:
            gr.Warning("Please enter a message.")
            yield {
                chat_input: "",  # this to return at least one output
            }  # No changes for other outputs
            return

        # waiting for the backend, but need to update the UI
        yield {
            chat_input: gr.Textbox(value=user_message),
            # itinerary_display: "# ⏳ Updating your personalized itinerary...",
            itinerary_feedback_component: gr.update(visible=False),
            chat_and_feedback_component: gr.update(visible=False),
        }

        backend_response = request_itinerary_draft_conversation(session_id=session_id, user_message=user_message)

        if backend_response.get("error"):
            gr.Error(f"Backend Error: {backend_response.get('response', 'Unknown error')}")
            # Keep current state on error
            yield {chat_input: ""}
            return

        ai_response = backend_response.get("response", "Error: No response from AI.")
        itinerary_md = format_itinerary_md(backend_response.get("itinerary"))
        # Use message dictionaries instead of tuples
        new_chat_history = current_chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response},
        ]

        logger.info(f"Updated Chat History: {new_chat_history}")
        logger.info(f"Updated Itinerary MD:\n{itinerary_md}")

        # Return updates for chat input, chatbot, and itinerary
        yield {
            chat_input: gr.Textbox(value=""),
            chatbot_display: new_chat_history,
            itinerary_display: itinerary_md,
            chat_history_state: new_chat_history,  # Update state as well
            itinerary_feedback_component: gr.update(visible=True),
            chat_and_feedback_component: gr.update(visible=True),
        }

    def send_feedback_to_backend(session_id: str, liked: bool | None, feedback_text: str | None):
        """Sends feedback data to the backend API."""
        if not session_id:
            gr.Warning("Cannot submit feedback without an active session.")
            return

        payload = {
            "session_id": session_id,
            "liked": liked,
            "feedback_text": feedback_text if feedback_text else None,  # Ensure null if empty
        }
        logger.info(f"Sending feedback to backend: {payload}")

        try:
            response = requests.post(FEEDBACK_ENDPOINT, json=payload, timeout=30)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            gr.Info("Feedback submitted successfully!")
            logger.info(f"Feedback for session {session_id} submitted successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending feedback for session {session_id}: {e}")
            gr.Error(f"Failed to submit feedback: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending feedback for session {session_id}: {e}")
            gr.Error("An unexpected error occurred while submitting feedback.")

    def handle_reset():
        """Resets the UI to the initial form view."""
        logger.info("Resetting UI")
        # Clear state and switch views
        return {
            form_view: gr.update(visible=True),
            main_view: gr.update(visible=False),
            session_id_state: None,
            chat_history_state: [],
            chatbot_display: None,
            itinerary_display: "",
        }

    # Removed handle_itinerary_feedback placeholder function

    def handle_chat_like(evt: gr.LikeData):
        """Placeholder for handling chatbot message like/dislike."""
        print(f"Chat Message Feedback: Index={evt.index}, Value='{evt.value}', Liked={evt.liked}")
        # TODO: Send feedback to backend API endpoint, potentially including session_id and message index/content
        gr.Info("Chat feedback recorded!")

    def handle_chat_feedback(session_id, chat_history):
        """Handle feedback for the last AI response"""
        if not session_id or not chat_history:
            gr.Warning("No conversation to provide feedback on.")
            return

        # Get the last message pair if possible
        if len(chat_history) > 0:
            # Adjust to get the last assistant message from the dict format
            last_messages = [msg for msg in chat_history if msg["role"] == "assistant"]
            if last_messages:
                last_msg = last_messages[-1]["content"]
                print(f"Feedback requested for message: {last_msg[:50]}...")
                # TODO: Send to backend API
                gr.Info("Thank you for your feedback on the last response!")
            else:
                gr.Warning("No assistant messages to provide feedback on.")
        else:
            gr.Warning("No messages to provide feedback on.")

    # --- Connect Event Handlers ---

    submit_form_button.click(
        fn=handle_form_submit,
        inputs=[city_input, days_input, start_date_input],
        outputs=[
            form_view,
            main_view,
            session_id_state,
            chat_history_state,
            chatbot_display,
            itinerary_display,
            itinerary_feedback_component,
            chat_and_feedback_component,
        ],
    )

    send_button.click(
        fn=handle_chat_submit,
        inputs=[session_id_state, chat_input, chat_history_state],
        outputs=[
            chat_input,
            chatbot_display,
            itinerary_display,
            chat_history_state,
            itinerary_feedback_component,
            chat_and_feedback_component,
        ],
    )

    # Also allow submitting chat with Enter key
    chat_input.submit(
        fn=handle_chat_submit,
        inputs=[session_id_state, chat_input, chat_history_state],
        outputs=[
            chat_input,
            chatbot_display,
            itinerary_display,
            chat_history_state,
            itinerary_feedback_component,
            chat_and_feedback_component,
        ],
    )

    reset_button.click(
        fn=handle_reset,
        inputs=[],
        outputs=[form_view, main_view, session_id_state, chat_history_state, chatbot_display, itinerary_display],
    )

    # Connect itinerary feedback buttons to the new handler
    like_button.click(
        fn=lambda sid: send_feedback_to_backend(sid, liked=True, feedback_text=None),
        inputs=[session_id_state],
        outputs=[],  # No output change needed, feedback function handles notification
    )
    dislike_button.click(
        fn=lambda sid: send_feedback_to_backend(sid, liked=False, feedback_text=None),
        inputs=[session_id_state],
        outputs=[],  # No output change needed
    )
    submit_feedback_button.click(
        fn=lambda sid, comment: send_feedback_to_backend(sid, liked=None, feedback_text=comment),
        inputs=[session_id_state, feedback_text],
        outputs=[feedback_text],  # Clear feedback textbox after submission
    )

    # Connect chat feedback button
    chat_feedback_button.click(
        fn=handle_chat_feedback,
        inputs=[session_id_state, chat_history_state],
        outputs=[],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Gradio Trip Planner app")
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1, use 0.0.0.0 for Docker)"
    )
    parser.add_argument("--server-port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    # Configure for Docker compatibility if needed
    demo.launch(server_name=args.server_name, server_port=args.server_port)
