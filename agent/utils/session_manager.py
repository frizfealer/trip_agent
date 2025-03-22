import time
import uuid
from typing import Dict, List, Optional

DEFAULT_EXPIRY_TIME = 3600  # 1 hour


# Session Manager for storing conversation history
class SessionManager:

    def __init__(self, expiry_time=DEFAULT_EXPIRY_TIME):  # Default expiry time: 1 hour
        self.sessions: Dict[str, Dict] = {}
        self.expiry_time = expiry_time

    def create_session(self) -> str:
        """Create a new session and return its ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_accessed": time.time(),
            "messages": [],
            "users_itinerary_details": [],
            "itinerary": {},
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID and update its last_accessed time"""
        session = self.sessions.get(session_id)
        if session:
            # Check if session has expired
            if time.time() - session["last_accessed"] > self.expiry_time:
                self.delete_session(session_id)
                return None

            # Update last accessed time
            session["last_accessed"] = time.time()
            return session
        return None

    def update_session(
        self,
        session_id: str,
        messages: List[dict] = None,
        users_itinerary_details: List[dict] = None,
        itinerary: dict = None,
    ) -> bool:
        """Update a session with new messages and details"""
        session = self.get_session(session_id)
        if not session:
            return False
        if time.time() - session["last_accessed"] > self.expiry_time:
            self.delete_session(session_id)
            return False
        if messages:
            session["messages"] = messages
        if users_itinerary_details:
            session["users_itinerary_details"] = users_itinerary_details
        if itinerary:
            session["itinerary"] = itinerary
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if current_time - session["last_accessed"] > self.expiry_time
        ]

        for session_id in expired_sessions:
            self.delete_session(session_id)
