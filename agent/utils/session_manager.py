import json
import os
import time
import uuid
from typing import Dict, List, Optional

import redis

DEFAULT_EXPIRY_TIME = 3600  # 1 hour


# Session Manager for storing conversation history
class SessionManager:

    def __init__(self, expiry_time=DEFAULT_EXPIRY_TIME):  # Default expiry time: 1 hour
        # Try to get Redis URL from environment, fallback to local Redis if not available
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Configure SSL options for Redis connection
        # Only apply SSL settings if using a remote Redis (not localhost)
        if "localhost" not in redis_url:
            self.redis = redis.from_url(
                redis_url, ssl_params={"verify_mode": None}  # Disable certificate verification
            )
        else:
            self.redis = redis.from_url(redis_url)

        self.expiry_time = expiry_time
        self.prefix = "trip_agent_session:"

    def create_session(self) -> str:
        """Create a new session and return its ID"""
        session_id = str(uuid.uuid4())
        session_data = {
            "created_at": time.time(),
            "last_accessed": time.time(),
            "messages": [],
            "users_itinerary_details": [],
            "itinerary": {},
        }
        # Store session data as JSON string
        self.redis.setex(f"{self.prefix}{session_id}", self.expiry_time, json.dumps(session_data))
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID and update its last_accessed time"""
        session_key = f"{self.prefix}{session_id}"
        session_data = self.redis.get(session_key)

        if not session_data:
            return None

        # Deserialize JSON data
        session = json.loads(session_data)

        # Update last accessed time
        session["last_accessed"] = time.time()

        # Update the session with new last_accessed time and reset expiry
        self.redis.setex(session_key, self.expiry_time, json.dumps(session))

        return session

    def update_session(
        self,
        session_id: str,
        messages: List[dict] = None,
        users_itinerary_details: List[dict] = None,
        itinerary: dict = None,
    ) -> bool:
        """Update a session with new messages and details"""
        session_key = f"{self.prefix}{session_id}"
        session_data = self.redis.get(session_key)

        if not session_data:
            return False

        # Deserialize JSON data
        session = json.loads(session_data)

        # Update session data
        if messages:
            session["messages"] = messages
        if users_itinerary_details:
            session["users_itinerary_details"] = users_itinerary_details
        if itinerary:
            session["itinerary"] = itinerary

        # Update last accessed time
        session["last_accessed"] = time.time()

        # Save the updated session back to Redis
        self.redis.setex(session_key, self.expiry_time, json.dumps(session))

        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID"""
        session_key = f"{self.prefix}{session_id}"
        if self.redis.exists(session_key):
            self.redis.delete(session_key)
            return True
        return False

    def cleanup_expired_sessions(self):
        """
        This method is no longer needed with Redis as it automatically
        handles expiration. Kept for API compatibility.
        """
        pass
