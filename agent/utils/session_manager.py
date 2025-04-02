import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import redis

DEFAULT_EXPIRY_TIME = 3600  # 1 hour
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Session Manager for storing conversation history
class SessionManager:

    def __init__(self, expiry_time=DEFAULT_EXPIRY_TIME):  # Default expiry time: 1 hour
        # Try to get Redis URL from environment, fallback to local Redis if not available
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Only apply SSL settings if using a remote Redis (not localhost)
        if "localhost" not in redis_url:
            # Append SSL parameters to the Redis URL if not already present
            if "?" not in redis_url:
                redis_url += "?ssl_cert_reqs=none"
            else:
                redis_url += "&ssl_cert_reqs=none"

        self.redis = redis.from_url(redis_url)
        self.expiry_time = expiry_time
        self.prefix = "trip_agent_session:"

    def _ensure_serializable(self, obj: Any) -> Any:
        """Ensure the object is JSON serializable by converting special objects to dictionaries"""
        if hasattr(obj, "__dict__"):
            # For objects with __dict__, convert to dictionary
            return self._ensure_serializable(obj.__dict__)
        elif isinstance(obj, dict):
            # Process each key-value pair in dictionaries
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Process each item in lists
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, "model_dump"):
            # For Pydantic models
            return self._ensure_serializable(obj.model_dump())
        else:
            # Return primitive values as is
            return obj

    def create_session(self, content_fields: dict) -> str:
        """Create a new session and return its ID"""
        session_id = str(uuid.uuid4())
        session_data = {
            "created_at": time.time(),
            "last_accessed": time.time(),
        } | content_fields
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

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update a session with any fields that are specified, but only if they already exist in the session"""
        session_key = f"{self.prefix}{session_id}"
        session_data = self.redis.get(session_key)

        if not session_data:
            return False

        # Deserialize JSON data
        session = json.loads(session_data)

        # Update session data for each provided field, but only if the field already exists in the session
        for key, value in kwargs.items():
            if key in session:
                session[key] = self._ensure_serializable(value)

        # Update last accessed time
        session["last_accessed"] = time.time()

        # Save the updated session back to Redis
        self.redis.setex(session_key, self.expiry_time, json.dumps(session))
        logger.info(f"session_id: {session_id}, Session data: {session}")

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
