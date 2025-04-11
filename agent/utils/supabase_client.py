import logging
import os

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

supabase_url: str | None = os.environ.get("SUPABASE_URL")
supabase_key: str | None = os.environ.get("SUPABASE_KEY")

supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Initializes and returns a Supabase client instance.
    Raises ValueError if Supabase URL or Key is not set in environment variables.
    """
    global supabase_client
    if supabase_client:
        return supabase_client

    if not supabase_url:
        logger.error("SUPABASE_URL environment variable not set.")
        raise ValueError("SUPABASE_URL environment variable not set.")
    if not supabase_key:
        logger.error("SUPABASE_KEY environment variable not set.")
        raise ValueError("SUPABASE_KEY environment variable not set.")

    try:
        supabase_client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
        raise ConnectionError(f"Failed to initialize Supabase client: {e}")


# Example usage (optional, for testing)
if __name__ == "__main__":
    try:
        client = get_supabase_client()
        print("Supabase client created successfully.")
        # You could add a simple test query here if needed, e.g.,
        # response = client.table('your_table_name').select('id', count='exact').limit(1).execute()
        # print("Test query response:", response)
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
