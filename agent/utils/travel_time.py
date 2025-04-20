import logging
import os
import time
from itertools import islice, product
from typing import Dict, List, Set, Tuple

import redis
import requests
from dotenv import load_dotenv

from agent.defaults import DEFAULT_LOCAL_REDIS_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
RATE_LIMIT = 100  # elements per second
CHUNK_SIZE = 10  # max locations per request
ALLOWED_MODES = {"driving", "walking", "bicycling", "transit"}
# Cache TTL (Time To Live) in seconds - configurable via environment variable
TRAVEL_TIME_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
CACHE_PREFIX = "travel_time"

load_dotenv()

# --- Redis Cache Setup ---
try:
    # Reuse Redis connection logic from SessionManager
    redis_url = os.getenv("REDIS_URL", DEFAULT_LOCAL_REDIS_URL)
    # Apply SSL settings based on environment variable
    redis_ssl_enabled = os.getenv("REDIS_SSL_ENABLED", "false").lower() == "true"
    if redis_ssl_enabled:
        if "?" not in redis_url:
            redis_url += "?ssl_cert_reqs=none"
        else:
            redis_url += "&ssl_cert_reqs=none"
    redis_client = redis.from_url(redis_url, decode_responses=True)  # Decode responses to strings
    logger.info("Successfully connected to Redis for travel time caching.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Failed to connect to Redis for caching: {e}. Caching will be disabled.")
    redis_client = None


def _normalize_location(location: str) -> str:
    """Normalize location string for consistent cache keys."""
    return location.strip().lower()


def _get_cache_key(origin: str, destination: str, mode: str) -> str:
    """Generate a standardized cache key."""
    norm_origin = _normalize_location(origin)
    norm_dest = _normalize_location(destination)
    return f"{CACHE_PREFIX}:{mode}:{norm_origin}:{norm_dest}"


def chunk_locations(locations: List[str], chunk_size: int = CHUNK_SIZE):
    """
    Split locations into chunks of specified size.

    Args:
        locations: List of location strings
        chunk_size: Maximum size of each chunk

    Returns:
        Generator yielding chunks of locations
    """
    iterator = iter(locations)
    return iter(lambda: list(islice(iterator, chunk_size)), [])


def calculate_delay(origin_size: int, dest_size: int) -> float:
    """
    Calculate required delay to maintain rate limit.
    Requirements-> elements per Second: The rate limit is 100 elements per second.
    Args:
        origin_size: Number of origins in the request
        dest_size: Number of destinations in the request

    Returns:
        Delay in seconds needed to maintain rate limit
    """
    elements = origin_size * dest_size  # number of elements in this request
    return elements / RATE_LIMIT - 1  # minimum time needed for these elements


def get_travel_time_matrix(
    locations: List[str], default_time: float = 15, mode: str = "driving"
) -> Dict[tuple, float]:
    """
    Get a matrix of travel times between all pairs of locations.
    Processes locations in batches to respect API limits.
    Rate limited to 100 elements per second.

    Args:
        locations: List of location strings (addresses or place names)
        default_time: Default travel time in minutes to use when travel time cannot be determined
        mode: Travel mode - must be one of 'driving', 'walking', 'bicycling', or 'transit'

    Returns:
        Dictionary with (origin, destination) tuples as keys and travel time in minutes as values.
        Uses original location names as keys. Returns default_time minutes when travel time cannot be determined.
        Note the returning pairs does not include the diagonal pairs (origin == destination).
    Raises:
        ValueError: If mode is not one of the allowed values
    """

    if mode not in ALLOWED_MODES:
        raise ValueError(f"Mode must be one of {ALLOWED_MODES}, got '{mode}'")

    travel_times = {}

    try:
        location_chunks = list(chunk_locations(locations))
        last_request_time = 0

        for origin_chunk in location_chunks:
            for dest_chunk in location_chunks:
                # Calculate required delay for rate limiting
                current_time = time.time()
                required_delay = calculate_delay(len(origin_chunk), len(dest_chunk))

                # Wait if needed to maintain rate limit
                time_since_last = current_time - last_request_time
                if time_since_last < required_delay:
                    time.sleep(required_delay - time_since_last)

                params = {
                    "origins": "|".join(origin_chunk),
                    "destinations": "|".join(dest_chunk),
                    "key": os.getenv("GOOGLE_MAPS_API_KEY"),
                    "mode": mode,
                }

                response = requests.get(BASE_URL, params=params)
                last_request_time = time.time()  # Update last request time

                response.raise_for_status()
                data = response.json()
                # Use original location names instead of geocoded addresses
                for i, origin in enumerate(origin_chunk):
                    for j, destination in enumerate(dest_chunk):
                        if origin == destination:
                            continue
                        try:
                            element = data["rows"][i]["elements"][j]
                            if element["status"] == "OK":
                                duration_seconds = element["duration"]["value"]
                                duration_minutes = duration_seconds / 60
                                travel_times[(origin, destination)] = duration_minutes
                            else:
                                travel_times[(origin, destination)] = default_time
                        except (KeyError, IndexError):
                            logger.warning(f"Error getting travel time: {data['status']}")
                            travel_times[(origin, destination)] = default_time

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error getting travel times: {str(e)}")
        # Create a matrix with default times if API call fails
        for origin in locations:
            for destination in locations:
                travel_times[(origin, destination)] = default_time

    return travel_times


# --- Cached Function ---


def get_travel_time_matrix_cached(
    locations: List[str], default_time: float = 15, mode: str = "driving"
) -> Dict[tuple, float]:
    """
    Get a matrix of travel times, utilizing a Redis cache first.
    If times are not in the cache, it calls the original get_travel_time_matrix
    for the missing pairs only.

    Args:
        locations: List of location strings (addresses or place names)
        default_time: Default travel time in minutes to use when travel time cannot be determined
        mode: Travel mode - must be one of 'driving', 'walking', 'bicycling', or 'transit'

    Returns:
        Dictionary with (origin, destination) tuples as keys and travel time in minutes as values.
        Uses original location names as keys. Returns default_time minutes when travel time cannot be determined.

    Raises:
        ValueError: If mode is not one of the allowed values
    """
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Mode must be one of {ALLOWED_MODES}, got '{mode}'")

    if not redis_client:
        logger.warning("Redis client not available. Falling back to non-cached version.")
        return get_travel_time_matrix(locations, default_time, mode)

    # Use original location names for the final dictionary keys
    original_locations = list(locations)  # Keep original casing/spacing
    unique_locations = list(set(original_locations))  # Preserve order while getting unique

    cached_results: Dict[Tuple[str, str], float] = {}
    pairs_to_fetch: List[Tuple[str, str]] = []
    all_required_pairs: List[Tuple[str, str]] = [(i, j) for i, j in product(unique_locations, repeat=2) if i != j]

    try:
        # --- 1. Check Cache ---
        cache_keys = {_get_cache_key(orig, dest, mode): (orig, dest) for orig, dest in all_required_pairs}

        if cache_keys:
            # Use pipeline for efficient bulk GET
            pipe = redis_client.pipeline()
            for key in cache_keys.keys():
                pipe.get(key)
            cached_values = pipe.execute()

            for (key, (orig, dest)), value in zip(cache_keys.items(), cached_values):
                if value is not None:
                    try:
                        cached_results[(orig, dest)] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid cache value for key {key}: {value}. Ignoring.")
                        pairs_to_fetch.append((orig, dest))  # Treat invalid cache as missing
                else:
                    pairs_to_fetch.append((orig, dest))
        else:  # Handle case with zero or one location
            cached_results = {}
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error during cache check: {e}. Falling back to non-cached version for this request.")
        return get_travel_time_matrix(locations, default_time, mode)

    # --- 2. Fetch Missing Pairs (if any) ---
    newly_fetched_results: Dict[Tuple[str, str], float] = {}
    if pairs_to_fetch:
        locations_to_fetch_set: Set[str] = set()
        for orig, dest in pairs_to_fetch:
            locations_to_fetch_set.add(orig)
            locations_to_fetch_set.add(dest)
        locations_to_fetch_list = list(locations_to_fetch_set)

        logger.info(
            f"Cache miss for {len(pairs_to_fetch)} pairs. Fetching from API for {len(locations_to_fetch_list)} unique locations."
        )

        # Call the original function with only the necessary locations
        try:
            # This call handles API interaction, chunking, rate limiting etc.
            raw_api_results = get_travel_time_matrix(
                locations=locations_to_fetch_list, default_time=default_time, mode=mode
            )

            # --- 3. Store New Results in Cache ---
            pipe = redis_client.pipeline()
            results_stored_in_cache = 0
            for (orig, dest), time_val in raw_api_results.items():
                # Only process pairs we originally needed to fetch
                if (orig, dest) in pairs_to_fetch:
                    newly_fetched_results[(orig, dest)] = time_val
                    # Store in cache
                    cache_key = _get_cache_key(orig, dest, mode)
                    try:
                        # Store time as string
                        pipe.set(cache_key, str(time_val), ex=TRAVEL_TIME_CACHE_TTL_SECONDS)
                        results_stored_in_cache += 1
                    except redis.exceptions.RedisError as e:
                        logger.error(f"Redis error storing key {cache_key}: {e}")

            if results_stored_in_cache > 0:
                try:
                    pipe.execute()
                    logger.info(f"Stored {results_stored_in_cache} new travel times in cache.")
                except redis.exceptions.RedisError as e:
                    logger.error(f"Redis pipeline error during cache store: {e}")

        except Exception as e:
            # If the underlying API call fails, we still return what we have from cache
            # but log the error. We don't want a temporary API issue to break everything.
            logger.error(f"Error calling original get_travel_time_matrix for missing pairs: {e}")
            # Fill missing pairs with default time
            for orig, dest in pairs_to_fetch:
                newly_fetched_results[(orig, dest)] = default_time

    # --- 4. Combine Results ---
    final_results = {}
    # Ensure all pairs from the original full list are present
    for orig in original_locations:
        for dest in original_locations:
            if orig == dest:
                continue
            if (orig, dest) in cached_results:
                final_results[(orig, dest)] = cached_results[(orig, dest)]
            elif (orig, dest) in newly_fetched_results:
                final_results[(orig, dest)] = newly_fetched_results[(orig, dest)]
            else:
                # Should ideally not happen if logic is correct, but fallback
                logger.warning(f"Result for ({orig}, {dest}) missing after cache check and API fetch. Using default.")
                final_results[(orig, dest)] = default_time

    return final_results


if __name__ == "__main__":
    # Test locations in New York City
    test_locations = [
        "Hotel in LA",
        "The Getty Center",
        "Los Angeles County Museum of Art",
        "The Grove",
        "Griffith Observatory",
        "Hollywood Restaurant",
        "Universal Studios Hollywood",
        "CityWalk",
        "Warner Bros. Studio Tour",
        "Santa Monica Beach and Pier",
        "Natural History Museum of Los Angeles",
        "The Broad",
        "Olvera Street",
        "Rooftop Restaurant in LA",
    ]
    print("Testing travel time matrix with locations:", test_locations)
    travel_times = get_travel_time_matrix(test_locations, default_time=20, mode="driving")

    print("\nTravel times between locations:")
    for (origin, dest), time in travel_times.items():
        if time is not None:
            print(f"From {origin} to {dest}: {time:.1f} minutes")
        else:
            print(f"From {origin} to {dest}: Route not found")
