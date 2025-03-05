import logging
import os
import time
from itertools import islice
from typing import Dict, List

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json'
RATE_LIMIT = 100  # elements per second
CHUNK_SIZE = 10   # max locations per request
ALLOWED_MODES = {'driving', 'walking', 'bicycling', 'transit'}

load_dotenv()


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


def get_travel_time_matrix(locations: List[str], default_time: float = float("nan"), mode: str = 'driving') -> Dict[tuple, float]:
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
                required_delay = calculate_delay(
                    len(origin_chunk), len(dest_chunk))

                # Wait if needed to maintain rate limit
                time_since_last = current_time - last_request_time
                if time_since_last < required_delay:
                    time.sleep(required_delay - time_since_last)

                params = {
                    'origins': '|'.join(origin_chunk),
                    'destinations': '|'.join(dest_chunk),
                    'key': os.getenv('GOOGLE_MAPS_API_KEY'),
                    'mode': mode
                }

                response = requests.get(BASE_URL, params=params)
                last_request_time = time.time()  # Update last request time

                response.raise_for_status()
                data = response.json()
                # Use original location names instead of geocoded addresses
                for i, origin in enumerate(origin_chunk):
                    for j, destination in enumerate(dest_chunk):
                        try:
                            element = data['rows'][i]['elements'][j]
                            if element['status'] == 'OK':
                                duration_seconds = element['duration']['value']
                                duration_minutes = duration_seconds / 60
                                travel_times[(origin, destination)
                                             ] = duration_minutes
                            else:
                                travel_times[(origin, destination)
                                             ] = default_time
                        except (KeyError, IndexError):
                            logger.warning(
                                f"Error getting travel time: {data['status']}")
                            travel_times[(origin, destination)] = default_time

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error getting travel times: {str(e)}")
        # Create a matrix with default times if API call fails
        for origin in locations:
            for destination in locations:
                travel_times[(origin, destination)] = default_time

    return travel_times


if __name__ == "__main__":
    # Test locations in New York City
    test_locations = [
        "Times Square, New York",
        "Central Park, New York",
        "Statue of Liberty",
        "Fake location"
    ]

    print("Testing travel time matrix with locations:", test_locations)
    travel_times = get_travel_time_matrix(test_locations, mode="walking")

    print("\nTravel times between locations:")
    for (origin, dest), time in travel_times.items():
        if time is not None:
            print(f"From {origin} to {dest}: {time:.1f} minutes")
        else:
            print(f"From {origin} to {dest}: Route not found")
