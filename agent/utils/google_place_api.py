import asyncio
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two coordinates in kilometers.

    Args:
        coord1: Tuple of (latitude, longitude) for the first point
        coord2: Tuple of (latitude, longitude) for the second point

    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


class GooglePlaceAPI:
    """
    A client for the Google Places API (New).

    This class provides methods to interact with various endpoints of the Google Places API,
    including text search, nearby search, place details, and autocomplete.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Google Places API client.

        Args:
            api_key: Your Google API key with Places API access enabled
        """
        self.api_key = api_key
        self.base_url = "https://places.googleapis.com/v1"
        self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"

        # Default fields to request for place details
        self.default_fields = (
            "places.displayName,places.formattedAddress,places.rating,"
            "places.websiteUri,places.googleMapsLinks,places.photos,"
            "places.types,places.regularOpeningHours,places.reviews,"
            "places.editorialSummary"
        )

        # Rate limiting parameters
        self.max_concurrent_requests = 5
        self.request_delay = 0.1  # 100ms between requests

    def get_coordinates_and_radius(self, city_name: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Get coordinates and calculate an appropriate radius for a city using Google Geocoding API.

        Args:
            city_name: Name of the city to geocode

        Returns:
            Tuple of (coordinates string in "lat,lng" format, radius in meters)
            Returns (None, None) if geocoding fails
        """
        params = {"address": city_name, "key": self.api_key}
        response = requests.get(self.geocoding_url, params=params)

        if response.status_code == 200:
            results = response.json().get("results")
            if results:
                location = results[0]["geometry"]["location"]
                bounds = results[0]["geometry"].get("bounds")

                if bounds:
                    northeast = bounds["northeast"]
                    southwest = bounds["southwest"]

                    coord1 = (northeast["lat"], northeast["lng"])
                    coord2 = (southwest["lat"], southwest["lng"])

                    city_diameter = haversine_distance(coord1, coord2)
                    radius = int(city_diameter * 1000 / 2)  # Convert km to meters and use half the diameter
                else:
                    radius = 20000  # Default to 20 km radius if bounds are not available

                coordinates = f"{location['lat']},{location['lng']}"
                return coordinates, radius

        print(f"Error retrieving data for {city_name}: {response.status_code}")
        return None, None

    def text_search(self, text_query: str, max_results: int = 20, fields: Optional[str] = None) -> List[Dict]:
        """
        Search for places using the Google Places API Text Search endpoint.
        Refer to this doc for details: https://developers.google.com/maps/documentation/places/web-service/reference/rest/v1/places/searchText

        Args:
            text_query: The search query
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to request. If None, uses default fields.

        Returns:
            List of dictionaries containing place information
        """
        endpoint = f"{self.base_url}/places:searchText"

        if fields is None:
            fields = self.default_fields

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": fields,
        }

        all_places = []
        page_token = None

        while len(all_places) < max_results:
            payload = {"textQuery": text_query, "maxResultCount": min(max_results - len(all_places), 20)}
            if page_token:
                payload["pageToken"] = page_token

            try:
                response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise exception for HTTP errors

                result = response.json()
                places = result.get("places", [])
                all_places.extend(places)

                page_token = result.get("nextPageToken")
                if not page_token or not places:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error in text search: {e}")
                if hasattr(e, "response") and e.response is not None:
                    print(f"Response: {e.response.text}")
                break

        # Limit results to max_results
        all_places = all_places[:max_results]

        # Extract properties for all places
        formatted_places = [self._extract_property_from_results(place) for place in all_places]

        return formatted_places

    def nearby_search(
        self,
        location: str,
        radius: int = 1000,
        keyword: Optional[str] = None,
        max_results: int = 20,
        fields: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for places nearby a specific location using the Google Places API.

        Args:
            location: The latitude/longitude around which to retrieve place information (format: "lat,lng")
            radius: Distance in meters within which to search
            keyword: Optional keyword to filter results
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to request. If None, uses default fields.

        Returns:
            List of dictionaries containing place information
        """
        endpoint = f"{self.base_url}/places:searchNearby"

        if fields is None:
            fields = self.default_fields

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": fields,
        }

        all_places = []
        page_token = None

        while len(all_places) < max_results:
            payload = {
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": float(location.split(",")[0]),
                            "longitude": float(location.split(",")[1]),
                        },
                        "radius": radius,
                    }
                },
                "maxResultCount": min(max_results - len(all_places), 20),
            }

            if keyword:
                payload["query"] = keyword

            if page_token:
                payload["pageToken"] = page_token

            try:
                response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
                response.raise_for_status()

                result = response.json()
                places = result.get("places", [])
                all_places.extend(places)

                page_token = result.get("nextPageToken")
                if not page_token or not places:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error in nearby search: {e}")
                if hasattr(e, "response") and e.response:
                    print(f"Response: {e.response.text}")
                break

        # Limit results to max_results
        all_places = all_places[:max_results]

        # Extract properties for all places
        formatted_places = [self._extract_property_from_results(place) for place in all_places]

        return formatted_places

    def get_place_details(self, place_id: str, fields: Optional[str] = None) -> Dict:
        """
        Get detailed information about a specific place using its place ID.

        Args:
            place_id: The place ID to get details for
            fields: Comma-separated list of fields to request. If None, uses default fields.

        Returns:
            Dictionary containing place details
        """
        endpoint = f"{self.base_url}/places/{place_id}"

        if fields is None:
            fields = (
                "displayName,formattedAddress,rating,websiteUri,googleMapsLinks,"
                "photos,types,regularOpeningHours,reviews,editorialSummary"
            )

        headers = {
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": fields,
        }

        try:
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()

            place = response.json()
            return self._extract_property_from_results(place)

        except requests.exceptions.RequestException as e:
            print(f"Error getting place details: {e}")
            if hasattr(e, "response") and e.response:
                print(f"Response: {e.response.text}")
            return {}

    def autocomplete(
        self,
        input_text: str,
        location: Optional[str] = None,
        radius: Optional[int] = None,
        country_codes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Get place autocomplete predictions for a text input.

        Args:
            input_text: The text to get autocomplete predictions for
            location: Optional location bias in "lat,lng" format
            radius: Optional radius in meters for location bias
            country_codes: Optional list of country codes to restrict results

        Returns:
            List of autocomplete prediction dictionaries
        """
        endpoint = f"{self.base_url}/places:autocomplete"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
        }

        payload = {
            "textQuery": input_text,
            "maxResultCount": 5,  # Default to 5 autocomplete suggestions
        }

        # Add location bias if provided
        if location and radius:
            lat, lng = location.split(",")
            payload["locationBias"] = {
                "circle": {"center": {"latitude": float(lat), "longitude": float(lng)}, "radius": radius}
            }

        # Add country restriction if provided
        if country_codes:
            payload["locationRestriction"] = {"countries": country_codes}

        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            predictions = result.get("predictions", [])

            formatted_predictions = []
            for prediction in predictions:
                formatted_predictions.append(
                    {
                        "description": prediction.get("description", ""),
                        "place_id": prediction.get("placeId", ""),
                        "types": prediction.get("types", []),
                    }
                )

            return formatted_predictions

        except requests.exceptions.RequestException as e:
            print(f"Error in autocomplete: {e}")
            if hasattr(e, "response") and e.response:
                print(f"Response: {e.response.text}")
            return []

    def get_place_photo(self, photo_name: str, max_width: int = 400) -> str:
        """
        Get a URL for a place photo.

        Args:
            photo_name: The photo reference name from a place result
            max_width: Maximum width of the photo in pixels

        Returns:
            URL to the photo
        """
        return f"{self.base_url}/{photo_name}/media?key={self.api_key}&maxWidthPx={max_width}"

    def _extract_property_from_results(self, place: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format relevant properties from a place result.

        Args:
            place: The place dictionary from the API response

        Returns:
            Dictionary with formatted place properties
        """
        formatted_address = place.get("formattedAddress", "NA")
        rating = float(place.get("rating", "nan"))
        website_uri = place.get("websiteUri", "NA")

        regular_opening_hours = place.get("regularOpeningHours", {}).get("weekdayDescriptions", [])
        regular_opening_hours = ", ".join(regular_opening_hours) if regular_opening_hours else "NA"

        types = place.get("types", [])
        reviews = place.get("reviews", [])

        place_name = place.get("displayName", {}).get("text", "NA")
        editorial_summary = place.get("editorialSummary", {}).get("text", "NA")

        # Extract photos and format them with width=400px
        photos = []
        if "photos" in place:
            for photo in place["photos"]:
                if "name" in photo:
                    photo_url = self.get_place_photo(photo["name"])
                    photos.append(photo_url)

        return {
            "place_name": place_name,
            "formatted_address": formatted_address,
            "rating": rating,
            "website_uri": website_uri,
            "regular_opening_hours": regular_opening_hours,
            "photos": photos,
            "types": types,
            "reviews": reviews,
            "editorial_summary": editorial_summary,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _async_text_search(
        self, text_query: str, max_results: int = 20, fields: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform an asynchronous text search with pagination and retry logic.

        This function handles:
        1. Making HTTP requests to the Google Places API
        2. Pagination through multiple result pages
        3. Automatic retries with exponential backoff
        4. Formatting and extracting relevant place data

        Args:
            text_query: The search query
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to request

        Returns:
            List of dictionaries containing formatted place information
        """
        if fields is None:
            fields = self.default_fields

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": fields,
        }

        endpoint = f"{self.base_url}/places:searchText"

        async with aiohttp.ClientSession() as session:
            all_places = []
            page_token = None

            while len(all_places) < max_results:
                try:
                    # Prepare request payload
                    payload = {"textQuery": text_query, "maxResultCount": min(max_results - len(all_places), 20)}
                    if page_token:
                        payload["pageToken"] = page_token

                    # Make the API request
                    async with session.post(endpoint, headers=headers, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()

                    # Process results
                    places = result.get("places", [])
                    all_places.extend(places)

                    # Check for more pages
                    page_token = result.get("nextPageToken")
                    if not page_token or not places:
                        break

                    # Add a small delay before fetching the next page to avoid rate limiting
                    await asyncio.sleep(self.request_delay)

                except Exception as e:
                    logger.error(f"Error in async text search: {e}")
                    raise  # Let the retry decorator handle retries

            # Limit results to max_results
            all_places = all_places[:max_results]

            # Extract properties for all places
            formatted_places = [self._extract_property_from_results(place) for place in all_places]

            return formatted_places

    async def batch_text_search(
        self, queries: List[str], max_results_per_query: int = 20, fields: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Perform multiple text searches concurrently with rate limiting.

        This function:
        1. Manages concurrent API requests with rate limiting
        2. Collects results from multiple search queries
        3. Ensures API usage stays within limits

        Args:
            queries: List of search queries to process
            max_results_per_query: Maximum number of results per query
            fields: Comma-separated list of fields to request

        Returns:
            Dictionary mapping each query to its search results
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def search_with_rate_limit(query: str) -> Tuple[str, List[Dict]]:
            async with semaphore:
                results = await self._async_text_search(query, max_results_per_query, fields)
                # Add delay between requests to avoid rate limiting
                await asyncio.sleep(self.request_delay)
                return query, results

        # Create tasks for all queries
        tasks = [search_with_rate_limit(query) for query in queries]

        # Run all tasks concurrently and collect results
        results = {}
        for task in asyncio.as_completed(tasks):
            query, query_results = await task
            results[query] = query_results

        return results

    def batch_text_search_sync(
        self, queries: List[str], max_results_per_query: int = 20, fields: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Synchronous wrapper for the batch_text_search method.

        This function allows using the asynchronous batch search functionality
        from synchronous code by creating and managing the event loop.

        Args:
            queries: List of search queries to process
            max_results_per_query: Maximum number of results per query
            fields: Comma-separated list of fields to request

        Returns:
            Dictionary mapping each query to its search results
        """
        return asyncio.run(self.batch_text_search(queries, max_results_per_query, fields))


def main():
    """
    Test function for the GooglePlaceAPI class.

    To run these tests, set your Google API key as an environment variable:
    export GOOGLE_API_KEY=your_api_key_here

    Then run this script:
    python google_place_api.py
    """
    import os

    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable")
        return

    # Initialize the API client
    client = GooglePlaceAPI(api_key)

    # # Test 1: Basic text search for restaurants in San Francisco
    # print("\n=== Test 1: Basic text search ===")
    # results = client.text_search("restaurants in San Francisco", max_results=3)
    # print(f"Found {len(results)} restaurants in San Francisco")
    # for i, place in enumerate(results, 1):
    #     print(f"\n{i}. {place['place_name']}")
    #     print(f"   Address: {place['formatted_address']}")
    #     print(f"   Rating: {place['rating']}")
    #     print(f"   Types: {', '.join(place['types'][:3])}")

    # # Pause between API calls to avoid rate limiting
    # time.sleep(2)

    # # Test 2: Search with specific fields
    # print("\n=== Test 2: Search with specific fields ===")
    # custom_fields = "places.displayName,places.formattedAddress,places.rating"
    # results = client.text_search("coffee shops in Seattle", max_results=2, fields=custom_fields)
    # print(f"Found {len(results)} coffee shops in Seattle")
    # for i, place in enumerate(results, 1):
    #     print(f"\n{i}. {place['place_name']}")
    #     print(f"   Address: {place['formatted_address']}")
    #     print(f"   Rating: {place['rating']}")

    # time.sleep(2)

    # # Test 3: Search for a specific landmark
    # print("\n=== Test 3: Search for a specific landmark ===")
    # results = client.text_search("Statue of Liberty", max_results=1)
    # if results:
    #     place = results[0]
    #     print(f"Found: {place['place_name']}")
    #     print(f"Address: {place['formatted_address']}")
    #     print(f"Editorial Summary: {place['editorial_summary']}")
    #     if place["photos"]:
    #         print(f"Photo URL: {place['photos'][0]}")
    # else:
    #     print("No results found for Statue of Liberty")

    # time.sleep(2)

    # # Test 4: Search with ambiguous query
    # print("\n=== Test 4: Search with ambiguous query ===")
    # results = client.text_search("Main Street", max_results=3)
    # print(f"Found {len(results)} results for 'Main Street'")
    # for i, place in enumerate(results, 1):
    #     print(f"\n{i}. {place['place_name']}")
    #     print(f"   Address: {place['formatted_address']}")

    # time.sleep(2)

    # Test 5: Test pagination by requesting more results than can be returned in a single request
    print("\n=== Test 5: Test pagination ===")
    start_time = time.time()
    results = client.text_search("restaurants in New York", max_results=25)
    end_time = time.time()

    print(f"Found {len(results)} hotels in Las Vegas")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    if results:
        print(f"First result: {results[0]['place_name']}")

        if len(results) > 20:
            print("\nPagination successful! Received more than 20 results.")
            print(f"Last result: {results[-1]['place_name']}")
        else:
            print("\nPagination may not have worked as expected. Received 20 or fewer results.")

        # Print a sample of results to verify diversity
        print("\nSample of results:")
        sample_indices = [0, len(results) // 4, len(results) // 2, 3 * len(results) // 4, -1]
        for idx in sample_indices:
            if 0 <= idx < len(results):
                print(f"Result #{idx+1}: {results[idx]['place_name']}")
    else:
        print("No results returned. API may be experiencing issues or rate limiting.")
        print("Try again later or check your API key permissions.")

    # Test 6: Test batch text search
    print("\n=== Test 6: Test batch text search ===")
    queries = ["restaurants in New York", "museums in London", "parks in Tokyo", "cafes in Paris", "landmarks in Rome"]

    start_time = time.time()
    batch_results = client.batch_text_search_sync(queries, max_results_per_query=5)
    end_time = time.time()

    print(f"Completed {len(queries)} batch searches in {end_time - start_time:.2f} seconds")

    for query, results in batch_results.items():
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results")
        if results:
            print(f"First result: {results[0]['place_name']}")


if __name__ == "__main__":
    main()
