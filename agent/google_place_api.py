import json
from math import atan2, cos, radians, sin, sqrt

import requests


def haversine_distance(coord1, coord2):
    """Calculate the haversine distance between two coordinates."""
    R = 6371000  # Earth radius in meters
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class GooglePlaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_coordinates_and_radius(self, city_name):
        """Get coordinates and calculate an appropriate radius for a city using Google Geocoding API."""
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": city_name, "key": self.api_key}
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        response = requests.get(base_url, params=params)
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
                    radius = int(
                        city_diameter / 2
                    )  # Use half the diameter as the radius
                else:
                    radius = (
                        20000  # Default to 20 km radius if bounds are not available
                    )

                coordinates = f"{location['lat']},{location['lng']}"
                return coordinates, radius

        print(f"Error retrieving data for {city_name}: {response.status_code}")
        return None, None


class GooglePlaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_coordinates_and_radius(self, city_name):
        """Get coordinates and calculate an appropriate radius for a city using Google Geocoding API."""
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": city_name, "key": self.api_key}
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        response = requests.get(base_url, params=params)
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
                    radius = int(
                        city_diameter / 2
                    )  # Use half the diameter as the radius
                else:
                    radius = (
                        20000  # Default to 20 km radius if bounds are not available
                    )

                coordinates = f"{location['lat']},{location['lng']}"
                return coordinates, radius

        print(f"Error retrieving data for {city_name}: {response.status_code}")
        return None, None

    def text_search(self, text_query, max_results=20, fields=None) -> list:
        """
        Search for places using the Google Places API.

        Args:
            text_query (str): The search query
            max_results (int): Maximum number of results to return
            fields (str, optional): Comma-separated list of fields to request. If None, uses default fields.
        """
        endpoint = "https://places.googleapis.com/v1/places:searchText"

        if fields is None:
            fields = (
                "nextPageToken,places.displayName,places.formattedAddress,places.rating,"
                "places.websiteUri,places.googleMapsLinks,places.photos,"
                "places.types,places.regularOpeningHours,places.reviews,"
                "places.editorialSummary"
            )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": fields,
        }

        all_places = []
        page_token = None

        while len(all_places) < max_results:
            payload = {
                "textQuery": text_query,
                "maxResultCount": max_results,
                # "rankPreference": "RELEVANCE",
            }
            if page_token:
                payload["pageToken"] = page_token

            response = requests.post(
                endpoint, headers=headers, data=json.dumps(payload)
            )

            if response.status_code == 200:
                result = response.json()
                places = result.get("places", [])
                all_places.extend(places)

                page_token = result.get("nextPageToken")
                if not page_token or not places:
                    break
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                break

        # Limit results to max_results
        all_places = all_places[:max_results]

        # Extract properties for all places
        formatted_places = [
            self._extract_property_from_results(place) for place in all_places
        ]

        return formatted_places[0]

    def _extract_property_from_results(self, place):
        formatted_address = place.get("formattedAddress", "NA")
        rating = float(place.get("rating", "nan"))
        website_uri = place.get("websiteUri", "NA")
        regular_opening_hours = place.get("regularOpeningHours", {}).get(
            "weekdayDescriptions", []
        )
        regular_opening_hours = (
            ", ".join(regular_opening_hours) if regular_opening_hours else "NA"
        )
        types = place.get("types", [])
        # format List[dict], fields: 'name', 'relativePublishTimeDescription', 'rating', 'text', 'originalText',
        # 'authorAttribution', 'publishTime', 'flagContentUri', 'googleMapsUri'
        reviews = place.get("reviews", [])
        # has two fields: "text" and "languageCode"
        place_name = place.get("displayName", {}).get("text", "NA")
        # has two fields: "text" and "languageCode"
        editorial_summary = place.get("editorialSummary", {}).get("text", "NA")
        # Extract photos and format them with width=400px
        photos = []
        if "photos" in place:
            for photo in place["photos"]:
                if "name" in photo:
                    photo_url = f"https://places.googleapis.com/v1/{photo['name']}/media?key={self.api_key}&maxWidthPx=400"
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
