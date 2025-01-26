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

    def text_search(self, text_query) -> dict:
        # Define the endpoint
        endpoint = "https://places.googleapis.com/v1/places:searchText"

        # Define the headers and field mask
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.websiteUri,places.regularOpeningHours,places.googleMapsLinks",
        }

        # Define the request payload
        payload = {"textQuery": text_query}

        # Make the POST request
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

        # Handle the response
        if response.status_code == 200:
            places = response.json().get("places", [])
            if places:
                return self._extract_property_from_results(places[0])
            else:
                return {}
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {}

    def _extract_property_from_results(self, place):
        formatted_address = place.get("formattedAddress", "NA")
        rating = float(place.get("rating", "nan"))
        website_uri = place.get("websiteUri", "NA")
        regular_opening_hours = place.get("regularOpeningHours", {}).get(
            "weekdayDescriptions", []
        )

        # Properly decode and clean the regular opening hours
        cleaned_regular_hours = (
            "<br />".join(
                [
                    hours.replace("\u202f", " ")
                    .replace("\u2009", " ")
                    .replace("\u2013", "-")
                    for hours in regular_opening_hours
                ]
            )
            if regular_opening_hours
            else "NA"
        )
        google_maps_place_link = place.get("googleMapsLinks", {}).get("placeUri", "NA")
        google_map_photo_uri = place.get("googleMapsLinks", {}).get("photosUri", "NA")

        return {
            "formated_address": formatted_address,
            "rating": rating,
            "website_uri": website_uri,
            "regular_opening_hours": cleaned_regular_hours,
            "google_maps_place_link": google_maps_place_link,
            "google_map_photo_uri": google_map_photo_uri,
        }
