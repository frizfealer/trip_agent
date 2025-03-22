# Example API Usage

Below are example curl commands to interact with the Trip Agent API.

## Testing the API

Test if the API is running:

```bash
curl -X GET http://localhost:8000/api/py/helloFastApi
```

## Get Categories for a City

```bash
curl -X POST http://localhost:8000/api/py/categories \
  -H "Content-Type: application/json" \
  -d '{"city": "Paris"}'
```

## Get Attraction Recommendations

```bash
curl -X POST http://localhost:8000/api/py/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Paris",
    "interests": ["art", "history", "food"],
    "excluded_recommendations": []
  }'
```

## Generate an Optimized Itinerary

First, get recommendations and then use them to generate an itinerary:

```bash
curl -X POST http://localhost:8000/api/py/greedy-itinerary \
  -H "Content-Type: application/json" \
  -d '{
    "recommendations": [
      {
        "name": "Eiffel Tower",
        "duration": 3,
        "category": "Landmarks",
        "cost": 25,
        "rating": 4.7,
        "regular_opening_hours": "Monday: 9:00 AM - 11:30 PM, Tuesday: 9:00 AM - 11:30 PM, Wednesday: 9:00 AM - 11:30 PM, Thursday: 9:00 AM - 11:30 PM, Friday: 9:00 AM - 11:30 PM, Saturday: 9:00 AM - 11:30 PM, Sunday: 9:00 AM - 11:30 PM",
        "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        "website_uri": "https://www.toureiffel.paris/en",
        "editorial_summary": "Iconic iron tower offering city views",
        "photos": ["https://example.com/eiffel-tower.jpg"]
      },
      {
        "name": "Louvre Museum",
        "duration": 4,
        "category": "Museums",
        "cost": 17,
        "rating": 4.8,
        "regular_opening_hours": "Monday: Closed, Tuesday: 9:00 AM - 6:00 PM, Wednesday: 9:00 AM - 6:00 PM, Thursday: 9:00 AM - 6:00 PM, Friday: 9:00 AM - 9:45 PM, Saturday: 9:00 AM - 6:00 PM, Sunday: 9:00 AM - 6:00 PM",
        "formatted_address": "Rue de Rivoli, 75001 Paris, France",
        "website_uri": "https://www.louvre.fr/en",
        "editorial_summary": "World's largest art museum",
        "photos": ["https://example.com/louvre.jpg"]
      },
      {
        "name": "Notre-Dame Cathedral",
        "duration": 2,
        "category": "Religious Sites",
        "cost": 0,
        "rating": 4.7,
        "regular_opening_hours": "Monday: 8:00 AM - 6:45 PM, Tuesday: 8:00 AM - 6:45 PM, Wednesday: 8:00 AM - 6:45 PM, Thursday: 8:00 AM - 6:45 PM, Friday: 8:00 AM - 6:45 PM, Saturday: 8:00 AM - 6:45 PM, Sunday: 8:00 AM - 6:45 PM",
        "formatted_address": "6 Parvis Notre-Dame - Pl. Jean-Paul II, 75004 Paris, France",
        "website_uri": "https://www.notredamedeparis.fr/en/",
        "editorial_summary": "Historic Gothic cathedral",
        "photos": ["https://example.com/notre-dame.jpg"]
      }
    ],
    "budget": 200,
    "start_day": "Monday",
    "num_days": 3,
    "travel_type": "walking",
    "itinerary_description": "I want to see the main highlights of Paris in 3 days, focusing on art and history. I prefer walking between attractions when possible and would like a relaxed pace."
  }'
```

## Conversation-Based Itinerary Planning

Start a new conversation:

```bash
curl -X POST http://localhost:8000/api/py/itinerary-details-conversation \
  -H "Content-Type: application/json" \
  -d '{}'
```

Continue the conversation with a session ID:

```bash
curl -X POST http://localhost:8000/api/py/itinerary-details-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "messages": [
      {"role": "user", "content": "I want to plan a trip to Paris for 3 days."}
    ]
  }'
```

## Image Proxy

To fetch images through the proxy:

```bash
curl -X POST http://localhost:8000/api/py/proxy-image \
  -H "Content-Type: application/json" \
  -d '{"url": "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=YOUR_PHOTO_REFERENCE"}'
``` 