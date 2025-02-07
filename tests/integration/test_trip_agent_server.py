import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from agent.trip_agent_server import app
from agent.trip_agent import RefinedAttraction

client = TestClient(app)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_trip_recommendations_integration():
    # Mock response data
    mock_recommendations = [
        RefinedAttraction(
            # ProposedAttraction fields
            name="Test Attraction",
            duration=2,  # in hours
            peopleCount=2,
            cost=50.0,
            category="SIGHTSEEING",
            agenda="Day 1",
            time="10:00",
            
            # RefinedAttraction additional fields
            rating=4.5,
            regular_opening_hours="Monday: 9:00 AM - 5:00 PM",
            formatted_address="123 Test Street, Test City, TC 12345",
            website_uri="https://example.com",
            editorial_summary="A wonderful test attraction in the heart of the city",
            photos=["https://example.com/image1.jpg"]
        )
    ]

    # Mock environment variables
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_openai_key',
        'GOOGLE_PLACE_API_KEY': 'test_google_key'
    }):
        # Mock the TripAgent.get_recommendations method
        with patch('agent.trip_agent.TripAgent.get_recommendations', new_callable=AsyncMock) as mock_get_recommendations:
            mock_get_recommendations.return_value = mock_recommendations

            # Test data
            test_payload = {
                "city": "Paris",
                "n_recommendations": 1,
                "people_count": 2,
                "budget": 1000,
                "interests": "museums, art, history"
            }

            # Make request to the endpoint
            response = client.post("/recommendations", json=test_payload)

            # Assertions
            assert response.status_code == 200
            recommendations = response.json()
            assert len(recommendations) == 1
            
            recommendation = recommendations[0]
            # Assert ProposedAttraction fields
            assert recommendation["name"] == "Test Attraction"
            assert recommendation["duration"] == 2
            assert recommendation["peopleCount"] == 2
            assert recommendation["cost"] == 50.0
            assert recommendation["category"] == "SIGHTSEEING"
            assert recommendation["agenda"] == "Day 1"
            assert recommendation["time"] == "10:00"
            
            # Assert RefinedAttraction additional fields
            assert recommendation["rating"] == 4.5
            assert recommendation["regular_opening_hours"] == "Monday: 9:00 AM - 5:00 PM"
            assert recommendation["formatted_address"] == "123 Test Street, Test City, TC 12345"
            assert recommendation["website_uri"] == "https://example.com"
            assert recommendation["editorial_summary"] == "A wonderful test attraction in the heart of the city"
            assert recommendation["photos"] == ["https://example.com/image1.jpg"]

            # Verify the mock was called with correct parameters
            mock_get_recommendations.assert_called_once()
            call_args = mock_get_recommendations.call_args[0]
            assert call_args[0] == 1  # n_recommendations
            preferences = call_args[1]
            assert preferences.location == "Paris"
            assert preferences.people_count == 2
            assert preferences.budget == 1000
            assert preferences.interests == ["museums", "art", "history"]
            assert preferences.trip_days == 7