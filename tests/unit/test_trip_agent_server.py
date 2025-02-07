import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from agent.trip_agent_server import app, RefinedAttraction
from agent.trip_agent import TripAgent
from agent.trip_preference import TripPreference
# Create a test client
client = TestClient(app)

# Mock recommendation data
MOCK_RECOMMENDATION = {
    "name": "Test Attraction",
    "description": "A test attraction description",
    "estimated_cost": 50,
    "rating": 4.5,
    "regular_opening_hours": "9 AM - 5 PM",
    "formatted_address": "123 Test St, Test City",
    "website_uri": "https://test-attraction.com",
    "editorial_summary": "A wonderful test attraction",
    "photos": ["photo1.jpg", "photo2.jpg"],
    "duration": 2,
    "peopleCount": 2,
    "cost": 50,
    "category": "Tourist Attraction",
    "agenda": "Visit during morning",
    "time": "09:00"
}

@pytest_asyncio.fixture
async def mock_trip_agent():
    with patch('agent.trip_agent_server.TripAgent') as mock:
        # Create an AsyncMock for get_recommendations method
        mock.return_value.get_recommendations = AsyncMock(
            return_value=[RefinedAttraction(**MOCK_RECOMMENDATION)]
        )
        yield mock

@pytest.mark.asyncio
async def test_get_trip_recommendations(mock_trip_agent):
    # Test data
    test_payload = {
        "city": "Test City",
        "n_recommendations": 1,
        "people_count": 2,
        "budget": 1000,
        "interests": "museums, food"
    }

    # Make the request
    response = client.post("/recommendations", json=test_payload)

    # Assert response status code
    assert response.status_code == 200

    # Assert response content
    recommendations = response.json()
    assert isinstance(recommendations, list)
    assert len(recommendations) == 1

    # Check the first recommendation
    recommendation = recommendations[0]
    assert recommendation["name"] == MOCK_RECOMMENDATION["name"]
    assert recommendation["rating"] == MOCK_RECOMMENDATION["rating"]
    assert recommendation["website_uri"] == MOCK_RECOMMENDATION["website_uri"]

@pytest.mark.asyncio
async def test_get_trip_recommendations_error_handling():
    # Test with missing API keys
    with patch('agent.trip_agent_server.os.getenv', return_value=None):
        response = client.post("/recommendations", json={
            "city": "Test City",
            "n_recommendations": 1,
            "people_count": 2,
            "budget": 1000,
            "interests": "museums, food"
        })
        assert response.status_code == 500
        assert "API key not configured" in response.json()["detail"]

@pytest.mark.asyncio
async def test_invalid_request_payload():
    # Test with missing required fields
    response = client.post("/recommendations", json={
        "city": "Test City"  # Missing other required fields
    })
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_empty_interests(mock_trip_agent):
    # Test with empty interests string
    test_payload = {
        "city": "Test City",
        "n_recommendations": 1,
        "people_count": 2,
        "budget": 1000,
        "interests": ""
    }
    
    response = client.post("/recommendations", json=test_payload)
    assert response.status_code == 200

if __name__ == "__main__":
    pytest.main(["-v"]) 