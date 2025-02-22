import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime

from agent.trip_agent import TripAgent, RefinedAttraction
from agent.trip_preference import TripPreference

@pytest_asyncio.fixture
async def mock_trip_agent():
    chat_openai_factory = Mock()
    google_place_api = Mock()
    return TripAgent(chat_openai_factory, google_place_api)

@pytest.fixture
def sample_attractions():
    """Sample NYC attractions for testing"""
    return [
        RefinedAttraction(
            name="Central Park",
            duration=4,
            category="Sightseeing",
            cost=0,
            time="morning",
            rating=4.5,
            regular_opening_hours="Monday: 6:00 AM – 1:00 AM, Tuesday: 6:00 AM – 1:00 AM, Wednesday: 6:00 AM – 1:00 AM, Thursday: 6:00 AM – 1:00 AM, Friday: 6:00 AM – 1:00 AM, Saturday: 6:00 AM – 1:00 AM, Sunday: 6:00 AM – 1:00 AM",
            formatted_address="New York, NY, USA",
            website_uri="https://www.centralparknyc.org/",
            editorial_summary="Sprawling park with pedestrian paths & ballfields, plus a zoo, carousel, boat rentals & a reservoir.",
            photos=[]
        ),
        RefinedAttraction(
            name="Statue of Liberty & Ellis Island",
            duration=5,
            category="Sightseeing",
            cost=25,
            time="morning",
            rating=4.7,
            regular_opening_hours="Monday: 9:00 AM – 6:30 PM, Tuesday: 9:00 AM – 6:30 PM, Wednesday: 9:00 AM – 6:30 PM, Thursday: 9:00 AM – 6:30 PM, Friday: 9:00 AM – 6:30 PM, Saturday: 9:00 AM – 6:30 PM, Sunday: 9:00 AM – 6:30 PM",
            formatted_address="New York, NY 10004, USA",
            website_uri="https://www.nps.gov/stli/index.htm",
            editorial_summary="Iconic National Monument opened in 1886, offering guided tours & a museum, plus harbor & city views.",
            photos=[]
        ),
        RefinedAttraction(
            name="The Metropolitan Museum of Art",
            duration=4,
            category="Museums",
            cost=30,
            time="morning",
            rating=4.8,
            regular_opening_hours="Monday: 10:00 AM – 5:00 PM, Tuesday: 10:00 AM – 5:00 PM, Wednesday: Closed, Thursday: 10:00 AM – 5:00 PM, Friday: 10:00 AM – 9:00 PM, Saturday: 10:00 AM – 9:00 PM, Sunday: 10:00 AM – 5:00 PM",
            formatted_address="1000 5th Ave, New York, NY 10028, USA",
            website_uri="https://www.metmuseum.org/",
            editorial_summary="A grand setting for one of the world's greatest collections of art, from ancient to contemporary.",
            photos=[]
        )
    ]

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler(mock_trip_agent, sample_attractions):
    """Test generating an itinerary using the greedy scheduler"""
    # Create test trip preferences
    trip_preference = TripPreference(
        trip_days=2,
        people_count=2,
        location="New York",
        budget=200.0,
        interests=["Sightseeing", "Museums"]
    )

    # Call the method
    itinerary_str = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=sample_attractions,
        trip_preference=trip_preference
    )

    # Verify the output is a string
    assert isinstance(itinerary_str, str)
    
    # Verify the itinerary contains attraction names
    for attraction in sample_attractions:
        assert attraction.name in itinerary_str, f"Itinerary should contain {attraction.name}"
    
    # Verify budget information is included
    assert "$" in itinerary_str, "Itinerary should contain budget information"
    
    # Verify basic structure elements are present
    assert "Day" in itinerary_str, "Itinerary should contain day information"
    assert "Morning" in itinerary_str or "Afternoon" in itinerary_str or "Evening" in itinerary_str, \
        "Itinerary should contain time of day information"

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler_empty_recommendations(mock_trip_agent):
    """Test generating an itinerary with no recommendations"""
    trip_preference = TripPreference(
        trip_days=2,
        people_count=2,
        location="New York",
        budget=200.0,
        interests=["Sightseeing", "Museums"]
    )

    itinerary_str = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=[],
        trip_preference=trip_preference
    )

    assert isinstance(itinerary_str, str)
    assert "No attractions" in itinerary_str or itinerary_str.strip() == "", "Empty recommendations should result in empty or 'no attractions' message"

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler_low_budget(mock_trip_agent, sample_attractions):
    """Test generating an itinerary with insufficient budget"""
    trip_preference = TripPreference(
        trip_days=2,
        people_count=2,
        location="New York",
        budget=10.0,  # Very low budget
        interests=["Sightseeing", "Museums"]
    )

    itinerary = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=sample_attractions,
        trip_preference=trip_preference
    )

    # Convert itinerary to string before checking
    itinerary_str = itinerary.gen_schedule_str()

    # Should only include free attractions like Central Park
    assert "Central Park" in itinerary_str, "Should include free attractions"
    assert "Metropolitan Museum of Art" not in itinerary_str, "Should not include paid attractions beyond budget"

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler_single_day(mock_trip_agent, sample_attractions):
    """Test generating an itinerary for a single day"""
    trip_preference = TripPreference(
        trip_days=1,
        people_count=2,
        location="New York",
        budget=200.0,
        interests=["Sightseeing", "Museums"]
    )

    itinerary = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=sample_attractions,
        trip_preference=trip_preference
    )
    itinerary_str = itinerary.gen_schedule_str()
    assert "Day 1" in itinerary_str, "Should include day 1"
    assert "Day 2" not in itinerary_str, "Should not include day 2"
    
    # Total duration of activities should not exceed a day
    activities_included = sum(1 for attraction in sample_attractions if attraction.name in itinerary_str)
    assert activities_included <= 3, "Should not schedule more activities than can fit in a day"

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler_respects_opening_hours(mock_trip_agent):
    """Test that the scheduler respects attraction opening hours"""
    # Create an attraction that's only open on weekends
    weekend_attraction = RefinedAttraction(
        name="Weekend Only Attraction",
        duration=2,
        category="Entertainment",
        cost=20,
        time="morning",
        rating=4.5,
        regular_opening_hours="Monday: Closed, Tuesday: Closed, Wednesday: Closed, Thursday: Closed, Friday: Closed, Saturday: 10:00 AM – 6:00 PM, Sunday: 10:00 AM – 6:00 PM",
        formatted_address="New York, NY",
        website_uri="https://example.com",
        editorial_summary="A weekend only attraction",
        photos=[]
    )

    trip_preference = TripPreference(
        trip_days=2,
        people_count=2,
        location="New York",
        budget=200.0,
        interests=["Entertainment"]
    )

    itinerary_str = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=[weekend_attraction],
        trip_preference=trip_preference
    )

    # The attraction should only be scheduled on weekend days
    if "Weekend Only Attraction" in itinerary_str:
        assert any(day in itinerary_str for day in ["Saturday", "Sunday"]), "Weekend attraction should only be scheduled on weekends"

@pytest.mark.asyncio
async def test_get_itinerary_with_greedy_scheduler_time_preferences(mock_trip_agent):
    """Test that the scheduler respects time of day preferences"""
    morning_attraction = RefinedAttraction(
        name="Morning Activity",
        duration=2,
        category="Activity",
        cost=20,
        time="morning",
        rating=4.5,
        regular_opening_hours="Monday: 6:00 AM – 12:00 PM",
        formatted_address="New York, NY",
        website_uri="https://example.com",
        editorial_summary="A morning activity",
        photos=[]
    )

    evening_attraction = RefinedAttraction(
        name="Evening Activity",
        duration=2,
        category="Activity",
        cost=20,
        time="evening",
        rating=4.5,
        regular_opening_hours="Monday: 6:00 PM – 11:00 PM",
        formatted_address="New York, NY",
        website_uri="https://example.com",
        editorial_summary="An evening activity",
        photos=[]
    )

    trip_preference = TripPreference(
        trip_days=1,
        people_count=2,
        location="New York",
        budget=200.0,
        interests=["Activity"]
    )

    itinerary_str = await mock_trip_agent.get_itinerary_with_greedy_scheduler(
        recommendations=[morning_attraction, evening_attraction],
        trip_preference=trip_preference
    )

    if "Morning Activity" in itinerary_str:
        assert "Morning:" in itinerary_str, "Morning activity should be scheduled in the morning"
    if "Evening Activity" in itinerary_str:
        assert "Evening:" in itinerary_str, "Evening activity should be scheduled in the evening" 