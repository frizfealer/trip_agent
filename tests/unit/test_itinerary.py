import pytest
from agent.scheduler.itinerary import Itinerary, Day, Event

@pytest.fixture
def itinerary():
    """Create a basic itinerary fixture."""
    return Itinerary(start_day=Day.MONDAY, num_days=2)

@pytest.fixture
def test_event1():
    """Create a sample event that's open all day Monday and Tuesday."""
    return Event(
        name="Test Event",
        cost=10.0,
        duration=4,  # 2 hours (4 * 30-minute slots)
        opening_hours={
            Day.MONDAY: (0, 48),  # Open all day
            Day.TUESDAY: (0, 48),
            Day.WEDNESDAY: None,
            Day.THURSDAY: None,
            Day.FRIDAY: None,
            Day.SATURDAY: None,
            Day.SUNDAY: None
        },
        base_exp=100.0,
        bonus_exp=50.0,
        bonus_start=16,  # 8:00
        bonus_end=20,    # 10:00
    )

@pytest.fixture
def test_event2():
    """Create a sample event that's open all day Monday and Tuesday."""
    return Event(
        name="Test Event 2",
        cost=10.0,
        duration=4,
        opening_hours={
            Day.MONDAY: (0, 48),
            Day.TUESDAY: (0, 48),
            Day.WEDNESDAY: None,
            Day.THURSDAY: None,
            Day.FRIDAY: (0, 48),
            Day.SATURDAY: None,
            Day.SUNDAY: None
        },              
        base_exp=100.0,
        bonus_exp=50.0,
    )


def test_copy(itinerary, test_event1):
    """Test that copying an itinerary creates a deep copy with independent data."""
    # Schedule an event in the original itinerary
    itinerary.schedule_event(test_event1, day=0, start_slot=16, duration=test_event1.duration)
    
    # Create a copy
    copied_itinerary = itinerary.copy()
    # Verify the copy has the same initial state
    assert itinerary.scheduled_events.keys() == copied_itinerary.scheduled_events.keys(), \
        "Copied itinerary should have the same scheduled events"
    assert itinerary.days == copied_itinerary.days, \
        "Copied itinerary should have the same days"
    # Create a new event to schedule in the copy
    new_event = Event(
        name="Another Test Event",
        cost=20.0,
        duration=2,
        opening_hours={day: (0, 48) for day in Day},
        base_exp=50.0,
        bonus_exp=25.0,
        bonus_start=12,
        bonus_end=16,
    )
    
    # Schedule the new event in the copy only
    copied_itinerary.schedule_event(new_event, day=1, start_slot=12, duration=new_event.duration)
    
    # Verify that the original itinerary is unchanged
    assert len(itinerary.scheduled_events) != len(copied_itinerary.scheduled_events), \
        "Original itinerary should not be affected by changes to the copy"

def test_schedule_event(itinerary, test_event1, test_event2):
    """Test scheduling an event with various conditions."""
    # Test successful scheduling
    success = itinerary.schedule_event(test_event1, day=0, start_slot=16, duration=test_event1.duration)
    assert success, "Event should be scheduled successfully"
    assert test_event1.id in itinerary.scheduled_events
    
    # Test scheduling on the wrong day
    success = itinerary.schedule_event(test_event1, day=3, start_slot=16, duration=test_event1.duration)
    assert not success, "Should not be able to schedule event on day 3"

    # Test scheduling same event twice (should fail)
    success = itinerary.schedule_event(test_event1, day=0, start_slot=20, duration=test_event1.duration)
    assert not success, "Should not be able to schedule the same event twice"

    # Test scheduling another event on a valid day/time (at a different time slot)
    success = itinerary.schedule_event(test_event2, day=2, start_slot=24, duration=test_event2.duration)  # Using slot 24 (12:00) instead of 16
    assert not success, "Should not be able to schedule event2 on day 2 (Wednesday) at an available time slot"
    
    # Test scheduling outside operating hours
    closed_day_event = Event(
        name="Closed Event",
        cost=10.0,
        duration=4,
        opening_hours={
            Day.MONDAY: None,  # Closed on Monday
            Day.TUESDAY: (0, 48),
            Day.WEDNESDAY: None,
            Day.THURSDAY: None,
            Day.FRIDAY: None,
            Day.SATURDAY: None,
            Day.SUNDAY: None
        },
        base_exp=100.0,
        bonus_exp=50.0,
        bonus_start=16,
        bonus_end=20,
    )
    
    success = itinerary.schedule_event(closed_day_event, day=0, start_slot=16, duration=closed_day_event.duration)
    assert not success, "Should not be able to schedule event on closed day"


def test_event_id_uniqueness():
    """Test that each event gets a unique ID."""
    opening_hours = {Day.MONDAY: (18, 36)}
    
    event1 = Event(
        name="Event 1",
        cost=10.0,
        duration=2,
        opening_hours=opening_hours,
        base_exp=100.0,
    )
    
    event2 = Event(
        name="Event 1",
        cost=10.0,
        duration=2,
        opening_hours=opening_hours,
        base_exp=100.0,
    )
    
    # default id is generated from uuid; so the two events should have different ids
    assert event1.id != event2.id 

def test_get_event_max_duration():
    # Create a basic itinerary starting on Monday
    itinerary = Itinerary(start_day=Day.MONDAY, num_days=1)
    
    # Create an event that's open all day Monday (0-48 slots)
    event = Event(
        name="Test Event",
        cost=10.0,
        duration=8,  # 4 hours
        opening_hours={Day.MONDAY: (0, 48)},
        base_exp=1.0
    )
    
    # Test 1: Normal case - should return full duration when nothing blocks it
    max_duration = itinerary.get_event_max_duration(event, day=0, start_slot=0)
    assert max_duration == 8, "Should return full duration when no constraints"
    
    # Test 2: Limited by closing time
    event_closing_time = Event(
        name="Event with closing time",
        cost=10.0,
        duration=10,  # 5 hours
        opening_hours={Day.MONDAY: (0, 6)},  # Opens 0:00, closes 3:00
        base_exp=1.0
    )
    max_duration = itinerary.get_event_max_duration(event_closing_time, day=0, start_slot=2)
    assert max_duration == 4, "Should be limited by closing time"
    
    # Test 3: Limited by another scheduled event
    # First schedule an event
    blocking_event = Event(
        name="Blocking Event",
        cost=10.0,
        duration=4,
        opening_hours={Day.MONDAY: (0, 48)},
        base_exp=1.0
    )
    itinerary.schedule_event(blocking_event, day=0, start_slot=10, duration=blocking_event.duration)
    
    # Try to get max duration for a slot that would run into the blocking event
    max_duration = itinerary.get_event_max_duration(event, day=0, start_slot=8)
    assert max_duration == 2, "Should be limited by blocking event" 