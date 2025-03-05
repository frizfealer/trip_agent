from collections import defaultdict

import pytest

from agent.scheduler.itinerary import Day, Event, Itinerary
from agent.scheduler.itinerary_scheduler import GreedyItineraryScheduler


@pytest.fixture
def sample_events():
    monday_to_friday = {
        Day.MONDAY: (14, 36),    # 7:00-18:00
        Day.TUESDAY: (14, 36),   # 7:00-18:00
        Day.WEDNESDAY: (14, 36), # 7:00-18:00
        Day.THURSDAY: (14, 36),  # 7:00-18:00
        Day.FRIDAY: (14, 36),    # 7:00-18:00
        Day.SATURDAY: None,      # Closed
        Day.SUNDAY: None         # Closed
    }
    
    return [
        Event(
            id="museum",
            name="Museum Visit",
            cost=50,
            duration=4,
            opening_hours=monday_to_friday,
            base_exp=100,
            bonus_exp=20,
            bonus_start=2,
            bonus_end=4
        ),
        Event(
            id="gallery",
            name="Art Gallery",
            cost=20,
            duration=2,
            opening_hours=monday_to_friday,
            base_exp=30,
            bonus_exp=10,
            bonus_start=5,
            bonus_end=6
        )
    ]

@pytest.fixture
def travel_cost_matrix(sample_events):
    return {
        ("hotel", "museum"): 10,
        ("hotel", "gallery"): 15,
        ("museum", "hotel"): 10,
        ("gallery", "hotel"): 15,
        ("museum", "gallery"): 5,
        ("gallery", "museum"): 5,
    }

@pytest.fixture
def travel_time_matrix(sample_events):
    """Fixture for travel time matrix in 30-minute slots"""
    return {
        ("hotel", "museum"): 2,  # 1 hour
        ("hotel", "gallery"): 3,  # 1.5 hours
        ("museum", "hotel"): 2,
        ("gallery", "hotel"): 3,
        ("museum", "gallery"): 1,  # 30 min
        ("gallery", "museum"): 1,
    }

@pytest.fixture
def scheduler(sample_events, travel_cost_matrix, travel_time_matrix):
    return GreedyItineraryScheduler(
        events=sample_events,
        start_day=Day.MONDAY,
        num_days=5,
        total_budget=200,
        travel_cost_matrix=travel_cost_matrix,
        travel_time_matrix=travel_time_matrix,
        score_fn_weights={
            'w_xp': 1.0,
            'w_count': 0.0,
            'w_cost': 0.0,
            'w_travel': 0.0,
            'w_time': 0.0,
            'w_dur': 0.0
        }
    )

@pytest.fixture
def expanded_sample_events():
    monday_to_friday = {
        Day.MONDAY: (14, 36),    # 7:00-18:00
        Day.TUESDAY: (14, 36),   # 7:00-18:00
        Day.WEDNESDAY: (14, 36), # 7:00-18:00
        Day.THURSDAY: (14, 36),  # 7:00-18:00
        Day.FRIDAY: (14, 36),    # 7:00-18:00
        Day.SATURDAY: None,      # Closed
        Day.SUNDAY: None         # Closed
    }
    
    return [
        Event(
            id="museum1",
            name="Museum Visit 1",
            cost=50,
            duration=4,
            opening_hours=monday_to_friday,
            base_exp=100,
            bonus_exp=20,
            bonus_start=2,
            bonus_end=4
        ),
        Event(
            id="museum2",
            name="Museum Visit 2",
            cost=45,
            duration=3,
            opening_hours=monday_to_friday,
            base_exp=80,
            bonus_exp=15,
            bonus_start=2,
            bonus_end=3
        ),
        Event(
            id="gallery1",
            name="Art Gallery 1",
            cost=20,
            duration=2,
            opening_hours=monday_to_friday,
            base_exp=30,
            bonus_exp=10,
            bonus_start=1,
            bonus_end=2
        ),
        Event(
            id="gallery2",
            name="Art Gallery 2",
            cost=25,
            duration=2,
            opening_hours=monday_to_friday,
            base_exp=35,
            bonus_exp=10,
            bonus_start=1,
            bonus_end=2
        )
    ]

@pytest.fixture
def expanded_travel_cost_matrix(expanded_sample_events):
    return {
        ("hotel", "museum1"): 10,
        ("hotel", "museum2"): 12,
        ("hotel", "gallery1"): 15,
        ("hotel", "gallery2"): 18,
        ("museum1", "hotel"): 10,
        ("museum2", "hotel"): 12,
        ("gallery1", "hotel"): 15,
        ("gallery2", "hotel"): 18,
        # Museums are close to each other
        ("museum1", "museum2"): 5,
        ("museum2", "museum1"): 5,
        # Galleries are close to each other
        ("gallery1", "gallery2"): 8,
        ("gallery2", "gallery1"): 8,
        # Museums and galleries are far from each other
        ("museum1", "gallery1"): 25,
        ("gallery1", "museum1"): 25,
        ("museum1", "gallery2"): 28,
        ("gallery2", "museum1"): 28,
        ("museum2", "gallery1"): 27,
        ("gallery1", "museum2"): 27,
        ("museum2", "gallery2"): 30,
        ("gallery2", "museum2"): 30,
    }

@pytest.fixture
def xp_only_weights():
    """Fixture for weights focused only on experience"""
    return {
        'w_xp': 1.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 0.0,
        'w_time': 0.0,
        'w_dur': 0.0
    }

@pytest.fixture
def museum(sample_events):
    """Fixture for the museum event"""
    return sample_events[0]

@pytest.fixture
def gallery(sample_events):
    """Fixture for the gallery event"""
    return sample_events[1]

@pytest.fixture
def basic_itinerary():
    """Fixture for a basic one-day itinerary"""
    return Itinerary(start_day=Day.MONDAY, num_days=1)

@pytest.fixture
def expanded_travel_time_matrix(expanded_sample_events):
    return {
        ("hotel", "museum1"): 2,      # 1 hour
        ("hotel", "museum2"): 2,      # 1 hour  
        ("hotel", "gallery1"): 3,     # 1.5 hours
        ("hotel", "gallery2"): 3,     # 1.5 hours
        ("museum1", "hotel"): 2,
        ("museum2", "hotel"): 2,
        ("gallery1", "hotel"): 3,
        ("gallery2", "hotel"): 3,
        # Museums are close to each other
        ("museum1", "museum2"): 1,    # 30 min
        ("museum2", "museum1"): 1,
        # Galleries are close to each other
        ("gallery1", "gallery2"): 1,  # 30 min
        ("gallery2", "gallery1"): 1,
        # Museums and galleries are moderately far
        ("museum1", "gallery1"): 2,   # 1 hour
        ("gallery1", "museum1"): 2,
        ("museum1", "gallery2"): 2,
        ("gallery2", "museum1"): 2,
        ("museum2", "gallery1"): 2,
        ("gallery1", "museum2"): 2,
        ("museum2", "gallery2"): 2,
        ("gallery2", "museum2"): 2,
    }


def test_greedy_schedule_empty_result(expanded_sample_events, expanded_travel_cost_matrix, expanded_travel_time_matrix):
    # Test with very low budget that should result in empty schedule
    scheduler = GreedyItineraryScheduler(
        events=expanded_sample_events,
        start_day=Day.MONDAY,
        num_days=2,
        total_budget=5,  # Too low to schedule anything
        travel_cost_matrix=expanded_travel_cost_matrix,
        travel_time_matrix=expanded_travel_time_matrix,
        score_fn_weights={
            'w_xp': 1.0,
            'w_count': 0.1,
            'w_cost': 0.5,
            'w_travel': 0.3,
            'w_time': 0.2,
        }
    )
    
    itinerary = scheduler.greedy_schedule(
    )
    
    assert isinstance(itinerary, Itinerary)
    assert len(itinerary.scheduled_events) == 0 

def test_score_event_and_itinerary_partial_duration(scheduler, museum, gallery, basic_itinerary, xp_only_weights):
    """Test that score_event and score_itinerary correctly handle partial duration"""
    # Part 1: Test individual event scoring
    scheduler.score_fn_weights = xp_only_weights
    
    # Test different durations for single event
    full_score = scheduler.score_event(event=museum, actual_duration=museum.duration)
    half_score = scheduler.score_event(event=museum, actual_duration=museum.duration // 2)
    
    # Museum has base_exp=100
    # Full duration (4): base_exp=100
    # Half duration (2): base_exp=50
    # Min duration (1): base_exp=25 
    assert full_score == 100.0, "Full duration should give full base experience"
    assert half_score == 50.0, "Half duration should give half base experience"
    assert  half_score < full_score, "Scores should scale with duration"
    
    # Part 2: Test scoring in itinerary context
    # Schedule museum for full duration and gallery for half duration
    basic_itinerary.schedule_event(
        event=museum,
        day=0,
        start_slot=14,
        duration=museum.duration
    )
    
    basic_itinerary.schedule_event(
        event=gallery,
        day=0,
        start_slot=20,
        duration=gallery.duration // 2
    )
    
    # Test with experience and travel weights
    scheduler.score_fn_weights = {
        'w_xp': 1.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 0.5,
        'w_time': 0.0,
        'w_dur': 0.0
    }
    
    itinerary_score = scheduler.score_itinerary(basic_itinerary)
    
    # Expected calculation:
    # Museum: base = 100 (full duration)
    # Gallery: base=15 (half duration, below bonus threshold)
    # Travel costs:
    # - hotel -> museum: -10 * 0.5 = -5
    # - museum -> gallery: -5 * 0.5 = -2.5
    # - gallery -> hotel: -15 * 0.5 = -7.5
    # Total expected = 120 + 15 - 15 = 120
    assert itinerary_score == 100.0, "Itinerary score should account for partial durations and travel costs"

def test_score_itinerary_travel_costs(scheduler, sample_events, travel_cost_matrix):
    """Test that score_itinerary correctly accounts for travel costs and times between events"""
    itinerary = Itinerary(start_day=Day.MONDAY, num_days=1)
    museum = sample_events[0]
    gallery = sample_events[1]
    
    # Schedule museum then gallery
    itinerary.schedule_event(
        event=museum,
        day=0,
        start_slot=14,
        duration=4
    )
    
    itinerary.schedule_event(
        event=gallery,
        day=0,
        start_slot=20,
        duration=2
    )
    
    # Test with only travel cost weight
    scheduler.score_fn_weights = {
        'w_xp': 0.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 1.0,
        'w_time': 0.0,
        'w_dur': 0.0
    }
    
    cost_score = scheduler.score_itinerary(itinerary)
    
    # Expected travel costs:
    # hotel -> museum: -10
    # museum -> gallery: -5
    # gallery -> hotel: -15
    # Total: -(10 + 5 + 15) = -30
    assert cost_score == -30.0, "Travel cost score should be negative sum of all travel costs"
    
    # Test with only travel time weight
    scheduler.score_fn_weights = {
        'w_xp': 0.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 0.0,
        'w_time': 1.0,
        'w_dur': 0.0
    }
    
    time_score = scheduler.score_itinerary(itinerary)
    
    # Expected travel times (in 30-min slots):
    # hotel -> museum: -2 slots
    # museum -> gallery: -1 slot
    # gallery -> hotel: -3 slots
    # Total: -(2 + 1 + 3) = -6
    assert time_score == -6.0, "Travel time score should be negative sum of all travel times"
    
    # Test with both weights
    scheduler.score_fn_weights = {
        'w_xp': 0.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 1.0,
        'w_time': 0.5,
        'w_dur': 0.0
    }
    
    combined_score = scheduler.score_itinerary(itinerary)
    
    # Expected combined score:
    # Travel costs: -30 * 1.0 = -30
    # Travel times: -6 * 0.5 = -3
    # Total: -30 - 3 = -33
    assert combined_score == -33.0, "Combined score should weight both travel costs and times"

def test_score_itinerary_empty(scheduler):
    """Test that score_itinerary handles empty itineraries correctly"""
    itinerary = Itinerary(start_day=Day.MONDAY, num_days=1)
    
    scheduler.score_fn_weights = {
        'w_xp': 1.0,
        'w_count': 1.0,
        'w_cost': 1.0,
        'w_travel': 1.0,
        'w_time': 0.0,
        'w_dur': 1.0
    }
    
    score = scheduler.score_itinerary(itinerary)
    assert score == 0.0, "Empty itinerary should have score of 0"

def test_score_itinerary_multiple_days(scheduler, sample_events):
    """Test that score_itinerary correctly handles events across multiple days"""
    itinerary = Itinerary(start_day=Day.MONDAY, num_days=2)
    museum = sample_events[0]
    gallery = sample_events[1]
    
    # Schedule museum on day 1
    itinerary.schedule_event(
        event=museum,
        day=0,
        start_slot=14,
        duration=4
    )
    
    # Schedule gallery on day 2
    itinerary.schedule_event(
        event=gallery,
        day=1,
        start_slot=14,
        duration=2
    )
    
    scheduler.score_fn_weights = {
        'w_xp': 1.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 0.5,
        'w_time': 0.0,
        'w_dur': 0.0
    }
    
    score = scheduler.score_itinerary(itinerary)
    
    # Expected calculation:
    # Day 1: 
    # - Museum exp: base=100 (full duration with bonus)
    # - Travel: (hotel->museum: -10) + (museum->hotel: -10) = -20 * 0.5 = -10
    # Day 2:
    # - Gallery exp: base=30 (full duration)
    # - Travel: (hotel->gallery: -15) + (gallery->hotel: -15) = -30 * 0.5 = -15
    # Total = 100 + 30 - 10 - 15 = 105
    assert score == 105.0, "Multi-day score should include all events and daily travel costs"

def test_score_event_edge_cases(scheduler, museum, xp_only_weights):
    """Test score_event with edge cases like zero/negative durations"""
    scheduler.score_fn_weights = xp_only_weights
    
    # Test zero duration
    zero_score = scheduler.score_event(event=museum, actual_duration=0)
    assert zero_score == 0.0, "Zero duration should give zero score"
    
    # Test negative duration (should be treated as zero)
    neg_score = scheduler.score_event(event=museum, actual_duration=-1)
    assert neg_score == 0.0, "Negative duration should give zero score"
    
    # Test duration longer than event's max duration
    over_score = scheduler.score_event(event=museum, actual_duration=museum.duration + 2)
    assert over_score == 100.0, "Over-duration should be capped at max duration score (base)"

def test_score_event_weight_combinations(scheduler, sample_events):
    """Test score_event with different weight combinations"""
    museum = sample_events[0]  # cost=50, duration=4
    
    # Test experience-focused weights
    scheduler.score_fn_weights = {
        'w_xp': 2.0,
        'w_count': 0.0,
        'w_cost': 0.0,
        'w_travel': 0.0,
        'w_time': 0.0,
        'w_dur': 0.0
    }
    exp_score = scheduler.score_event(event=museum, actual_duration=museum.duration)
    assert exp_score == 200.0, "Double XP weight should double experience"
    
    # Test cost-focused weights
    scheduler.score_fn_weights = {
        'w_xp': 0.0,
        'w_count': 0.0,
        'w_cost': 1.0,
        'w_travel': 0.0,
        'w_time': 0.0,
        'w_dur': 0.0
    }
    cost_score = scheduler.score_event(event=museum, actual_duration=museum.duration)
    assert cost_score == -50.0, "Cost weight should give negative score"
    
    # Test balanced weights
    scheduler.score_fn_weights = {
        'w_xp': 1.0,
        'w_count': 1.0,
        'w_cost': 0.5,
        'w_dur': 0.5,
        'w_travel': 0.0,
        'w_time': 0.0
    }
    balanced_score = scheduler.score_event(event=museum, actual_duration=museum.duration)
    # Expected: 
    # XP(100 base) + count(1) - cost(50*0.5) - duration(4*0.5)
    # = 100 + 1 - 25 - 2 = 74
    assert abs(balanced_score - 74.0) < 0.001, "Balanced weights should combine all factors"

def test_greedy_schedule_basic(expanded_sample_events, expanded_travel_cost_matrix, expanded_travel_time_matrix):
    """Test basic functionality of greedy_schedule with sufficient budget and time"""
    scheduler = GreedyItineraryScheduler(
        events=expanded_sample_events,
        start_day=Day.MONDAY,
        num_days=2,
        total_budget=200,  # Enough for multiple events
        travel_cost_matrix=expanded_travel_cost_matrix,
        travel_time_matrix=expanded_travel_time_matrix,
        score_fn_weights={
            'w_xp': 1.0,
            'w_count': 0.1,
            'w_cost': 0.5,
            'w_travel': 0.3,
            'w_time': 0.2,
            'w_dur': 0.0
        }
    )
    
    itinerary = scheduler.greedy_schedule()
    
    # Basic checks
    assert isinstance(itinerary, Itinerary)
    assert len(itinerary.scheduled_events) > 0, "Should schedule at least one event"
    
    # Get events sorted by day and start time
    sorted_events = sorted(
        itinerary.scheduled_events.items(),
        key=lambda x: (x[1][0], x[1][1])  # Sort by (day, start_time)
    )
    
    # Check schedule validity
    last_location = "hotel"
    last_day = -1
    last_end = 0
    total_cost = 0
    
    for event_id, (day, start, end) in sorted_events:
        # Check day bounds
        assert 0 <= day < itinerary.num_days, "Day should be within bounds"
        # Check time slot bounds
        assert 0 <= start < 48, "Start slot should be within bounds"
        assert 0 < end <= 48, "End slot should be within bounds"
        assert start < end, "End slot should be after start slot"
        
        # Check event exists in scheduler's events
        event = next((e for e in expanded_sample_events if e.id == event_id), None)
        assert event is not None, "Scheduled event should exist in event list"
        
        # Check opening hours
        day_of_week = itinerary.get_day_of_week(day)
        opening_hours = event.opening_hours.get(day_of_week)
        assert opening_hours is not None, "Event should be scheduled on an open day"
        open_start, open_end = opening_hours
        assert start >= open_start, "Event should start after opening time"
        assert end <= open_end, "Event should end before closing time"
        
        # If it's a new day, reset last_end and add return to hotel cost for previous day
        if day != last_day and last_day != -1:
            total_cost += expanded_travel_cost_matrix.get((last_location, "hotel"), 0)
            last_location = "hotel"
            last_end = 0
        
        # Check travel time constraints
        if last_location != "hotel" or last_end != 0:
            travel_time = expanded_travel_time_matrix.get((last_location, event_id), 0)
            assert start >= last_end + travel_time, (
                f"Not enough time allowed for travel between {last_location} and {event_id}"
            )
        
        # Add event cost and travel cost
        total_cost += event.cost
        travel_cost = expanded_travel_cost_matrix.get((last_location, event_id), 0)
        total_cost += travel_cost
        
        # Update tracking variables
        last_location = event_id
        last_day = day
        last_end = end
        
        # Check no overlapping events on same day
        for other_id, (other_day, other_start, other_end) in sorted_events:
            if event_id != other_id and day == other_day:
                assert end <= other_start or start >= other_end, "Events should not overlap"
    
    # Add final return to hotel cost
    if last_location != "hotel":
        total_cost += expanded_travel_cost_matrix.get((last_location, "hotel"), 0)
    
    assert total_cost <= scheduler.total_budget, "Total cost should not exceed budget"

def test_greedy_schedule_time_constraints(expanded_sample_events, expanded_travel_cost_matrix, expanded_travel_time_matrix):
    """Test that greedy_schedule respects time constraints and travel times"""
    scheduler = GreedyItineraryScheduler(
        events=expanded_sample_events,
        start_day=Day.MONDAY,
        num_days=1,  # Single day to make it easier to verify
        total_budget=1000,  # High budget to focus on time constraints
        travel_cost_matrix=expanded_travel_cost_matrix,
        travel_time_matrix=expanded_travel_time_matrix,
        score_fn_weights={
            'w_xp': 1.0,
            'w_time': 1.0,  # High weight on travel time
            'w_count': 0.0,
            'w_cost': 0.0,
            'w_travel': 0.0,
            'w_dur': 0.0
        }
    )
    
    itinerary = scheduler.greedy_schedule()
    
    # Check that travel times are respected between events
    events_by_day = defaultdict(list)
    for event_id, (day, start, end) in itinerary.scheduled_events.items():
        events_by_day[day].append((event_id, start, end))
    
    for day, day_events in events_by_day.items():
        # Sort events by start time
        day_events.sort(key=lambda x: x[1])
        
        last_location = "hotel"
        last_end = 0
        
        for event_id, start, end in day_events:
            # Check that enough time was allowed for travel
            travel_time = expanded_travel_time_matrix.get((last_location, event_id), 0)
            assert start >= last_end + travel_time, (
                f"Not enough time allowed for travel between {last_location} and {event_id}"
            )
            
            last_location = event_id
            last_end = end
