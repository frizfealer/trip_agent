import pytest

from agent.scheduler.itinerary import (
    ITINERARY_START_EVENT_NAME,
    Day,
    Event,
)
from agent.scheduler.itinerary_scheduler import ItineraryScheduler


@pytest.fixture
def events():
    """Create sample events for testing."""
    monday_to_friday = {
        Day.MONDAY: (14, 36),  # 7:00-18:00
        Day.TUESDAY: (14, 36),  # 7:00-18:00
        Day.WEDNESDAY: (14, 36),  # 7:00-18:00
        Day.THURSDAY: (14, 36),  # 7:00-18:00
        Day.FRIDAY: (14, 36),  # 7:00-18:00
        Day.SATURDAY: None,  # Closed
        Day.SUNDAY: None,  # Closed
    }

    return [
        Event(
            name="Museum Visit",
            cost=50,
            duration=4,
            opening_hours=monday_to_friday,
            base_exp=100,
        ),
        Event(
            name="Art Gallery",
            cost=20,
            duration=2,
            opening_hours=monday_to_friday,
            base_exp=30,
        ),
        Event(
            name="Historical Site",
            cost=40,
            duration=3,
            opening_hours=monday_to_friday,
            base_exp=60,
        ),
        Event(
            name="City Tour",
            cost=70,
            duration=4,
            opening_hours=monday_to_friday,
            base_exp=130,
        ),
    ]


@pytest.fixture
def travel_cost_matrix(events):
    """Create a travel cost matrix for testing."""
    travel_cost_matrix = {
        (ITINERARY_START_EVENT_NAME, events[0].name): 10,
        (ITINERARY_START_EVENT_NAME, events[1].name): 15,
        (ITINERARY_START_EVENT_NAME, events[2].name): 20,
        (ITINERARY_START_EVENT_NAME, events[3].name): 25,
        (events[0].name, ITINERARY_START_EVENT_NAME): 10,
        (events[1].name, ITINERARY_START_EVENT_NAME): 15,
        (events[2].name, ITINERARY_START_EVENT_NAME): 20,
        (events[3].name, ITINERARY_START_EVENT_NAME): 25,
        # Add costs between events
        (events[0].name, events[1].name): 5,
        (events[1].name, events[0].name): 5,
        (events[0].name, events[2].name): 10,
        (events[2].name, events[0].name): 10,
        (events[1].name, events[2].name): 5,
        (events[2].name, events[1].name): 5,
        (events[1].name, events[3].name): 10,
        (events[3].name, events[1].name): 10,
        (events[2].name, events[3].name): 15,
        (events[3].name, events[2].name): 15,
    }

    # Add zero cost for same event
    for event in events:
        travel_cost_matrix[(event.name, event.name)] = 0
    travel_cost_matrix[(ITINERARY_START_EVENT_NAME, ITINERARY_START_EVENT_NAME)] = 0

    return travel_cost_matrix


@pytest.fixture
def travel_time_matrix(travel_cost_matrix):
    """Create a travel time matrix for testing."""
    travel_time_matrix = {}
    for (event1, event2), cost in travel_cost_matrix.items():
        travel_time_matrix[(event1, event2)] = cost // 5
    return travel_time_matrix


@pytest.fixture
def scheduler(events, travel_cost_matrix, travel_time_matrix):
    """Create a scheduler instance for testing."""
    return ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=100,
        travel_cost_matrix=travel_cost_matrix,
        travel_time_matrix=travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )


def test_gap_weight_affects_scheduling(scheduler, events):
    """Test that the gap weight affects the scheduling of events."""
    # Create two schedulers with different gap weights
    low_gap_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    high_gap_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    # Create schedules with different gap weights
    low_gap_weights = {
        "w_xp": 1.0,
        "w_count": 1.0,
        "w_cost": -1.0,
        "w_dur": -0.2,
        "w_travel_time": -1.0,
        "w_gap": -5.0,
    }
    high_gap_weights = {
        "w_xp": 1.0,
        "w_count": 1.0,
        "w_cost": -1.0,
        "w_dur": -0.2,
        "w_travel_time": -1.0,
        "w_gap": 5.0,
    }

    low_gap_itinerary = low_gap_weight_scheduler.greedy_schedule(score_fn_weights=low_gap_weights)
    high_gap_itinerary = high_gap_weight_scheduler.greedy_schedule(score_fn_weights=high_gap_weights)

    # Calculate total gap time for both itineraries
    _, low_gap_total_gap_time = low_gap_itinerary.calculate_total_travel_time_and_gap_time()
    _, high_gap_total_gap_time = high_gap_itinerary.calculate_total_travel_time_and_gap_time()
    # With negative gap weight, the scheduler should minimize gaps
    # With positive gap weight, the scheduler should allow more gaps
    assert low_gap_total_gap_time <= high_gap_total_gap_time


def test_experience_weight_affects_scheduling(scheduler, events):
    """Test that the experience weight affects the scheduling of events."""
    # Create two schedulers with different experience weights
    low_xp_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=1,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    high_xp_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=1,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    # Create schedules with different experience weights
    low_xp_weights = {"w_xp": 0.1, "w_count": 1.0, "w_cost": -1.0, "w_dur": -0.2, "w_travel_time": 0, "w_gap": -1.0}
    high_xp_weights = {
        "w_xp": 10.0,
        "w_count": 1.0,
        "w_cost": -1.0,
        "w_dur": -0.2,
        "w_travel_time": 0,
        "w_gap": -1.0,
    }

    low_xp_itinerary = low_xp_weight_scheduler.greedy_schedule(score_fn_weights=low_xp_weights)
    high_xp_itinerary = high_xp_weight_scheduler.greedy_schedule(score_fn_weights=high_xp_weights)

    # Calculate total experience for both itineraries
    low_xp_score = low_xp_weight_scheduler.normalized_score_itinerary(low_xp_itinerary, **low_xp_weights)
    high_xp_score = high_xp_weight_scheduler.normalized_score_itinerary(high_xp_itinerary, **high_xp_weights)

    # With lower experience weight, the scheduler should not schedule any events
    assert len(low_xp_itinerary.scheduled_events) == 0
    assert low_xp_score == 0
    # With higher experience weight, the scheduler should schedule the highest experience event
    assert events[3].name in high_xp_itinerary.scheduled_events
    assert high_xp_score > 0


def test_cost_weight_affects_scheduling(scheduler, events):
    """Test that the cost weight affects the scheduling of events."""
    # Create two schedulers with different cost weights
    low_cost_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    high_cost_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    # Create schedules with different cost weights
    low_cost_weights = {
        "w_xp": 2.0,
        "w_count": 1.0,
        "w_cost": -0.1,
        "w_dur": -0.2,
        "w_travel_time": -1.0,
        "w_gap": -1.0,
    }
    high_cost_weights = {
        "w_xp": 2.0,
        "w_count": 1.0,
        "w_cost": -1,
        "w_dur": -0.2,
        "w_travel_time": -1.0,
        "w_gap": -1.0,
    }

    low_cost_itinerary = low_cost_weight_scheduler.greedy_schedule(score_fn_weights=low_cost_weights)
    high_cost_itinerary = high_cost_weight_scheduler.greedy_schedule(score_fn_weights=high_cost_weights)

    # Calculate total cost for both itineraries
    low_cost_total_event_cost, low_cost_total_travel_cost = low_cost_itinerary.calculate_total_cost()
    high_cost_total_event_cost, high_cost_total_travel_cost = high_cost_itinerary.calculate_total_cost()
    # With higher negative cost weight, the scheduler should prioritize lower-cost events
    assert (high_cost_total_event_cost + high_cost_total_travel_cost) <= (
        low_cost_total_event_cost + low_cost_total_travel_cost
    )


def test_travel_time_weight_affects_scheduling(scheduler, events):
    """Test that the travel time weight affects the scheduling of events."""
    # Create two schedulers with different travel time weights
    low_travel_time_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    high_travel_time_weight_scheduler = ItineraryScheduler(
        events=events.copy(),
        start_day=Day.MONDAY,
        num_days=3,
        total_budget=300,
        travel_cost_matrix=scheduler.travel_cost_matrix,
        travel_time_matrix=scheduler.travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    # Create schedules with different travel time weights
    low_travel_time_weights = {
        "w_xp": 2.0,
        "w_count": 1.0,
        "w_cost": -1.0,
        "w_dur": -0.2,
        "w_travel_time": -0.1,
        "w_gap": -1.0,
    }
    high_travel_time_weights = {
        "w_xp": 2.0,
        "w_count": 1.0,
        "w_cost": -1.0,
        "w_dur": -0.2,
        "w_travel_time": -10.0,
        "w_gap": -1.0,
    }

    low_travel_time_itinerary = low_travel_time_weight_scheduler.greedy_schedule(
        score_fn_weights=low_travel_time_weights
    )
    high_travel_time_itinerary = high_travel_time_weight_scheduler.greedy_schedule(
        score_fn_weights=high_travel_time_weights
    )

    # Calculate total travel time for both itineraries
    low_travel_time_total_travel_time, _ = low_travel_time_itinerary.calculate_total_travel_time_and_gap_time()
    high_travel_time_total_travel_time, _ = high_travel_time_itinerary.calculate_total_travel_time_and_gap_time()
    # With higher negative travel time weight, the scheduler should minimize travel time
    assert high_travel_time_total_travel_time <= low_travel_time_total_travel_time
