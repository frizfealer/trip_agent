import logging
from typing import Dict, List, Tuple

from agent.scheduler.defaults import DEFAULT_DAY_RESOLUTION
from agent.scheduler.event import Day, Event, convert_slot_to_time, score_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ITINERARY_START_EVENT_NAME = "DEFAULT_HOTEL"
MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS = 1
MINIMUM_TRAVEL_COST_BETWEEN_EVENTS = 0.0


def postprocess_travel_time_matrix(
    events: List[Event],
    travel_time_mat: Dict[Tuple[str, str], int],
    default_travel_time: int = MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS,
) -> Dict[Tuple[str, str], int]:
    """Postprocess the travel time matrix to ensure that the travel time between events is at least MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS."""
    # add hotel connections
    for event in events:
        travel_time_mat[(ITINERARY_START_EVENT_NAME, event.name)] = default_travel_time
        travel_time_mat[(event.name, ITINERARY_START_EVENT_NAME)] = default_travel_time
    return travel_time_mat


def generate_travel_cost_matrix_from_travel_time_matrix(
    travel_time_mat: Dict[Tuple[str, str], int], travel_type: str
) -> Dict[Tuple[str, str], float]:
    """Postprocess the travel cost matrix to ensure that the travel cost between events is at least MINIMUM_TRAVEL_COST_BETWEEN_EVENTS."""
    travel_cost_mat = {}
    if travel_type == "driving":
        # Assume it is 0.72 USD per mile, driving at 45 mph
        for origin, destination in travel_time_mat.keys():
            travel_cost_mat[(origin, destination)] = 0.72 * travel_time_mat[(origin, destination)] / 60 * 45
            travel_cost_mat
    elif travel_type == "walking" or travel_type == "bicycling":
        for origin, destination in travel_time_mat.keys():
            travel_cost_mat[(origin, destination)] = 0.0
    elif travel_type == "transit":
        # Assume it is 2.75 USD per ride
        for origin, destination in travel_time_mat.keys():
            if origin == destination:
                travel_cost_mat[(origin, destination)] = 0.0
            else:
                travel_cost_mat[(origin, destination)] = 2.75
    return travel_cost_mat


class Itinerary:
    def __init__(
        self,
        start_day: Day,
        num_days: int,
        budget: float,
        travel_cost_mat: Dict[Tuple[str, str], float],
        travel_time_mat: Dict[Tuple[str, str], int],
        day_resolution: int = DEFAULT_DAY_RESOLUTION,
        min_travel_time: int = MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS,
        min_travel_cost: float = MINIMUM_TRAVEL_COST_BETWEEN_EVENTS,
    ):
        """
        Initialize an itinerary.
        Args:
            travel_cost_mat: Maps event_id to event_id to travel cost. cost is 0 if the events are the same.
            travel_time_mat: Maps event_id to event_id to travel time. time is 0 if the events are the same.
        """
        self.start_day = start_day
        self.num_days = num_days
        # Initialize empty slots for each day (N slots per day)
        self.days = [[None for _ in range(day_resolution)] for _ in range(num_days)]
        # Track which events are scheduled and their time slots
        # Dict[str, Tuple[Event, int, int, int] maps event_id to list of (event, day, start_slot, end_slot (not inclusive))
        self.scheduled_events: Dict[str, Tuple[Event, int, int, int]] = {}
        logger.info(f"Created new itinerary starting on {start_day} for {num_days} days, with budget {budget}")
        self.day_resolution = day_resolution
        self.budget = budget
        self.travel_cost_mat = travel_cost_mat
        self.travel_time_mat = travel_time_mat
        self.min_travel_time = min_travel_time
        self.min_travel_cost = min_travel_cost
        self.itinerary_start_event = Event(
            name=ITINERARY_START_EVENT_NAME,
            id=ITINERARY_START_EVENT_NAME,
            cost=0.0,
            duration=0,
            base_exp=0.0,
            opening_hours=[(0, day_resolution) for _ in Day],
        )
        self.total_cost = 0.0
        # TODO: maintain total travel time and gap time when we schedule events
        self.total_travel_time = 0
        self.total_gap_time = 0

    def reset(self):
        """Reset the itinerary to the initial state."""
        self.days = [[None for _ in range(self.day_resolution)] for _ in range(self.num_days)]
        self.scheduled_events = {}
        self.total_cost = 0.0
        self.total_travel_time = 0
        self.total_gap_time = 0

    def get_day_of_week(self, day_index: int) -> Day:
        """Convert a day index in the itinerary to the actual day of the week."""
        return Day((self.start_day + day_index) % 7)

    def get_previous_event(self, day: int, start_slot: int) -> Tuple[Event, int]:
        """Get the previous event on the same day, return the event and the last slot index of the previous event."""
        for slot in range(start_slot - 1, -1, -1):
            if self.days[day][slot] is not None:
                return self.days[day][slot], slot
        return self.itinerary_start_event, -1

    def get_next_event(self, day: int, end_slot: int) -> Tuple[Event, int]:
        """Get the next event on the same day, return the event and the first slot index of the next event."""
        for slot in range(end_slot, self.day_resolution):
            if self.days[day][slot] is not None:
                return self.days[day][slot], slot
        return self.itinerary_start_event, -1

    def calculate_travel_cost_change_for_event(
        self, event: Event, prev_event: Event, next_event: Event, add_event: bool
    ) -> float:
        """Get the additional cost for scheduling an event on a specific day and start time."""
        prev_to_next_event_travel_cost = self.travel_cost_mat.get(
            (prev_event.name, next_event.name), self.min_travel_cost
        )
        prev_to_current_event_travel_cost = self.travel_cost_mat.get(
            (prev_event.name, event.name), self.min_travel_cost
        )
        current_to_next_event_travel_cost = self.travel_cost_mat.get(
            (event.name, next_event.name), self.min_travel_cost
        )
        if add_event:
            travel_cost_change = (
                prev_to_current_event_travel_cost + current_to_next_event_travel_cost - prev_to_next_event_travel_cost
            )
        else:
            travel_cost_change = (
                prev_to_next_event_travel_cost - prev_to_current_event_travel_cost - current_to_next_event_travel_cost
            )
        return travel_cost_change

    def calculate_additional_travel_time_for_event(self, event: Event, prev_event: Event, next_event: Event) -> int:
        """Get the additional travel time for scheduling an event on a specific day and start time."""
        prev_to_next_event_travel_time = self.travel_time_mat.get(
            (prev_event.name, next_event.name), self.min_travel_time
        )
        prev_to_current_event_travel_time = self.travel_time_mat.get(
            (prev_event.name, event.name), self.min_travel_time
        )
        current_to_next_event_travel_time = self.travel_time_mat.get(
            (event.name, next_event.name), self.min_travel_time
        )
        additional_travel_time = (
            prev_to_current_event_travel_time + current_to_next_event_travel_time - prev_to_next_event_travel_time
        )
        return additional_travel_time

    def check_schedule_event(self, event: Event, day: int, start_slot: int, duration: int) -> int:
        """
        Attempt to schedule an event on a specific day and start time.
        Args:
            event: The event to schedule
            day: The day index to schedule on
            start_slot: The starting time slot
            duration: duration to schedule (can be partial duration)
        Returns:
            0 if successful; other non-zero code if failed. The code is as follows:
            1: event not open in the day.
            2: event scheduled too early.
            3: event end time is beyond closing time or event end time.
            4: time slot is already occupied.
            5: exceeds itinerary budget.
            6: cannot make it in time for the next event.
        """
        if not (0 <= day < self.num_days) or not (0 <= start_slot < self.day_resolution):
            raise ValueError(
                f"Attempted to schedule event on day {day} with start slot {start_slot}, which is out of bounds"
            )
        if event.duration < duration:
            raise ValueError(
                f"Attempted to schedule event {event.name} (ID: {event.id}) with duration {duration}, "
                f"which is less than the event duration {event.duration}"
            )
        # Get the actual day of the week
        day_of_week = self.get_day_of_week(day)
        # Check if the event is open on this day
        if event.opening_hours.get(day_of_week) is None:
            logger.debug(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                f"on day {day}, which is not open on {day_of_week}"
            )
            return 1
        day_open_start, day_open_end = event.opening_hours[day_of_week]
        # Check if the event timing falls within opening hours
        if start_slot < day_open_start:
            logger.debug(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                f"on day {day} with start slot {start_slot}, which is before the opening time {day_open_start}"
            )
            return 2
        end_slot = start_slot + duration
        if end_slot > day_open_end or end_slot > self.day_resolution:  # Can't go beyond closing time or midnight
            logger.debug(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                f"on day {day} between [{start_slot}, {end_slot}),"
                f"which is after the closing time {day_open_end} or beyond midnight"
            )
            return 3
        # Check if all slots are empty
        for slot in range(start_slot, end_slot):
            if self.days[day][slot] is not None:
                logger.debug(
                    f"Attempted to schedule event {event.name} (ID: {event.id}) "
                    f"on day {day} between [{start_slot}, {end_slot}),"
                    f"which is already occupied by {self.days[day][slot].name} (ID: {self.days[day][slot].id})"
                )
                return 4

        prev_event, prev_slot = self.get_previous_event(day, start_slot)
        next_event, next_slot = self.get_next_event(day, end_slot)
        additional_travel_cost = self.calculate_travel_cost_change_for_event(
            event, prev_event, next_event, add_event=True
        )
        if self.total_cost + additional_travel_cost + event.cost > self.budget:
            logger.debug(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                + f"after {prev_event.name} (ID: {prev_event.id}), "
                + f"before {next_event.name} (ID: {next_event.id}), "
                + f"but the current total cost ({self.total_cost}) + "
                + f"additional travel cost ({additional_travel_cost}) + event cost ({event.cost}) "
                + f"exceeds the budget ({self.budget})"
            )
            return 5

        prev_to_current_event_travel_time = self.travel_time_mat.get(
            (prev_event.name, event.name), self.min_travel_time
        )
        current_to_next_event_travel_time = self.travel_time_mat.get(
            (event.name, next_event.name), self.min_travel_time
        )
        if (prev_slot != -1 and prev_slot + prev_to_current_event_travel_time >= start_slot) or (
            next_slot != -1 and (end_slot - 1) + current_to_next_event_travel_time >= next_slot
        ):
            logger.debug(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                + f"after {prev_event.name} (ID: {prev_event.id}),"
                + f"but the last slot of the previous event ({prev_slot}) + travel time ({prev_to_current_event_travel_time})"
                + f" is greater than or equal to current event start slot ({start_slot})"
            )
            return 6
        return 0

    def schedule_event(self, event: Event, day: int, start_slot: int, duration: int) -> int:
        end_slot = start_slot + duration
        status_code = self.check_schedule_event(event, day, start_slot, duration)
        if status_code == 0:
            for slot in range(start_slot, end_slot):
                self.days[day][slot] = event
            prev_event, _ = self.get_previous_event(day, start_slot)
            next_event, _ = self.get_next_event(day, end_slot)
            travel_cost_change = self.calculate_travel_cost_change_for_event(
                event, prev_event, next_event, add_event=True
            )
            # Update the itinerary cost and experience
            self.total_cost += travel_cost_change + event.cost
            # Update the scheduled_events dictionary
            self.scheduled_events[event.name] = (event, day, start_slot, end_slot)
            logger.debug(
                f"Successfully scheduled event {event.name} (ID: {event.id}) on day {day} in [{start_slot}, {end_slot})"
            )
        return status_code

    def unschedule_event(self, event: Event):
        """unschedule an event from the itinerary."""
        if event.name not in self.scheduled_events:
            raise ValueError(f"Attempted to remove event {event.name} (ID: {event.id}), which is not scheduled")
        _, day, start_slot, end_slot = self.scheduled_events[event.name]
        for slot in range(start_slot, end_slot):
            self.days[day][slot] = None

        prev_event, _ = self.get_previous_event(day, start_slot)
        next_event, _ = self.get_next_event(day, end_slot)
        travel_cost_change = self.calculate_travel_cost_change_for_event(
            event, prev_event, next_event, add_event=False
        )
        self.total_cost += -event.cost + travel_cost_change
        del self.scheduled_events[event.name]

    def calculate_day_cost(self, day_index: int) -> Tuple[float, float]:
        """Calculate the total cost for a single day in the itinerary.

        Args:
            day_index: The index of the day to calculate the cost for

        Returns:
            float: The total cost for the specified day including event costs and travel costs
        """
        if not (0 <= day_index < self.num_days):
            raise ValueError(f"Day index {day_index} is out of bounds")

        day_event_cost = 0.0
        day_travel_cost = 0.0
        day = self.days[day_index]

        # Calculate costs starting from hotel
        prev_event = self.itinerary_start_event
        # Add cost for each event and travel between events

        for event in day:
            if event is not None and prev_event != event:
                # Add travel cost to this event
                day_travel_cost += self.travel_cost_mat.get((prev_event.name, event.name), self.min_travel_cost)
                # Add event cost
                day_event_cost += event.cost
                prev_event = event
        # Add cost to return to hotel
        if prev_event != self.itinerary_start_event:
            day_travel_cost += self.travel_cost_mat.get(
                (prev_event.name, self.itinerary_start_event.name), self.min_travel_cost
            )

        logger.debug(f"Day {day_index} event cost: {day_event_cost}, travel cost: {day_travel_cost}")
        return (day_event_cost, day_travel_cost)

    def calculate_total_cost(self) -> Tuple[float, float]:
        """Calculate the total cost of all scheduled events across all days."""
        total_event_cost = 0.0
        total_travel_cost = 0.0
        for day_index in range(self.num_days):
            event_cost, travel_cost = self.calculate_day_cost(day_index)
            total_event_cost += event_cost
            total_travel_cost += travel_cost
        return total_event_cost, total_travel_cost

    def calculate_day_travel_and_gap_time(self, day_index: int) -> Tuple[int, int]:
        """Calculate the total travel time and gap time of all scheduled events for a given day."""
        if not (0 <= day_index < self.num_days):
            raise ValueError(f"Day index {day_index} is out of bounds")

        day_travel_time = 0
        day_gap_time = 0
        day = self.days[day_index]

        # Calculate costs starting from hotel
        prev_event = self.itinerary_start_event
        # Add cost for each event and travel between events
        for event in day:
            if event is not None and prev_event != event:
                travel_time = self.travel_time_mat.get((prev_event.name, event.name), self.min_travel_time)
                day_travel_time += travel_time
                # we don't count the gap time for the first event (hotel)
                if prev_event != self.itinerary_start_event:
                    # current event start slot - previous event end slot - travel time
                    day_gap_time += (
                        self.scheduled_events[event.name][2] - self.scheduled_events[prev_event.name][3] - travel_time
                    )
                prev_event = event
        # Add cost to return to hotel
        if prev_event != self.itinerary_start_event:
            day_travel_time += self.travel_time_mat.get(
                (prev_event.name, self.itinerary_start_event.name), self.min_travel_time
            )

        logger.debug(f"Day {day_index} travel time: {day_travel_time}, gap time: {day_gap_time}")
        return day_travel_time, day_gap_time

    def calculate_total_travel_time_and_gap_time(self) -> Tuple[int, int]:
        """Calculate the total travel time and gap time of all scheduled events across all days."""
        total_travel_time = 0
        total_gap_time = 0
        for day_index in range(self.num_days):
            day_travel_time, day_gap_time = self.calculate_day_travel_and_gap_time(day_index)
            total_travel_time += day_travel_time
            total_gap_time += day_gap_time
        return total_travel_time, total_gap_time

    def __str__(self) -> str:
        """Return a human-readable schedule."""
        schedule_str = ""
        for day in range(self.num_days):
            schedule_str += f"\nDay {day + 1}:\n"
            prev_printed_event = None
            prev_event = None
            start_slot = None
            for slot in range(self.day_resolution):
                event = self.days[day][slot]
                # a new event starts
                if prev_event is None and event is not None:
                    if prev_printed_event is not None:
                        travel_time = self.travel_time_mat.get(
                            (prev_printed_event.name, event.name), self.min_travel_time
                        )
                        travel_time_hour, travel_time_minute = convert_slot_to_time(travel_time, self.day_resolution)
                        schedule_str += f"\ttravel time: {travel_time_hour*60 + travel_time_minute} minutes\n"
                    start_slot = slot
                # an event ends
                if prev_event is not None and event is None:
                    start_hour, start_minute = convert_slot_to_time(start_slot, self.day_resolution)
                    end_hour, end_minute = convert_slot_to_time(slot, self.day_resolution)
                    time_start = f"{start_hour:02d}:{start_minute:02d}"
                    time_end = f"{end_hour:02d}:{end_minute:02d}"
                    schedule_str += f"  {time_start}-{time_end}: {prev_event.name}\n"
                    prev_printed_event = prev_event
                prev_event = event
        return schedule_str


def score_day_itinerary(
    itinerary: Itinerary,
    day_index: int,
    w_xp: float,
    w_count: float,
    w_cost: float,
    w_dur: float,
    w_travel_time: float,
    w_gap: float,
) -> float:
    """Score the itinerary for a given day.

    This function calculates a score for a specific day in an itinerary based on multiple factors:
    - The experience value of scheduled events
    - The number of events scheduled
    - The cost of events
    - The duration of events
    - The travel time between events
    - The gap time between events
    - The travel cost between events

    The formula used is:

    score = sum(score_event(event, duration, w_xp, w_count, w_cost, w_dur) for each event)
            + w_travel_time * day_travel_time
            + w_gap * day_gap_time
            + w_cost * day_travel_cost

    Where score_event() calculates the score for an individual event.

    Args:
        itinerary (Itinerary): The itinerary object containing scheduled events
        day_index (int): The index of the day to score
        w_xp (float): Weight for experience value of events
        w_count (float): Weight for number of events
        w_cost (float): Weight for cost (both event and travel costs)
        w_dur (float): Weight for event duration
        w_travel_time (float): Weight for travel time between events
        w_gap (float): Weight for gap time between events

    Returns:
        float: The calculated score for the specified day
    """
    score = 0.0
    for event, day, start_slot, end_slot in itinerary.scheduled_events.values():
        if day == day_index:
            score += score_event(event, end_slot - start_slot, w_xp, w_count, w_cost, w_dur)
    day_travel_time, day_gap_time = itinerary.calculate_day_travel_and_gap_time(day_index)
    score += w_travel_time * day_travel_time + w_gap * day_gap_time
    _, day_travel_cost = itinerary.calculate_day_cost(day_index)
    score += w_cost * day_travel_cost
    return score


def score_itinerary(
    itinerary: Itinerary, w_xp: float, w_count: float, w_cost: float, w_dur: float, w_gap: float, w_travel_time: float
) -> float:
    """Score the entire itinerary across all days.

    This function calculates a comprehensive score for the entire itinerary by summing
    the scores of each individual day. The score considers multiple factors including
    event experience values, number of events, costs, durations, travel times, and gaps.

    The formula used is:

    score = sum(score_day_itinerary(itinerary, day_index, w_xp, w_count, w_cost, w_dur, w_travel_time, w_gap)
                for each day_index in range(itinerary.num_days))

    Where score_day_itinerary() calculates:
    sum(score_event(event, duration, w_xp, w_count, w_cost, w_dur) for each event on that day)
    + w_travel_time * day_travel_time
    + w_gap * day_gap_time
    + w_cost * day_travel_cost

    And score_event() calculates:
    score_event = (base_exp * (actual_duration/event.duration)) * w_xp + 1.0 * w_count +
                  event.cost * w_cost + actual_duration * w_dur

    Args:
        itinerary (Itinerary): The itinerary object containing scheduled events across multiple days
        w_xp (float): Weight for experience value of events
        w_count (float): Weight for number of events
        w_cost (float): Weight for cost (both event and travel costs)
        w_dur (float): Weight for event duration
        w_gap (float): Weight for gap time between events
        w_travel_time (float): Weight for travel time between events

    Returns:
        float: The calculated score for the entire itinerary
    """
    score = 0.0
    for day_index in range(itinerary.num_days):
        score += score_day_itinerary(itinerary, day_index, w_xp, w_count, w_cost, w_dur, w_travel_time, w_gap)
    return score
