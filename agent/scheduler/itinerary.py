import logging
from typing import Dict, Tuple

from agent.scheduler.defaults import DEFAULT_DAY_RESOLUTION
from agent.scheduler.event import Day, Event, score_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ITINERARY_START_EVENT_NAME = "DEFAULT_HOTEL"
MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS = 1
MINIMUM_TRAVEL_COST_BETWEEN_EVENTS = 0.0


class Itinerary:
    def __init__(self, start_day: Day, num_days: int, budget: float,
                 travel_cost_mat: Dict[Tuple[str, str], float],
                 travel_time_mat: Dict[Tuple[str, str], int],
                 day_resolution: int = DEFAULT_DAY_RESOLUTION,
                 min_travel_time: int = MINIMUM_TRAVEL_TIME_BETWEEN_EVENTS,
                 min_travel_cost: float = MINIMUM_TRAVEL_COST_BETWEEN_EVENTS):
        """
        Initialize an itinerary.
        Args:
            travel_cost_mat: Maps event_id to event_id to travel cost. cost is 0 if the events are the same.
            travel_time_mat: Maps event_id to event_id to travel time. time is 0 if the events are the same.
        """
        self.start_day = start_day
        self.num_days = num_days
        # Initialize empty slots for each day (N slots per day)
        self.days = [[None for _ in range(day_resolution)]
                     for _ in range(num_days)]
        # Track which events are scheduled and their time slots
        # Dict[str, Tuple[Event, int, int, int] maps event_id to list of (event, day, start_slot, end_slot (not inclusive))
        self.scheduled_events: Dict[str, Tuple[Event, int, int, int]] = {}
        logger.info(
            f"Created new itinerary starting on {start_day} for {num_days} days, with budget {budget}"
        )
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
        self.days = [[None for _ in range(self.day_resolution)]
                     for _ in range(self.num_days)]
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

    def calculate_travel_cost_change_for_event(self, event: Event, prev_event: Event, next_event: Event, add_event: bool) -> float:
        """Get the additional cost for scheduling an event on a specific day and start time."""
        prev_to_next_event_travel_cost = self.travel_cost_mat.get(
            (prev_event.id, next_event.id), self.min_travel_cost)
        prev_to_current_event_travel_cost = self.travel_cost_mat.get(
            (prev_event.id, event.id), self.min_travel_cost)
        current_to_next_event_travel_cost = self.travel_cost_mat.get(
            (event.id, next_event.id), self.min_travel_cost)
        if add_event:
            travel_cost_change = prev_to_current_event_travel_cost + \
                current_to_next_event_travel_cost - prev_to_next_event_travel_cost
        else:
            travel_cost_change = prev_to_next_event_travel_cost - \
                prev_to_current_event_travel_cost - current_to_next_event_travel_cost
        return travel_cost_change

    def calculate_additional_travel_time_for_event(self, event: Event, prev_event: Event, next_event: Event) -> int:
        """Get the additional travel time for scheduling an event on a specific day and start time."""
        prev_to_next_event_travel_time = self.travel_time_mat.get(
            (prev_event.id, next_event.id), self.min_travel_time)
        prev_to_current_event_travel_time = self.travel_time_mat.get(
            (prev_event.id, event.id), self.min_travel_time)
        current_to_next_event_travel_time = self.travel_time_mat.get(
            (event.id, next_event.id), self.min_travel_time)
        additional_travel_time = prev_to_current_event_travel_time + \
            current_to_next_event_travel_time - prev_to_next_event_travel_time
        return additional_travel_time

    def check_schedule_event(
        self, event: Event, day: int, start_slot: int, duration: int
    ) -> int:
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
        if not (0 <= day < self.num_days) and not (0 <= start_slot < self.day_resolution):
            raise ValueError(
                f"Attempted to schedule event on day {day} with start slot {start_slot}, which is out of bounds"
            )
        if event.id in self.scheduled_events:
            raise ValueError(
                f"Attempted to schedule event {event.name} (ID: {event.id}), which is already scheduled"
            )
        if event.duration < duration:
            raise ValueError(
                f"Attempted to schedule event {event.name} (ID: {event.id}) with duration {duration}, which is less than the event duration {event.duration}"
            )
        # Get the actual day of the week
        day_of_week = self.get_day_of_week(day)
        # Check if the event is open on this day
        if event.opening_hours.get(day_of_week) is None:
            logger.info(
                f"Attempted to schedule event {event.name} (ID: {event.id}) on day {day}, which is not open on {day_of_week}"
            )
            return 1
        day_open_start, day_open_end = event.opening_hours[day_of_week]
        # Check if the event timing falls within opening hours
        if start_slot < day_open_start:
            logger.info(
                f"Attempted to schedule event {event.name} (ID: {event.id}) on day {day} with start slot {start_slot}, which is before the opening time {day_open_start}"
            )
            return 2
        end_slot = start_slot + duration
        if (
            end_slot > day_open_end or end_slot > self.day_resolution
        ):  # Can't go beyond closing time or midnight
            logger.info(
                f"Attempted to schedule event {event.name} (ID: {event.id}) on day {day} between [{start_slot}, {end_slot}),"
                + "which is after "
            )
            return 3
        # Check if all slots are empty
        for slot in range(start_slot, end_slot):
            if self.days[day][slot] is not None:
                logger.info(
                    f"Attempted to schedule event {event.name} (ID: {event.id}) on day {day} between [{start_slot}, {end_slot}),"
                    + f"which is already occupied by {self.days[day][slot].name} (ID: {self.days[day][slot].id})"
                )
                return 4

        prev_event, prev_slot = self.get_previous_event(day, start_slot)
        next_event, next_slot = self.get_next_event(day, end_slot)
        additional_travel_cost = self.calculate_travel_cost_change_for_event(
            event, prev_event, next_event, add_event=True)
        if self.total_cost + additional_travel_cost + event.cost > self.budget:
            logger.info(
                f"Attempted to schedule event {event.name} (ID: {event.id}) "
                + f"after {prev_event.name} (ID: {prev_event.id}), "
                + f"before {next_event.name} (ID: {next_event.id}), "
                + f"but the current total cost ({self.total_cost}) + additional travel cost ({additional_travel_cost}) + event cost ({event.cost}) "
                + f"exceeds the budget ({self.budget})"
            )
            return 5

        prev_to_current_event_travel_time = self.travel_time_mat.get(
            (prev_event.id, event.id), self.min_travel_time)
        current_to_next_event_travel_time = self.travel_time_mat.get(
            (event.id, next_event.id), self.min_travel_time)
        if (prev_slot != -1 and prev_slot + prev_to_current_event_travel_time >= start_slot) or \
                (next_slot != -1 and (end_slot - 1) + current_to_next_event_travel_time >= next_slot):
            logger.info(
                f"Attempted to schedule event {event.name} (ID: {event.id}) after {prev_event.name} (ID: {prev_event.id}),"
                + f"but the last slot of the previous event ({prev_slot}) + travel time ({prev_to_current_event_travel_time})"
                + f" is greater than or equal to current event start slot ({start_slot})"
            )
            return 6
        return 0

    def schedule_event(
        self, event: Event, day: int, start_slot: int, duration: int
    ) -> int:
        end_slot = start_slot + duration
        status_code = self.check_schedule_event(
            event, day, start_slot, duration)
        if status_code == 0:
            for slot in range(start_slot, end_slot):
                self.days[day][slot] = event
            prev_event, _ = self.get_previous_event(day, start_slot)
            next_event, _ = self.get_next_event(day, end_slot)
            travel_cost_change = self.calculate_travel_cost_change_for_event(
                event, prev_event, next_event, add_event=True)
            # Update the itinerary cost and experience
            self.total_cost += (travel_cost_change + event.cost)
            # Update the scheduled_events dictionary
            self.scheduled_events[event.id] = (
                event, day, start_slot, end_slot)
            logger.info(
                f"Successfully scheduled event {event.name} (ID: {event.id}) on day {day} in [{start_slot}, {end_slot})"
            )
        return status_code

    def unschedule_event(self, event: Event):
        """unschedule an event from the itinerary."""
        if event.id not in self.scheduled_events:
            raise ValueError(
                f"Attempted to remove event {event.name} (ID: {event.id}), which is not scheduled"
            )
        _, day, start_slot, end_slot = self.scheduled_events[event.id]
        for slot in range(start_slot, end_slot):
            self.days[day][slot] = None

        prev_event, _ = self.get_previous_event(day, start_slot)
        next_event, _ = self.get_next_event(day, end_slot)
        travel_cost_change = self.calculate_travel_cost_change_for_event(
            event, prev_event, next_event, add_event=False)
        self.total_cost += (-event.cost + travel_cost_change)
        del self.scheduled_events[event.id]

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
                day_travel_cost += self.travel_cost_mat.get(
                    (prev_event.id, event.id), self.min_travel_cost)
                # Add event cost
                day_event_cost += event.cost
                prev_event = event
        # Add cost to return to hotel
        if prev_event != self.itinerary_start_event:
            day_travel_cost += self.travel_cost_mat.get(
                (prev_event.id, self.itinerary_start_event.id),
                self.min_travel_cost)

        logger.info(
            f"Day {day_index} event cost: {day_event_cost}, travel cost: {day_travel_cost}")
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
                travel_time = self.travel_time_mat.get(
                    (prev_event.id, event.id), self.min_travel_time)
                day_travel_time += travel_time
                # we don't count the gap time for the first event (hotel)
                if prev_event != self.itinerary_start_event:
                    # current event start slot - previous event end slot - travel time
                    day_gap_time += self.scheduled_events[event.id][2] - \
                        self.scheduled_events[prev_event.id][3] - travel_time
                prev_event = event
        # Add cost to return to hotel
        if prev_event != self.itinerary_start_event:
            day_travel_time += self.travel_time_mat.get(
                (prev_event.id, self.itinerary_start_event.id),
                self.min_travel_time)

        logger.info(f"Day {day_index} travel time: {day_travel_time}")
        return day_travel_time, day_gap_time

    def calculate_total_travel_time_and_gap_time(self) -> Tuple[int, int]:
        """Calculate the total travel time and gap time of all scheduled events across all days."""
        total_travel_time = 0
        total_gap_time = 0
        for day_index in range(self.num_days):
            day_travel_time, day_gap_time = self.calculate_day_travel_and_gap_time(
                day_index)
            total_travel_time += day_travel_time
            total_gap_time += day_gap_time
        return total_travel_time, total_gap_time

    def __str__(self) -> str:
        """Return a human-readable schedule."""
        schedule_str = ""
        for day in range(self.num_days):
            schedule_str += f"\nDay {day + 1}:\n"
            current_event = None
            start_slot = None
            min_unit = 60*24//48
            for slot in range(self.day_resolution):
                event = self.days[day][slot]
                if event != current_event:
                    if current_event is not None:
                        time_start = (
                            f"{start_slot // 2:02d}:{(start_slot % 2) * min_unit:02d}"
                        )
                        time_end = f"{(slot) // 2:02d}:{((slot) % 2) * min_unit:02d}"
                        schedule_str += (
                            f"  {time_start}-{time_end}: {current_event.id}\n"
                        )
                    current_event = event
                    start_slot = slot
            # Handle the last event of the day
            if current_event is not None:
                time_start = f"{start_slot // 2:02d}:{(start_slot % 2) * min_unit:02d}"
                time_end = f"{(slot) // 2:02d}:{((slot) % 2) * min_unit:02d}"
                schedule_str += f"  {time_start}-{time_end}: {current_event.id}\n"
        return schedule_str


def score_day_itinerary(itinerary: Itinerary, day_index: int,
                        w_xp: float, w_count: float, w_cost: float, w_dur: float,
                        w_travel_time: float, w_gap: float) -> float:
    """Score the itinerary for a given day."""
    score = 0.0
    for event, day, start_slot, end_slot in itinerary.scheduled_events.values():
        if day == day_index:
            score += score_event(event, end_slot - start_slot,
                                 w_xp, w_count, w_cost, w_dur)
    day_travel_time, day_gap_time = itinerary.calculate_day_travel_and_gap_time(
        day_index)
    score += (-w_travel_time * day_travel_time + w_gap * day_gap_time)
    _, day_travel_cost = itinerary.calculate_day_cost(day_index)
    score += (-w_cost * day_travel_cost)
    return score


def score_itinerary(itinerary: Itinerary, w_xp: float, w_count: float, w_cost: float, w_dur: float,
                    w_gap: float, w_travel_time: float) -> float:
    """
    Score the itinerary based on the number of events, the cost of the itinerary, and the duration of the itinerary.
    """
    score = 0.0
    for day_index in range(itinerary.num_days):
        score += score_day_itinerary(itinerary, day_index,
                                     w_xp, w_count, w_cost, w_dur, w_travel_time, w_gap)
    return score
