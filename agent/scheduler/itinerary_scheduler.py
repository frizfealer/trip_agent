import logging
import math
from typing import Dict, List, Tuple

from agent.scheduler.event import convert_hours_to_slots
from agent.scheduler.itinerary import (
    ITINERARY_START_EVENT_NAME,
    Day,
    Event,
    Itinerary,
    score_day_itinerary,
    score_itinerary,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ItineraryScheduler:
    def __init__(
        self,
        events: List[Event],
        start_day: Day,
        num_days: int,
        total_budget: float,
        travel_cost_matrix: Dict[Tuple[str, str], float],
        travel_time_matrix: Dict[Tuple[str, str], int],
        allow_partial_attendance: bool = False,
        largest_waiting_time: int = 8,
    ):
        """
        waiting_time_options: A list of candidate waiting times (in time slots) to try before an event.
                              For example, [0, 2, 4] would mean the algorithm tries 0 slots (no extra wait),
                              2 extra slots, or 4 extra slots.
        """
        self.events = events
        self.start_day = start_day
        self.num_days = num_days
        self.total_budget = total_budget
        self.travel_cost_matrix = travel_cost_matrix
        self.travel_time_matrix = travel_time_matrix
        self.allow_partial_attendance = allow_partial_attendance
        # If not provided, set default candidate waiting times (in time slots).
        self.largest_waiting_time = largest_waiting_time
        self.set_normalization_factors()

    def set_normalization_factors(self):
        self.max_possible_event_exp = max([event.base_exp for event in self.events])
        self.max_possible_event_cost = max([event.cost for event in self.events])
        self.max_possible_event_duration = max([event.duration for event in self.events])
        self.max_possible_event_count = len(self.events)
        self.max_possible_travel_cost = max([value for value in self.travel_cost_matrix.values()])
        self.max_possible_travel_time = max([value for value in self.travel_time_matrix.values()])
        self.max_possible_gap_time = self.largest_waiting_time

    def normalized_score_itinerary(
        self,
        itinerary: Itinerary,
        w_xp: float,
        w_count: float,
        w_cost: float,
        w_dur: float,
        w_travel_time: float,
        w_gap: float,
    ) -> float:
        return score_itinerary(
            itinerary,
            w_xp / self.max_possible_event_exp,
            w_count / self.max_possible_event_count,
            w_cost / self.max_possible_event_cost,
            w_dur / self.max_possible_event_duration,
            w_travel_time / self.max_possible_travel_time,
            w_gap / self.max_possible_gap_time,
        )

    def normalized_score_day_itinerary(
        self,
        itinerary: Itinerary,
        day_index: int,
        w_xp: float,
        w_count: float,
        w_cost: float,
        w_dur: float,
        w_travel_time: float,
        w_gap: float,
    ) -> float:
        return score_day_itinerary(
            itinerary,
            day_index,
            w_xp / self.max_possible_event_exp,
            w_count / self.max_possible_event_count,
            w_cost / self.max_possible_event_cost,
            w_dur / self.max_possible_event_duration,
            w_travel_time / self.max_possible_travel_time,
            w_gap / self.max_possible_gap_time,
        )

    @staticmethod
    def get_score_event_documentation() -> str:
        return ItineraryScheduler.score_event.__doc__

    def greedy_schedule(self, score_fn_weights: Dict[str, float]) -> Itinerary:
        """Schedule events using a greedy algorithm that maximizes the weighted score of the itinerary.

        This function implements a greedy scheduling algorithm that attempts to create an optimal itinerary by:
        1. Iterating through each day in the itinerary
        2. For each day, trying to schedule remaining events at different time slots and with different durations
        3. Selecting the event, start time, and duration combination that gives the highest score improvement

        The scoring mechanism considers multiple factors weighted by the provided parameters:
        - Event experience value (w_xp): Base experience points scaled by actual/planned duration ratio
        - Event count (w_count): Fixed value per event
        - Costs (w_cost): Both event costs and travel costs
        - Duration (w_dur): Time spent at events
        - Travel time (w_travel_time): Time spent traveling between events
        - Gap time (w_gap): Unused time between events

        The algorithm ensures all constraints are satisfied:
        - Events are scheduled within their opening hours
        - Total costs stay within budget
        - Travel times between events are feasible
        - No event overlapping
        - Partial event attendance if allowed
        - Each event is scheduled at most once.

        Args:
            score_fn_weights (Dict[str, float]): Dictionary containing the following weight parameters:
                - w_xp: Weight for experience value. Higher weight prioritizes events with higher experiential value.
                - w_count: Weight for number of events. Higher weight prioritizes more events.
                - w_cost: Weight for costs (negative for penalties). Higher weight encourages higher expenses.
                - w_dur: Weight for duration (negative for penalties). Higher weight places importance on longer events.
                - w_travel_time: Weight for travel time (negative for penalties). Higher weight encourages more travel time.
                - w_gap: Weight for gap time (negative for penalties). Higher weight increase the idle time.

        Returns:
            Itinerary: A scheduled itinerary object containing all successfully scheduled events
                      and their corresponding time slots.
        """

        # Create a new itinerary with the updated constructor parameters
        itinerary = Itinerary(
            start_day=self.start_day,
            num_days=self.num_days,
            budget=self.total_budget,
            travel_cost_mat=self.travel_cost_matrix,
            travel_time_mat=self.travel_time_matrix,
        )

        duration_percentages = [0.25, 0.5, 0.75, 1.0] if self.allow_partial_attendance else [1.0]

        for day in range(itinerary.num_days):
            day_of_week = itinerary.get_day_of_week(day)
            if self.events == []:
                break
            current_time = max(
                min(
                    [
                        event.opening_hours.get(day_of_week)[0]
                        for event in self.events
                        if event.opening_hours.get(day_of_week) is not None
                    ]
                ),
                convert_hours_to_slots(6),
            )
            latest_start_slot = min(
                itinerary.day_resolution,
                max(
                    [
                        event.opening_hours.get(day_of_week)[1]
                        for event in self.events
                        if event.opening_hours.get(day_of_week) is not None
                    ]
                ),
            )
            original_score = self.normalized_score_day_itinerary(itinerary, day, **score_fn_weights)
            logger.info(f"day {day}: Remaining events are {[event.name for event in self.events]}")
            while current_time < latest_start_slot:
                best_event = None
                best_start_slot = None
                best_partial_duration = None
                best_net_score = 0

                for event in self.events:

                    if event.opening_hours.get(day_of_week) is None:
                        continue
                    allowed_durations = sorted(
                        list(set([math.ceil(event.duration * p) for p in duration_percentages]))
                    )
                    # For each waiting time option, try different candidate start times.
                    for candidate_wait in range(self.largest_waiting_time):
                        for partial_duration in allowed_durations:
                            status = itinerary.schedule_event(
                                event, day, current_time + candidate_wait, partial_duration
                            )
                            if status == 0:
                                score = self.normalized_score_day_itinerary(itinerary, day, **score_fn_weights)
                                current_net_score = score - original_score
                                logger.debug(
                                    f"Day {day}: event {event.name} "
                                    + f"can be scheduled on [{current_time+candidate_wait, current_time+candidate_wait+partial_duration}] "
                                    + f"with duration {partial_duration} get current_net_score: {current_net_score}"
                                )
                                if current_net_score > best_net_score:
                                    best_net_score = current_net_score
                                    best_event = event
                                    best_start_slot = current_time + candidate_wait
                                    best_partial_duration = partial_duration
                                itinerary.unschedule_event(event)
                if best_event is not None:
                    logger.info(
                        f"Day {day}: event {best_event.name} "
                        + f"is scheduled on [{best_start_slot, best_start_slot+best_partial_duration}] "
                        + f"with duration {best_partial_duration}"
                    )
                    itinerary.schedule_event(best_event, day, best_start_slot, best_partial_duration)
                    logger.info(f"Total cost: {itinerary.total_cost}")
                    current_time = best_start_slot + best_partial_duration + itinerary.min_travel_time
                    self.events.remove(best_event)
                else:
                    current_time += 1

        return itinerary


# -------------------------------
# 5) Example Usage
# -------------------------------
if __name__ == "__main__":
    # Define some dummy events
    monday_to_friday = {
        Day.MONDAY: (14, 36),  # 7:00-18:00
        Day.TUESDAY: (14, 36),  # 7:00-18:00
        Day.WEDNESDAY: (14, 36),  # 7:00-18:00
        Day.THURSDAY: (14, 36),  # 7:00-18:00
        Day.FRIDAY: (14, 36),  # 7:00-18:00
        Day.SATURDAY: None,  # Closed
        Day.SUNDAY: None,  # Closed
    }

    events = [
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
            base_exp=120,
        ),
    ]

    # Create a simple travel cost matrix
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

    travel_time_matrix = {}
    for (event1, event2), cost in travel_cost_matrix.items():
        travel_time_matrix[(event1, event2)] = cost / 5
        travel_time_matrix[(event2, event1)] = cost / 5

    # Create scheduler instance
    scheduler = ItineraryScheduler(
        events=events,
        start_day=Day.MONDAY,
        num_days=5,
        total_budget=300,
        travel_cost_matrix=travel_cost_matrix,
        travel_time_matrix=travel_time_matrix,
        allow_partial_attendance=True,
        largest_waiting_time=4,
    )

    # Create schedule with custom weights
    custom_weights = {"w_xp": 4.0, "w_count": 1, "w_cost": -1, "w_dur": -0.2, "w_travel_time": -1, "w_gap": 0.1}
    itinerary = scheduler.greedy_schedule(score_fn_weights=custom_weights)

    # Evaluate and print results
    score = scheduler.normalized_score_itinerary(itinerary, **custom_weights)
    print(f"\nTotal Score: {score:.2f}")
    print("\nSchedule:")
    print(itinerary)  # Use the __str__ method instead of print_schedule

    for day in range(itinerary.num_days):
        print(f"day {day}: {itinerary.calculate_day_cost(day)}")
        print(f"day {day}: {itinerary.calculate_day_travel_and_gap_time(day)}")
