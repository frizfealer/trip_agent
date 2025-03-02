import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, FrozenSet

# Assuming the previously defined classes (Day, Event, Itinerary) exist.

@dataclass(order=True)
class State:
    f: float
    itinerary: Itinerary = field(compare=False)
    remaining_budget: float = field(compare=False)
    last_location: str = field(compare=False)
    g: float = field(compare=False)  # g = cost so far (we define cost = -score)

class AStarItineraryScheduler:
    def __init__(self, events: List[Event], start_day: Day, num_days: int, 
                 total_budget: float, travel_cost_matrix: Dict[Tuple[str, str], float],
                 travel_time_matrix: Dict[Tuple[str, str], int],
                 score_fn_weights: Dict[str, float] = None):
        self.events = events
        self.start_day = start_day
        self.num_days = num_days
        self.total_budget = total_budget
        self.travel_cost_matrix = travel_cost_matrix
        self.travel_time_matrix = travel_time_matrix
        self.score_fn_weights = score_fn_weights or {}
        self.event_id_to_event = {event.id: event for event in events}
        
        # Use the same itinerary structure as before
        self.initial_itinerary = Itinerary(start_day, num_days)
    
    def heuristic(self, state: State) -> float:
        # A trivial (admissible) heuristic: assume no additional benefit
        # (i.e. h = 0).  You could sum over unscheduled events the maximum possible
        # score each could contribute, ignoring travel costs.
        return 0.0

    def state_key(self, state: State) -> Tuple[FrozenSet[Tuple[str, Tuple[int, int, int]]], float, str]:
        # Create a key from the scheduled events (order-independent),
        # remaining budget, and last location.
        # Convert scheduled_events dictionary into a frozenset of items.
        return (frozenset(state.itinerary.scheduled_events.items()),
                round(state.remaining_budget, 2),
                state.last_location)

    def generate_successors(self, state: State) -> List[State]:
        successors = []
        # Identify unscheduled events.
        scheduled_ids = set(state.itinerary.scheduled_events.keys())
        unscheduled_events = [e for e in self.events if e.id not in scheduled_ids]
        
        # Try to schedule each unscheduled event in every day and slot.
        for event in unscheduled_events:
            for day in range(state.itinerary.num_days):
                # Get the day-of-week for the current day.
                day_of_week = state.itinerary.get_day_of_week(day)
                # Check if the event is open on this day.
                if event.opening_hours.get(day_of_week) is None:
                    continue
                day_open_start, day_open_end = event.opening_hours[day_of_week]
                # For simplicity, try all possible start slots in the valid window.
                latest_start = min(day_open_end - event.duration, 48 - event.duration)
                for start_slot in range(day_open_start, latest_start + 1):
                    # Copy the itinerary so far.
                    new_itinerary = state.itinerary.copy()
                    # Try to schedule the event (using full duration).
                    if new_itinerary.schedule_event(event, day, start_slot, event.duration):
                        # Compute travel cost/time from the last location to this event.
                        travel_cost = self.travel_cost_matrix.get((state.last_location, event.id), 0)
                        travel_time = self.travel_time_matrix.get((state.last_location, event.id), 0)
                        total_event_cost = event.cost + travel_cost
                        
                        # Budget constraint: only consider if within budget.
                        if state.remaining_budget < total_event_cost:
                            continue
                        
                        new_remaining_budget = state.remaining_budget - total_event_cost
                        new_last_location = event.id
                        
                        # Compute the new itinerary score and convert to cost (we minimize -score)
                        new_score = self.score_itinerary(new_itinerary)
                        new_g = -new_score  # cost so far is negative of the itinerary score
                        
                        # Create new state.
                        new_state = State(
                            f=new_g + self.heuristic(state),
                            itinerary=new_itinerary,
                            remaining_budget=new_remaining_budget,
                            last_location=new_last_location,
                            g=new_g
                        )
                        successors.append(new_state)
                        # We break after the first successful scheduling for this event 
                        # on this day/slot combination to reduce branching.
                        # (Remove break statements if you want to consider all possibilities.)
                        break  # break out of start_slot loop
                # Optionally, break if a scheduling was found on this day.
        return successors

    def score_itinerary(self, itinerary: Itinerary) -> float:
        """
        Compute the overall score of an itinerary by summing the event base scores
        and subtracting travel costs between consecutive events.
        The same as defined in the GreedyItineraryScheduler.
        """
        total_score = 0.0
        last_location = "hotel"
        # Get events in chronological order.
        # scheduled_events: Dict[event_id, (day, start_slot, end_slot)]
        events_ordered = sorted(itinerary.scheduled_events.items(), key=lambda x: (x[1][0], x[1][1]))
        for event_id, (start_day, int_start, int_end) in events_ordered:
            event = self.event_id_to_event[event_id]
            # For scoring, we assume the full duration was scheduled.
            total_score += self.score_event(event, event.duration)
            # Subtract travel cost/time penalty.
            travel_cost = self.travel_cost_matrix.get((last_location, event.id), 0)
            travel_time = self.travel_time_matrix.get((last_location, event.id), 0)
            total_score -= (travel_cost * self.score_fn_weights.get('w_travel', 0.0) + 
                            travel_time * self.score_fn_weights.get('w_time', 0.0))
            last_location = event.id

        # Account for return trip to hotel.
        if last_location != "hotel":
            travel_cost_back = self.travel_cost_matrix.get((last_location, "hotel"), 0)
            travel_time_back = self.travel_time_matrix.get((last_location, "hotel"), 0)
            total_score -= (travel_cost_back * self.score_fn_weights.get('w_travel', 0.0) + 
                            travel_time_back * self.score_fn_weights.get('w_time', 0.0))
        return total_score

    def score_event(self, event: Event, actual_duration: int) -> float:
        """
        Calculate score for a single event based on its duration and the scoring weights.
        This is the same as the one in GreedyItineraryScheduler.
        """
        if actual_duration <= 0:
            return 0.0
        actual_duration = min(actual_duration, event.duration)
        duration_ratio = actual_duration / event.duration
        base_exp = event.base_exp * duration_ratio

        bonus_exp = 0.0
        if hasattr(event, 'bonus_exp') and hasattr(event, 'bonus_start') and hasattr(event, 'bonus_end'):
            if actual_duration >= event.bonus_start:
                bonus_duration = min(actual_duration, event.bonus_end) - event.bonus_start
                bonus_total_possible = event.bonus_end - event.bonus_start
                if bonus_total_possible > 0:
                    bonus_ratio = max(0.0, bonus_duration / bonus_total_possible)
                    bonus_exp = event.bonus_exp * bonus_ratio

        score = (
            (base_exp + bonus_exp) * self.score_fn_weights.get('w_xp', 0.0) +
            1.0 * self.score_fn_weights.get('w_count', 0.0) +
            -event.cost * self.score_fn_weights.get('w_cost', 0.0) +
            -actual_duration * self.score_fn_weights.get('w_dur', 0.0)
        )
        return score

    def search(self, max_iterations: int = 1000) -> Optional[State]:
        """
        Perform an A* search to find the itinerary with the highest overall score.
        We treat cost as negative score so that lower cost corresponds to a higher score.
        """
        # Initial state: empty itinerary, full budget, starting from "hotel".
        initial_score = self.score_itinerary(self.initial_itinerary)
        initial_g = -initial_score
        initial_state = State(
            f=initial_g + self.heuristic(None),  # heuristic(None) since h = 0
            itinerary=self.initial_itinerary,
            remaining_budget=self.total_budget,
            last_location="hotel",
            g=initial_g
        )

        open_set = []
        heapq.heappush(open_set, initial_state)
        visited: Set[Tuple[FrozenSet[Tuple[str, Tuple[int, int, int]]], float, str]] = set()
        
        best_state = initial_state
        iterations = 0

        while open_set and iterations < max_iterations:
            current_state = heapq.heappop(open_set)
            state_key = self.state_key(current_state)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Update best state found so far if this one has a higher itinerary score.
            current_score = -current_state.g
            best_score = -best_state.g
            if current_score > best_score:
                best_state = current_state

            # Generate successors (i.e. try adding additional events).
            successors = self.generate_successors(current_state)
            if not successors:
                # No further events can be scheduled; this is a terminal state.
                iterations += 1
                continue

            for succ in successors:
                succ.f = succ.g + self.heuristic(succ)
                succ_key = self.state_key(succ)
                if succ_key not in visited:
                    heapq.heappush(open_set, succ)
            
            iterations += 1

        return best_state if best_state != initial_state else None

# -------------------------------
# Example usage:
# -------------------------------

# Assume you have defined a list of Event objects: `events`
# and matrices: travel_cost_matrix, travel_time_matrix, and score_fn_weights

# Example:
# events = [ ... ]  # your events here
# travel_cost_matrix = {("hotel", event.id): 10.0 for event in events}  # sample values
# travel_time_matrix = {("hotel", event.id): 15 for event in events}  # sample values
# score_fn_weights = {'w_xp': 1.0, 'w_count': 0.5, 'w_cost': 0.1, 'w_dur': 0.05, 'w_travel': 0.2, 'w_time': 0.1}

# scheduler = AStarItineraryScheduler(
#     events=events,
#     start_day=Day.MONDAY,
#     num_days=3,
#     total_budget=100.0,
#     travel_cost_matrix=travel_cost_matrix,
#     travel_time_matrix=travel_time_matrix,
#     score_fn_weights=score_fn_weights
# )
# best_state = scheduler.search(max_iterations=2000)
# if best_state:
#     print("Best itinerary score:", -best_state.g)
#     best_state.itinerary.print_schedule()
# else:
#     print("No valid itinerary found.")
