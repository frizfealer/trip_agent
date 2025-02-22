import random
from collections import namedtuple
from typing import List, Optional, Dict, Set, Tuple, Callable, Protocol, Any
from enum import IntEnum
from agent.scheduler.itinerary import Itinerary, Event, Day
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field
from dataclasses import dataclass

DEFAULT_NULL_TRAVEL_COST = 100 #100 USD
DEFAULT_NULL_TRAVEL_TIME = 4 #2hours



class GreedyItineraryScheduler:
    def __init__(self, events: List[Event], start_day: Day, num_days: int, 
                 total_budget: float, travel_cost_matrix: Dict[Tuple[str, str], float],
                 travel_time_matrix: Dict[Tuple[str, str], int],
                 score_fn_weights: Dict[str, float] = None, allow_partial_attendance: bool = False):
        self.events = events
        self.start_day = start_day
        self.num_days = num_days
        self.total_budget = total_budget
        self.travel_cost_matrix = travel_cost_matrix
        self.travel_time_matrix = travel_time_matrix
        self.score_fn_weights = score_fn_weights
        self.event_id_to_event = {event.id: event for event in events}
        self.reset()
        self.allow_partial_attendance = allow_partial_attendance
    def reset(self):
        self.remaining_budget = self.total_budget
        self.scheduled_event_ids = set()

    def score_event(self, event: Event, actual_duration: int) -> float:
        """
        Calculate a weighted score for a single event based on its characteristics and actual scheduled duration.
        
        The score is calculated using a weighted sum of different factors:
        - Experience points (w_xp): Rewards events with higher base experience, scaled by attendance duration
        - Event count (w_count): Fixed bonus for each event scheduled
        - Cost penalty (w_cost): Penalizes expensive events
        - Duration penalty (w_dur): Penalizes longer events
        
        Args:
            event: The Event object to score
            actual_duration: The actual duration this event will be scheduled for (in time slots)
        
        Returns:
            float: The weighted score for this event
            
        Weight Parameters:
            w_xp: Weight for experience points (higher = prefer high-XP events)
            w_count: Weight for including events (higher = prefer more events)
            w_cost: Weight for event cost (higher = avoid expensive events)
            w_dur: Weight for event duration (higher = prefer shorter events)
        
        Example:
            With weights {w_xp: 1.0, w_count: 0.1, w_cost: 0.5, w_dur: 0.2}:
            - An event with high XP (100) but high cost ($50) might score:
              100 * 1.0 + 1 * 0.1 - 50 * 0.5 - duration * 0.2
        """
        # Return 0 for zero or negative durations
        if actual_duration <= 0:
            return 0.0
        
        # Cap actual duration at event's max duration
        actual_duration = min(actual_duration, event.duration)
        
        # Calculate base experience proportional to duration
        duration_ratio = actual_duration / event.duration
        base_exp = event.base_exp * duration_ratio
        


        # Apply weights to different components
        score = (
            (base_exp) * self.score_fn_weights.get('w_xp', 0.0) +
            1.0 * self.score_fn_weights.get('w_count', 0.0) -
            event.cost * self.score_fn_weights.get('w_cost', 0.0) -
            actual_duration * self.score_fn_weights.get('w_dur', 0.0)
        )
        
        return score

    def score_itinerary(self, itinerary: Itinerary) -> float:
        """
        Compute the overall score of an itinerary by combining event scores and travel penalties.
        
        The score includes:
        1. Sum of all individual event scores (see score_event())
        2. Travel penalties between consecutive events
        3. Travel penalties for returning to hotel each day
        
        Args:
            itinerary: The Itinerary object to score
        
        Returns:
            float: The total score for the itinerary
            
        Weight Parameters:
            w_xp: Weight for experience points (higher = prefer high-XP events)
            w_count: Weight for number of events (higher = prefer more events)
            w_cost: Weight for monetary costs (higher = avoid expensive events)
            w_dur: Weight for time duration (higher = prefer shorter events)
            w_travel: Weight for travel costs (higher = prefer cheaper travel)
            w_time: Weight for travel time (higher = prefer shorter travel times)
        
        Example:
            With weights {w_travel: 0.1, w_time: 0.1}:
            - Travel cost of $20 and time of 2 slots between events would subtract:
              20 * 0.1 + 2 * 0.1 = 2.2 from the score
            
        Notes:
            - Travel penalties are applied between consecutive events
            - Additional travel penalties are applied for returning to hotel each day
            - Higher weights for w_travel and w_time will encourage the scheduler to
              cluster events geographically to minimize travel overhead
        """
        total_score = 0.0
        last_location = "hotel"
        
        # Sort events by start day and start time
        events_ordered = sorted(itinerary.scheduled_events.items(), key=lambda x: (x[1][0], x[1][1]))
        last_start_day = -1
        last_end_time = 0
        
        for event_id, (start_day, int_start, int_end) in events_ordered:
            # If this is a new day, reset last_end_time
            if last_start_day != start_day:
                last_end_time = 0
            
            # Add the travel cost between the last event and the hotel if it's a new day
            if last_start_day != -1 and last_start_day != start_day:
                travel_cost = self.travel_cost_matrix.get((last_location, "hotel"), 0)
                travel_time = self.travel_time_matrix.get((last_location, "hotel"), 0)
                total_score -= (travel_cost * self.score_fn_weights.get('w_travel', 0.0) + 
                               travel_time * self.score_fn_weights.get('w_time', 0.0))
                last_location = "hotel"
            
            # Score the event itself
            event = self.event_id_to_event[event_id]
            actual_duration = int_end - int_start
            total_score += self.score_event(event, actual_duration)
            
            # Subtract travel cost from last event to current event
            travel_cost = self.travel_cost_matrix.get((last_location, event.id), 0)
            travel_time = self.travel_time_matrix.get((last_location, event.id), 0)
            total_score -= (travel_cost * self.score_fn_weights.get('w_travel', 0.0) + 
                           travel_time * self.score_fn_weights.get('w_time', 0.0))
            
            last_location = event.id
            last_start_day = start_day
            last_end_time = int_end

        # Only calculate return trip to hotel if there were any events
        if last_location != "hotel":
            travel_cost_back = self.travel_cost_matrix.get((last_location, "hotel"), 0)
            travel_time_back = self.travel_time_matrix.get((last_location, "hotel"), 0)
            total_score -= (travel_cost_back * self.score_fn_weights.get('w_travel', 0.0) + 
                           travel_time_back * self.score_fn_weights.get('w_time', 0.0))
        
        return total_score

    def greedy_schedule(self, score_fn_weights: Dict[str, float] = None) -> Itinerary:
        """
        Build an itinerary day-by-day using a greedy approach.
        
        Args:
            score_fn_weights: Optional dictionary of weights for scoring function.
                             Keys can include: 'w_xp', 'w_count', 'w_cost', 'w_dur', 
                             'w_travel', 'w_time'
        """
        # Update score function weights if provided
        if score_fn_weights:
            self.score_fn_weights = score_fn_weights
        elif self.score_fn_weights is None:
            # Default weights if none provided
            self.score_fn_weights = {
                'w_xp': 1.0,
                'w_count': 0.1, 
                'w_cost': 0.5,
                'w_dur': 0.2,
                'w_travel': 0.1,
                'w_time': 0.1,
            }

        itinerary = Itinerary(start_day=self.start_day, num_days=self.num_days)
        remaining_budget = self.total_budget

        duration_percentages = [0.25, 0.5, 0.75, 1.0] if self.allow_partial_attendance else [1.0]
        # Process each day independently.
        for day in range(itinerary.num_days):
            current_time = 0  # current time slot within the day.
            current_location = "hotel"  # start each day at the hotel.
            day_of_week = itinerary.get_day_of_week(day)

            # While there is time left in the day.
            while current_time < 48:
                best_event = None
                best_start_slot = None
                best_partial_duration = None
                best_net_score = float('-inf')

                # Iterate over all events not yet scheduled.
                for event in self.events:
                    if event.id in itinerary.scheduled_events:
                        continue

                    # Skip if the event is not open on this day.
                    if event.opening_hours.get(day_of_week) is None:
                        continue

                    open_start, open_end = event.opening_hours[day_of_week]
                    # Add travel time to the start time calculation
                    travel_time = self.travel_time_matrix.get((current_location, event.id), 0)
                    # The event cannot start before current time plus travel time and its open time
                    candidate_start = max(current_time + travel_time, open_start)
                    
                    # Determine allowed participation durations (at least 1 slot).
                    allowed_durations = sorted(set(
                        max(1, round(event.duration * p)) for p in duration_percentages
                    ))

                    for partial_duration in allowed_durations:
                        # Check that the event can finish before closing time and midnight.
                        if candidate_start + partial_duration > open_end or candidate_start + partial_duration > 48:
                            continue

                        # Verify that the required slots are free.
                        if not all(itinerary.days[day][slot] is None for slot in range(candidate_start, candidate_start + partial_duration)):
                            continue

                        # Compute travel cost/time from the current location to this event.
                        travel_cost = self.travel_cost_matrix.get((current_location, event.id), 0)
                        # Assume cost scales linearly with participation.
                        partial_cost = event.cost * (partial_duration / event.duration)
                        # Include return to hotel cost in budget check if this would be the last event
                        return_cost = self.travel_cost_matrix.get((event.id, "hotel"), 0)
                        total_event_cost = partial_cost + travel_cost + return_cost
                        
                        if remaining_budget < total_event_cost:
                            continue

                        # Evaluate the event's benefit
                        event_score = self.score_event(event, partial_duration)
                        travel_penalty = (travel_cost * self.score_fn_weights.get('w_travel', 0.0) + 
                                        travel_time * self.score_fn_weights.get('w_time', 0.0))
                        net_score = event_score - travel_penalty

                        if net_score > best_net_score:
                            best_net_score = net_score
                            best_event = event
                            best_start_slot = candidate_start
                            best_partial_duration = partial_duration

                if best_event is not None:
                    # Schedule the best candidate event with the chosen partial duration.
                    success = itinerary.schedule_event(best_event, day, best_start_slot, best_partial_duration)
                    if success:
                        # Deduct the cost (partial event cost plus travel cost) from the remaining budget.
                        travel_cost = self.travel_cost_matrix.get((current_location, best_event.id), 0)
                        partial_cost = best_event.cost * (best_partial_duration / best_event.duration)
                        # Don't deduct return cost yet as we might schedule more events
                        remaining_budget -= (partial_cost + travel_cost)
                        # Advance the current time pointer to after the event ends
                        current_time = best_start_slot + best_partial_duration
                        # Update current location.
                        current_location = best_event.id
                    else:
                        # In the unlikely event that scheduling fails, move forward one slot.
                        current_time += 1
                else:
                    # No event can be scheduled at the current time slot; move forward.
                    current_time += 1
            
            # At end of day, deduct return to hotel cost if we scheduled any events
            if current_location != "hotel":
                return_cost = self.travel_cost_matrix.get((current_location, "hotel"), 0)
                remaining_budget -= return_cost
        return itinerary


# -------------------------------
# 5) Example Usage
# -------------------------------
if __name__ == "__main__":
    # Define some dummy events
    monday_to_friday = {
        Day.MONDAY: (14, 36),    # 7:00-18:00
        Day.TUESDAY: (14, 36),   # 7:00-18:00
        Day.WEDNESDAY: (14, 36), # 7:00-18:00
        Day.THURSDAY: (14, 36),  # 7:00-18:00
        Day.FRIDAY: (14, 36),    # 7:00-18:00
        Day.SATURDAY: None,      # Closed
        Day.SUNDAY: None         # Closed
    }

    events = [
        Event(
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
            name="Art Gallery",
            cost=20,
            duration=2,
            opening_hours=monday_to_friday,
            base_exp=30,
            bonus_exp=10,
            bonus_start=5,
            bonus_end=6
        ),
        Event(
            name="Historical Site",
            cost=40,
            duration=3,
            opening_hours=monday_to_friday,
            base_exp=60,
            bonus_exp=5,
            bonus_start=2,
            bonus_end=3
        ),
        Event(
            name="City Tour",
            cost=70,
            duration=4,
            opening_hours=monday_to_friday,
            base_exp=120,
            bonus_exp=20,
            bonus_start=1,
            bonus_end=2
        ),
    ]

    # Create a simple travel cost matrix
    travel_cost_matrix = {
        ("hotel", events[0].id): 10,
        ("hotel", events[1].id): 15,
        ("hotel", events[2].id): 20,
        ("hotel", events[3].id): 25,
        (events[0].id, "hotel"): 10,
        (events[1].id, "hotel"): 15,
        (events[2].id, "hotel"): 20,
        (events[3].id, "hotel"): 25,
        # Add costs between events
        (events[0].id, events[1].id): 5,
        (events[1].id, events[0].id): 5,
        (events[1].id, events[2].id): 8,
        (events[2].id, events[1].id): 8,
        (events[2].id, events[3].id): 12,
        (events[3].id, events[2].id): 12,
    }

    # Create scheduler instance
    scheduler = GreedyItineraryScheduler(
        events=events,
        start_day=Day.MONDAY,
        num_days=5,
        total_budget=200,
        travel_cost_matrix=travel_cost_matrix,
        travel_time_matrix=travel_cost_matrix
    )

    # Create schedule with custom weights
    custom_weights = {
        'w_xp': 1.0,
        'w_count': 0.1,
        'w_cost': 0.5,
        'w_dur': 0.2,
        'w_travel': 0.1,
        'w_time': 0.1
    }
    itinerary = scheduler.greedy_schedule(score_fn_weights=custom_weights)

    # Evaluate and print results
    score = scheduler.score_itinerary(itinerary)
    print(f"\nTotal Score: {score:.2f}")
    print("\nSchedule:")
    itinerary.print_schedule()
