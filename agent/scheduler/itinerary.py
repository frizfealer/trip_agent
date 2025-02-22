import random
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from enum import IntEnum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1) Define a Data Structure for Events
# -------------------------------
class Day(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

def generate_unique_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())

class Event:
    def __init__(self, name: str, cost: float, duration: int, 
                 base_exp: Optional[float], opening_hours: Optional[Dict[Day, Optional[Tuple[int, int]]]], bonus_exp: float = 0.0, bonus_start: Optional[int] = None, 
                 bonus_end: Optional[int] = None, id: Optional[str] = None):
        self.name = name
        self.cost = cost
        self.duration = duration
        if opening_hours is None:
            self.opening_hours = {day: (20, 35) for day in Day}
        else:
            self.opening_hours = opening_hours
        if base_exp is None:
            self.base_exp = 4.0
        else:
            self.base_exp = base_exp
        self.bonus_exp = bonus_exp
        self.bonus_start = bonus_start
        self.bonus_end = bonus_end
        # Use generate_unique_id() instead of name as default ID
        self.id = id if id is not None else generate_unique_id()

    def __str__(self) -> str:
        """Return a human-readable string representation of the event."""
        # Format opening hours
        hours_str = "\n".join(
            f"  {day.name}: {f'{times[0]//2:02d}:{(times[0]%2)*30:02d}-{times[1]//2:02d}:{(times[1]%2)*30:02d}' if times is not None else 'Closed'}"
            for day, times in self.opening_hours.items()
        )
        
        # Format bonus time if applicable
        bonus_time = ""
        if self.bonus_exp > 0 and self.bonus_start is not None and self.bonus_end is not None:
            bonus_time = f"\nBonus Time: {self.bonus_start//2:02d}:{(self.bonus_start%2)*30:02d}-{self.bonus_end//2:02d}:{(self.bonus_end%2)*30:02d}"
            bonus_time += f"\nBonus EXP: {self.bonus_exp}"

        return (f"Event: {self.name} (ID: {self.id})\n"
                f"Duration: {self.duration//2}h{(self.duration%2)*30}m\n"
                f"Cost: ${self.cost:.2f}\n"
                f"Base EXP: {self.base_exp}"
                f"{bonus_time}\n"
                f"Opening Hours:\n{hours_str}")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.id == other.id

class Itinerary:
    def __init__(self, start_day: Day, num_days: int):
        self.start_day = start_day
        self.num_days = num_days
        # Initialize empty slots for each day (48 slots per day)
        self.days = [[None for _ in range(48)] for _ in range(num_days)]
        # Track which events are scheduled and their time slots
        # Dict[str, Tuple[int, int, int] maps event_id to list of (day, start_slot, end_slot (not inclusive))
        self.scheduled_events: Dict[str, Tuple[int, int, int]] = {}
        logger.info(f"Created new itinerary starting on {start_day} for {num_days} days")

    def copy(self: 'Itinerary') -> 'Itinerary':
        """Create a deep copy of the itinerary."""
        new_itinerary = Itinerary(start_day=self.start_day, num_days=self.num_days)
        # Copy scheduled events
        new_itinerary.scheduled_events = {
            event_id: day_slots  # day_slots is already a tuple (day, start_slot, end_slot)
            for event_id, day_slots in self.scheduled_events.items()
        }
        new_itinerary.days = [day.copy() for day in self.days]
        return new_itinerary

    def get_day_of_week(self, day_index: int) -> Day:
        """Convert a day index in the itinerary to the actual day of the week."""
        return Day((self.start_day + day_index) % 7)

    def schedule_event(self, event: Event, day: int, start_slot: int, duration: int) -> bool:
        """
        Attempt to schedule an event on a specific day and start time.
        Args:
            event: The event to schedule
            day: The day index to schedule on
            start_slot: The starting time slot
            duration: duration to schedule (can be partial duration)
        Returns: True if successful, False otherwise.
        """
        if not (0 <= day < self.num_days):
            return False
        if event.id in self.scheduled_events:
            return False
        # Get the actual day of the week
        day_of_week = self.get_day_of_week(day)
        # Check if the event is open on this day
        if event.opening_hours.get(day_of_week) is None:
            return False
        day_open_start, day_open_end = event.opening_hours[day_of_week]
        # Check if the event timing falls within opening hours
        if start_slot < day_open_start:
            return False
        # Determine actual duration to schedule
        actual_duration = min(event.duration, duration)
        # Check if all required slots are available
        end_slot = start_slot + actual_duration
        if end_slot > day_open_end or end_slot > 48:  # Can't go beyond closing time or midnight
            return False
        # Check if all slots are empty
        for slot in range(start_slot, end_slot):
            if self.days[day][slot] is not None:
                return False
        # If we get here, we can schedule the event
        for slot in range(start_slot, end_slot):
            self.days[day][slot] = event
        # Update the scheduled_events dictionary
        self.scheduled_events[event.id] = (day, start_slot, end_slot)
        logger.info(f"Successfully scheduled event {event.name} (ID: {event.id}) on day {day} from slot {start_slot} to {end_slot}")
        return True
    
    def get_event_max_duration(self, event: Event, day: int, start_slot: int) -> int:
        """Get the maximum duration of an event on a specific day and start time."""
        day_of_week = self.get_day_of_week(day)
        _, day_open_end = event.opening_hours[day_of_week]
        max_duration = min(event.duration, day_open_end - start_slot)
        for slot in range(start_slot, start_slot + max_duration):
            if self.days[day][slot] is not None:
                max_duration = slot - start_slot
                break
        return max_duration

    def remove_event(self, event_id: str) -> bool:
        if event_id not in self.scheduled_events:
            logger.warning(f"Attempted to remove non-existent event with ID: {event_id}")
            return False

        day, start_slot, end_slot = self.scheduled_events[event_id]
        for slot in range(start_slot, end_slot):
            self.days[day][slot] = None
        del self.scheduled_events[event_id]
        logger.info(f"Successfully removed event with ID: {event_id}")
        return True

    def __str__(self) -> str:
        """Return a human-readable schedule."""
        schedule_str = ""
        for day in range(self.num_days):
            schedule_str += f"\nDay {day + 1}:\n"
            current_event = None
            start_slot = None
            for slot in range(48):
                event = self.days[day][slot]
                if event != current_event:
                    if current_event is not None:
                        time_start = f"{start_slot // 2:02d}:{(start_slot % 2) * 30:02d}"
                        time_end = f"{(slot) // 2:02d}:{((slot) % 2) * 30:02d}"
                        schedule_str += f"  {time_start}-{time_end}: {current_event.id}\n"
                    current_event = event
                    start_slot = slot
            # Handle the last event of the day
            if current_event is not None:
                time_start = f"{start_slot // 2:02d}:{(start_slot % 2) * 30:02d}"
                time_end = f"{(slot) // 2:02d}:{((slot) % 2) * 30:02d}"
                schedule_str += f"  {time_start}-{time_end}: {current_event.id}\n"
        return schedule_str
