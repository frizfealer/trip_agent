import logging
import math
import uuid
from datetime import datetime
from enum import IntEnum
from typing import Dict, Optional, Tuple

from agent.scheduler.defaults import DEFAULT_DAY_RESOLUTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Day(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


DEFAULT_OPENING_HOURS = (20, 35)  # (10:00 PM, 5:30 PM)


def generate_unique_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def convert_time_to_slot(hour: int, minute: int, day_resolution: int = DEFAULT_DAY_RESOLUTION) -> int:
    """Convert a time to a slot number."""
    slot_minute = 24 * 60 // day_resolution
    # e.g. 10:30, day_resolution = 48 -> 21
    return hour * 60 // slot_minute + round(minute / slot_minute)


def convert_slot_to_time(slot: int, day_resolution: int = DEFAULT_DAY_RESOLUTION) -> Tuple[int, int]:
    """Convert a slot number to a time."""
    slot_minute = 24 * 60 // day_resolution
    hour = slot * slot_minute // 60
    minute = (slot * slot_minute) % 60
    return hour, minute


def convert_hours_to_slots(hours: float, day_resolution: int = DEFAULT_DAY_RESOLUTION) -> int:
    """Convert hours to slots."""
    slot_minute = 24 * 60 // day_resolution
    return math.ceil(hours * 60 / slot_minute)


def convert_minutes_to_slots(minutes: int, day_resolution: int = DEFAULT_DAY_RESOLUTION) -> int:
    """Convert minutes to slots."""
    slot_minute = 24 * 60 // day_resolution
    return math.ceil(minutes / slot_minute)


def gen_str_from_slot(slot: int, day_resolution: int) -> str:
    """Generate a string representation of a slot."""
    hour, minute = convert_slot_to_time(slot, day_resolution)
    return f"{hour:02d}:{minute:02d}"


def parse_regular_opening_hours(
    opening_hours_str: str, day_resolution: int = DEFAULT_DAY_RESOLUTION
) -> Dict[Day, Optional[Tuple[int, int]]]:
    """Parse opening hours string into a dictionary mapping Day to (start_slot, end_slot) tuples.

    Args:
        opening_hours_str: String in format "Monday: 9:00 AM – 6:30 PM, Tuesday: 9:00 AM – 6:30 PM, ..."
        Can contain Unicode whitespace characters like \u202f (NARROW NO-BREAK SPACE) and \u2009 (THIN SPACE)
        Can also handle "Open 24 hours" format and times without AM/PM specification

    Returns:
        Dictionary mapping Day enum values to tuples of (start_slot, end_slot) (inclusive of start, exclusive of end)
        where slots are 0-N (each slot represents 30 minutes, 0=00:00, 47=23:30)
    """
    if opening_hours_str == "NA":
        return {day: DEFAULT_OPENING_HOURS for day in Day}

    slot_mins = 24 * 60 // day_resolution
    hours_in_slots = 60 // slot_mins
    # Initialize result dictionary with None values
    result = {day: None for day in Day}

    # Split into individual day strings
    day_strings = opening_hours_str.split(", ")

    for day_str in day_strings:
        # Split day and hours
        day_name, hours = day_str.split(": ")
        day_enum = Day[day_name.strip().upper()]
        # Handle "Closed" case
        if hours.strip() == "Closed":
            result[day_enum] = None
            continue

        # Handle "Open 24 hours" case
        if "Open 24 hours" in hours:
            # All slots for 24 hours
            result[day_enum] = (0, day_resolution)
            continue

        # Split start and end times
        try:
            # Replace Unicode whitespace characters with regular space
            hours = hours.replace("\u202f", " ").replace("\u2009", " ")
            start_time_str, end_time_str = hours.split("–")

            # Clean up any extra spaces and standardize format
            start_time_str = " ".join(start_time_str.split())
            end_time_str = " ".join(end_time_str.split())

            # Handle times without AM/PM specification
            def parse_time(time_str: str) -> datetime:
                time_str = time_str.strip()
                try:
                    # Try parsing with AM/PM format first
                    return datetime.strptime(time_str, "%I:%M %p")
                except ValueError:
                    try:
                        # Try parsing without AM/PM, assume 24-hour format
                        hour, minute = map(int, time_str.split(":"))
                        time_obj = datetime.now().replace(hour=hour, minute=minute)
                        # If end time is before start time, assume PM
                        if "PM" in end_time_str and "AM" not in start_time_str:
                            time_obj = time_obj.replace(hour=(hour + 12) % 24)
                        return time_obj
                    except ValueError:
                        # Try parsing just the hour
                        hour = int(time_str)
                        time_obj = datetime.now().replace(hour=hour, minute=0)
                        if "PM" in end_time_str and "AM" not in start_time_str:
                            time_obj = time_obj.replace(hour=(hour + 12) % 24)
                        return time_obj

            start_time = parse_time(start_time_str)
            end_time = parse_time(end_time_str)

            # Convert to slots (each slot is 30 minutes)
            start_slot = start_time.hour * hours_in_slots + (start_time.minute // slot_mins)
            end_slot = min(day_resolution, end_time.hour * hours_in_slots + (end_time.minute // slot_mins) + 1)
            if end_slot < start_slot:
                end_slot = day_resolution

            result[day_enum] = (start_slot, end_slot)
        except ValueError:
            print(f"Error parsing hours for {day_name}: {hours}")
            result[day_enum] = None

    return result


class Event:

    def __init__(
        self,
        name: str,
        cost: float,
        duration: int,
        base_exp: Optional[float],
        opening_hours: Optional[Dict[Day, Optional[Tuple[int, int]]]],
        bonus_exp: Optional[float] = None,
        bonus_start: Optional[int] = None,
        bonus_end: Optional[int] = None,
        id: Optional[str] = None,
        default_opening_hours: Tuple[int, int] = DEFAULT_OPENING_HOURS,
        day_resolution: int = DEFAULT_DAY_RESOLUTION,
    ):
        """
        Represents an event in the itinerary.

        Args:
            name: The name of the event.
            cost: The cost of the event.
            duration: The duration of the event in time slots.
            base_exp: The base experience points gained from attending the whole event.
            opening_hours: A dictionary mapping days to opening hours.
                Opening hours for a day are represented as a tuple (start, end), inclusive of start and exclusive of end.
                If opening_hours[day] is None, the event is closed on that day.
            bonus_exp: The bonus experience points for the event during bonus time.
            bonus_start: The start time slot of the bonus period (inclusive).
            bonus_end: The end time slot of the bonus period (exclusive).
            id: A unique identifier for the event. Defaults to a UUID.
            default_opening_hours: The default opening hours for the event, used if `opening_hours` is None.
            day_resolution: The number of time slots per day.
        """
        self.name = name
        self.cost = cost
        self.duration = duration
        if opening_hours is None:
            self.opening_hours = {day: default_opening_hours for day in Day}
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
        self.day_resolution = day_resolution

    def __str__(self) -> str:
        """Return a human-readable string representation of the event."""
        # Format opening hours

        hours_str = "\n".join(
            [
                (
                    f" {day.name}: "
                    + f"{gen_str_from_slot(times[0], self.day_resolution)}-{gen_str_from_slot(times[1], self.day_resolution)}"
                    if times is not None
                    else f" {day.name}: closed"
                )
                for day, times in self.opening_hours.items()
            ]
        )

        # Format bonus time if applicable
        bonus_time = ""
        if self.bonus_exp is not None and self.bonus_start is not None and self.bonus_end is not None:
            bonus_time = (
                "\nBonus Time: "
                + f"{gen_str_from_slot(self.bonus_start, self.day_resolution)}-"
                + f"{gen_str_from_slot(self.bonus_end, self.day_resolution)}"
            )
            bonus_time += f"\nBonus EXP: {self.bonus_exp}"
        duration_hour, duration_minute = convert_slot_to_time(self.duration, self.day_resolution)
        return (
            f"Event: {self.name} (ID: {self.id})\n"
            f"Duration: {duration_hour}h{duration_minute}m\n"
            f"Cost: ${self.cost:.2f}\n"
            f"Base EXP: {self.base_exp}"
            f"{bonus_time}\n"
            f"Opening Hours:\n{hours_str}"
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.id == other.id


def score_event(
    event: Event,
    actual_duration: int,
    w_xp: float,
    w_count: float,
    w_cost: float,
    w_dur: float,
) -> float:
    """
    Calculates the score for a single event based on its duration, cost, and experience points.

    The scoring formula is:
    score = (base_exp * (actual_duration/event.duration)) * w_xp + 1.0 * w_count - \
             event.cost * w_cost - actual_duration * w_dur

    Args:
        event (Event): The event to score
        actual_duration (int): The actual duration the event will be attended for.
            You only get partial experience points if you don't attend the whole event.
        w_xp (float): weight for experience points.
            The higher the weight, the more important it is to attend the event for the whole duration.
        w_count (float): weight for event count.
            The higher the weight, the more important to attend multiple events.
        w_cost (float): weight for monetary cost.
            The higher the weight, the more important to avoid expensive events.
        w_dur (float): weight for time duration.
            The higher the weight, the more important to avoid long events.

    Returns:
        float: The calculated score for the event. Returns 0.0 if actual_duration <= 0
    """
    if actual_duration <= 0:
        return 0.0
    actual_duration = min(actual_duration, event.duration)
    duration_ratio = actual_duration / event.duration
    base_exp = event.base_exp * duration_ratio
    score = (base_exp) * w_xp + 1.0 * w_count + event.cost * w_cost + actual_duration * w_dur
    return score
