from typing import List, Optional

from pydantic import BaseModel, Field


class TripPreference(BaseModel):
    """Trip preference of a trip specified by the user."""

    trip_days: int = Field(..., description="Duration of the trip in days")
    people_count: int = Field(..., description="Number of people in the group")
    location: str = Field(..., description="Location of the trip")
    budget: float = Field(..., description="Budget for the trip in USD")
    interests: Optional[List[str]] = Field(
        None, description="List of interests for the trip"
    )
