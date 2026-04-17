import json
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


class TrackSchema(BaseModel):
    """Validation schema (Data Transfer Object) for an F1 track."""

    SEASON_WEEKS: ClassVar[int] = 40  # Number of weeks in the F1 season

    id: int = Field(alias='ID')
    city: str = Field(alias='City')
    country: str = Field(alias='Country')
    track_name: str = Field(alias='Track_Name')
    region: str = Field(alias='Region')
    latitude: float = Field(alias='Latitude')
    longitude: float = Field(alias='Longitude')
    utc_offset: int = Field(alias='UTC_Offset')

    # Temperature list for each week of the season
    temperatures: list[float] = Field(alias='Temperatures')

    @field_validator('temperatures', mode='before')
    @classmethod
    def parse_temperatures_from_string(cls, value: str | list[float]) -> list[float]:
        """Parse temperatures from JSON string or accept a list of floats.

        Args:
            value: Either a JSON-formatted string or a list of floats.

        Returns:
            A list of floats representing temperatures.

        Raises:
            ValueError: If the value cannot be parsed as temperatures.
        """
        if isinstance(value, str):
            # Prevent denial-of-service attacks with excessively large input
            max_size = 1024  # 1 KB limit for JSON temperature string
            if len(value) > max_size:
                raise ValueError(
                    f"Temperature JSON string exceeds maximum size ({max_size} bytes)."
                )
            
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in Temperatures field: {e}") from e
            
            if not isinstance(parsed, list):
                raise ValueError("Temperature JSON must contain a list.")
            
            temps = parsed
        elif isinstance(value, list):
            temps = value
        else:
            raise ValueError(
                f"Temperatures must be a JSON string or list, got {type(value).__name__}."
            )
        
        # Verify all elements are numeric (applies to both JSON and direct list inputs)
        for i, temp in enumerate(temps):
            if not isinstance(temp, (int, float)) or isinstance(temp, bool):
                raise ValueError(
                    f"Temperature at index {i} must be numeric, got {type(temp).__name__} "
                    f"(value: {temp!r})."
                )
        
        return temps

    @field_validator('temperatures', mode='after')
    @classmethod
    def validate_temperature_count(cls, value: list[float]) -> list[float]:
        """Validate that exactly SEASON_WEEKS temperature values are provided.

        Args:
            value: The list of temperature values.

        Returns:
            The validated list.

        Raises:
            ValueError: If the list does not contain exactly SEASON_WEEKS values.
        """
        if len(value) != cls.SEASON_WEEKS:
            raise ValueError(
                f"Expected {cls.SEASON_WEEKS} temperature values "
                f"(one per week of the season), got {len(value)}."
            )
        return value