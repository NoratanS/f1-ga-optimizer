import json
from pydantic import BaseModel, Field, field_validator


class TrackSchema(BaseModel):
    """Validation schema (Data Transfer Object) for an F1 track."""

    id: int = Field(alias='ID')
    city: str = Field(alias='City')
    country: str = Field(alias='Country')
    track_name: str = Field(alias='Track_Name')
    region: str = Field(alias='Region')
    latitude: float = Field(alias='Latitude')
    longitude: float = Field(alias='Longitude')
    utc_offset: int = Field(alias='UTC_Offset')

    # Temperature list for each week of the season (40 weeks)
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
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("JSON must contain a list.")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in Temperatures field: {e}") from e
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(
                f"Temperatures must be a JSON string or list, got {type(value).__name__}."
            )