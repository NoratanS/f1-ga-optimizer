import json
import pytest
from pydantic import ValidationError

from f1_optimizer.data_access.schemas import TrackSchema


class TestTrackSchemaInitialization:
    """Tests for TrackSchema initialization with valid data."""

    def test_init_with_json_temperatures(self) -> None:
        """TrackSchema should parse JSON temperature string."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Melbourne',
            'Country': 'Australia',
            'Track_Name': 'Albert Park Circuit',
            'Region': 'Oceania',
            'Latitude': -37.8497,
            'Longitude': 144.9680,
            'UTC_Offset': 10,
            'Temperatures': '[18.8, 20.2, 18.7]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.id == 1
        assert track.city == 'Melbourne'
        assert track.temperatures == [18.8, 20.2, 18.7]

    def test_init_with_list_temperatures(self) -> None:
        """TrackSchema should accept list of floats for temperatures."""
        # Arrange
        data = {
            'ID': 2,
            'City': 'Shanghai',
            'Country': 'China',
            'Track_Name': 'Shanghai International Circuit',
            'Region': 'Asia',
            'Latitude': 31.3389,
            'Longitude': 121.2200,
            'UTC_Offset': 8,
            'Temperatures': [9.2, 12.0, 11.0, 14.0],
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.id == 2
        assert track.country == 'China'
        assert track.temperatures == [9.2, 12.0, 11.0, 14.0]

    def test_all_fields_present(self) -> None:
        """TrackSchema should contain all required fields."""
        # Arrange
        data = {
            'ID': 3,
            'City': 'Suzuka',
            'Country': 'Japan',
            'Track_Name': 'Suzuka International Racing Course',
            'Region': 'Asia',
            'Latitude': 34.8431,
            'Longitude': 136.5410,
            'UTC_Offset': 9,
            'Temperatures': '[7.5, 9.4, 9.3]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.id == 3
        assert track.city == 'Suzuka'
        assert track.country == 'Japan'
        assert track.track_name == 'Suzuka International Racing Course'
        assert track.region == 'Asia'
        assert track.latitude == 34.8431
        assert track.longitude == 136.5410
        assert track.utc_offset == 9
        assert isinstance(track.temperatures, list)

    def test_full_temperature_array(self) -> None:
        """TrackSchema should handle 40 temperature values (full season)."""
        # Arrange
        temps = list(range(1, 41))  # 40 temperatures
        data = {
            'ID': 4,
            'City': 'Sakhir',
            'Country': 'Bahrain',
            'Track_Name': 'Bahrain International Circuit',
            'Region': 'Middle East',
            'Latitude': 26.0325,
            'Longitude': 50.5106,
            'UTC_Offset': 3,
            'Temperatures': json.dumps(temps),
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert len(track.temperatures) == 40
        assert track.temperatures == temps


class TestTrackSchemaTemperatureValidator:
    """Tests for the temperature field validator."""

    def test_validator_parses_valid_json(self) -> None:
        """Validator should parse valid JSON temperature string."""
        # Arrange
        temps_json = '[15.5, 16.2, 17.1, 18.0]'
        data = {
            'ID': 5,
            'City': 'Test City',
            'Country': 'Test Country',
            'Track_Name': 'Test Track',
            'Region': 'Test Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': temps_json,
        }

        # Act & Assert
        track = TrackSchema(**data)
        assert track.temperatures == [15.5, 16.2, 17.1, 18.0]

    def test_validator_rejects_invalid_json(self) -> None:
        """Validator should raise ValueError for malformed JSON."""
        # Arrange
        data = {
            'ID': 6,
            'City': 'Test City',
            'Country': 'Test Country',
            'Track_Name': 'Test Track',
            'Region': 'Test Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': 'not valid json',
        }

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TrackSchema(**data)
        assert 'Invalid JSON' in str(exc_info.value)

    def test_validator_rejects_json_non_list(self) -> None:
        """Validator should reject JSON that doesn't contain a list."""
        # Arrange
        data = {
            'ID': 7,
            'City': 'Test City',
            'Country': 'Test Country',
            'Track_Name': 'Test Track',
            'Region': 'Test Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': '{"key": "value"}',  # Valid JSON but not a list
        }

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TrackSchema(**data)
        assert 'list' in str(exc_info.value).lower()

    def test_validator_rejects_invalid_type(self) -> None:
        """Validator should reject temperatures of unsupported type."""
        # Arrange
        data = {
            'ID': 8,
            'City': 'Test City',
            'Country': 'Test Country',
            'Track_Name': 'Test Track',
            'Region': 'Test Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': 12345,  # Integer instead of list or JSON string
        }

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TrackSchema(**data)
        assert 'Temperatures must be' in str(exc_info.value)

    def test_validator_accepts_empty_list(self) -> None:
        """Validator should accept empty list (edge case)."""
        # Arrange
        data = {
            'ID': 9,
            'City': 'Test City',
            'Country': 'Test Country',
            'Track_Name': 'Test Track',
            'Region': 'Test Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': [],
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.temperatures == []


class TestTrackSchemaFieldAliasing:
    """Tests for CSV field name aliasing."""

    def test_field_aliases_work(self) -> None:
        """TrackSchema should map CSV column names to Python attributes."""
        # Arrange - use CSV-style field names
        data = {
            'ID': 10,
            'City': 'London',
            'Country': 'United Kingdom',
            'Track_Name': 'Silverstone',
            'Region': 'Europe',
            'Latitude': 52.0786,
            'Longitude': -1.0169,
            'UTC_Offset': 0,
            'Temperatures': '[5.0, 6.0]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert - verify fields are accessible via Python names
        assert track.id == 10
        assert track.city == 'London'
        assert track.country == 'United Kingdom'
        assert track.track_name == 'Silverstone'
        assert track.region == 'Europe'
        assert track.latitude == 52.0786
        assert track.longitude == -1.0169
        assert track.utc_offset == 0


class TestTrackSchemaValidation:
    """Tests for field validation and constraints."""

    def test_missing_required_field(self) -> None:
        """TrackSchema should raise ValidationError for missing required field."""
        # Arrange - missing 'City'
        data = {
            'ID': 11,
            'Country': 'France',
            'Track_Name': 'Circuit Paul Ricard',
            'Region': 'Europe',
            'Latitude': 43.2500,
            'Longitude': 5.7917,
            'UTC_Offset': 1,
            'Temperatures': '[10.0, 11.0]',
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)

    def test_invalid_latitude_type(self) -> None:
        """TrackSchema should validate latitude as float."""
        # Arrange
        data = {
            'ID': 12,
            'City': 'Monaco',
            'Country': 'Monaco',
            'Track_Name': 'Circuit de Monaco',
            'Region': 'Europe',
            'Latitude': 'not a number',
            'Longitude': 7.4206,
            'UTC_Offset': 1,
            'Temperatures': '[12.0, 13.0]',
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)

    def test_invalid_utc_offset_type(self) -> None:
        """TrackSchema should validate UTC_Offset as integer."""
        # Arrange
        data = {
            'ID': 13,
            'City': 'Barcelona',
            'Country': 'Spain',
            'Track_Name': 'Circuit de Barcelona-Catalunya',
            'Region': 'Europe',
            'Latitude': 41.5700,
            'Longitude': 2.2611,
            'UTC_Offset': 'invalid',
            'Temperatures': '[14.0, 15.0]',
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)

    def test_temperature_values_can_be_negative(self) -> None:
        """Validator should accept negative temperature values (winter races)."""
        # Arrange
        data = {
            'ID': 14,
            'City': 'Montreal',
            'Country': 'Canada',
            'Track_Name': 'Circuit Gilles-Villeneuve',
            'Region': 'North America',
            'Latitude': 45.5000,
            'Longitude': -73.5228,
            'UTC_Offset': -5,
            'Temperatures': '[-4.2, -1.1, 1.4, 0.5, -2.0]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.temperatures == [-4.2, -1.1, 1.4, 0.5, -2.0]
        assert any(t < 0 for t in track.temperatures)


class TestTrackSchemaIntegration:
    """Integration tests with real-world data."""

    def test_real_track_data_melbourne(self) -> None:
        """TrackSchema should validate real Melbourne track data."""
        # Arrange - Real Melbourne data
        data = {
            'ID': 1,
            'City': 'Melbourne',
            'Country': 'Australia',
            'Track_Name': 'Albert Park Circuit',
            'Region': 'Oceania',
            'Latitude': -37.8497,
            'Longitude': 144.9680,
            'UTC_Offset': 10,
            'Temperatures': '[18.8, 20.2, 18.7, 17.0, 16.1, 15.8, 16.5, 14.5, 13.9, 12.8, 12.3, 10.7, 11.8, 10.9, 10.4, 9.5, 10.2, 9.3, 9.6, 9.8, 9.8, 9.9, 10.4, 10.8, 11.3, 12.0, 12.3, 12.5, 13.0, 12.4, 14.1, 13.3, 14.0, 15.0, 14.0, 16.1, 14.7, 15.3, 17.4, 18.1]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.id == 1
        assert track.city == 'Melbourne'
        assert len(track.temperatures) == 40
        assert min(track.temperatures) == 9.3
        assert max(track.temperatures) == 20.2

    def test_real_track_data_lusail(self) -> None:
        """TrackSchema should validate real Lusail track data."""
        # Arrange - Real Lusail data
        data = {
            'ID': 23,
            'City': 'Lusail',
            'Country': 'Qatar',
            'Track_Name': 'Lusail International Circuit',
            'Region': 'Middle East',
            'Latitude': 25.4900,
            'Longitude': 51.4542,
            'UTC_Offset': 3,
            'Temperatures': '[21.0, 22.9, 23.2, 22.9, 24.9, 26.7, 26.4, 27.9, 29.2, 30.0, 31.8, 33.4, 34.6, 33.8, 34.7, 35.2, 35.5, 36.4, 35.0, 35.9, 36.5, 36.0, 36.3, 35.8, 35.1, 35.4, 34.5, 34.1, 33.8, 32.7, 31.8, 31.1, 29.9, 29.3, 27.7, 27.0, 25.9, 24.8, 23.5, 21.8]',
        }

        # Act
        track = TrackSchema(**data)

        # Assert
        assert track.id == 23
        assert track.city == 'Lusail'
        assert track.region == 'Middle East'
        assert len(track.temperatures) == 40
        assert all(isinstance(t, float) for t in track.temperatures)
