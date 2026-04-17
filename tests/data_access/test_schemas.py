import pytest
from pydantic import ValidationError

from f1_optimizer.data_access.schemas import TrackSchema


class TestTrackSchemaInitialization:
    """Tests for TrackSchema initialization."""

    def test_init_with_json_temperatures(self) -> None:
        """TrackSchema should parse JSON temperature string."""
        # Arrange
        temps = list(range(1, 41))
        data = {
            'ID': 1,
            'City': 'Melbourne',
            'Country': 'Australia',
            'Track_Name': 'Albert Park Circuit',
            'Region': 'Oceania',
            'Latitude': -37.8497,
            'Longitude': 144.9680,
            'UTC_Offset': 10,
            'Temperatures': str(temps),
        }
        
        # Act
        track = TrackSchema(**data)
        
        # Assert
        assert track.id == 1
        assert track.city == 'Melbourne'
        assert len(track.temperatures) == 40

    def test_init_with_list_temperatures(self) -> None:
        """TrackSchema should accept list of floats."""
        # Arrange
        temps = list(range(1, 41))
        data = {
            'ID': 2,
            'City': 'Shanghai',
            'Country': 'China',
            'Track_Name': 'Shanghai International Circuit',
            'Region': 'Asia',
            'Latitude': 31.3389,
            'Longitude': 121.2200,
            'UTC_Offset': 8,
            'Temperatures': temps,
        }
        
        # Act
        track = TrackSchema(**data)
        
        # Assert
        assert track.id == 2
        assert track.country == 'China'
        assert len(track.temperatures) == 40

    def test_field_aliases_work(self) -> None:
        """TrackSchema should map CSV column names to Python attributes."""
        # Arrange
        temps = list(range(1, 41))
        data = {
            'ID': 10,
            'City': 'Silverstone',
            'Country': 'UK',
            'Track_Name': 'Silverstone Circuit',
            'Region': 'Europe',
            'Latitude': 52.0786,
            'Longitude': -1.0169,
            'UTC_Offset': 0,
            'Temperatures': temps,
        }
        
        # Act
        track = TrackSchema(**data)
        
        # Assert
        assert track.id == 10
        assert track.city == 'Silverstone'
        assert track.track_name == 'Silverstone Circuit'
        assert track.utc_offset == 0


class TestTrackSchemaTemperatureValidation:
    """Tests for temperature field constraints."""

    def test_rejects_too_few_temperatures(self) -> None:
        """TrackSchema should reject fewer than 40 temperatures."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            'Country': 'Country',
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': list(range(1, 30)),  # Only 29
        }
        
        # Act & Assert
        with pytest.raises(ValidationError, match="40"):
            TrackSchema(**data)

    def test_rejects_too_many_temperatures(self) -> None:
        """TrackSchema should reject more than 40 temperatures."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            'Country': 'Country',
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': list(range(1, 50)),  # 49 values
        }
        
        # Act & Assert
        with pytest.raises(ValidationError, match="40"):
            TrackSchema(**data)

    def test_rejects_empty_list(self) -> None:
        """TrackSchema should reject empty temperature list."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            'Country': 'Country',
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': [],
        }
        
        # Act & Assert
        with pytest.raises(ValidationError, match="40"):
            TrackSchema(**data)

    def test_accepts_negative_temperatures(self) -> None:
        """TrackSchema should accept negative temperature values."""
        # Arrange
        temps = [-5.0, -4.0, -3.0, -2.0, -1.0] + list(range(0, 35))
        data = {
            'ID': 1,
            'City': 'Montreal',
            'Country': 'Canada',
            'Track_Name': 'Circuit Gilles-Villeneuve',
            'Region': 'North America',
            'Latitude': 45.5,
            'Longitude': -73.5,
            'UTC_Offset': -5,
            'Temperatures': temps,
        }
        
        # Act
        track = TrackSchema(**data)
        
        # Assert
        assert any(t < 0 for t in track.temperatures)
        assert len(track.temperatures) == 40


class TestTrackSchemaFieldValidation:
    """Tests for field validation."""

    def test_missing_required_field(self) -> None:
        """TrackSchema should require all fields."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            # Missing Country
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': list(range(1, 41)),
        }
        
        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)

    def test_invalid_latitude_type(self) -> None:
        """TrackSchema should validate latitude as float."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            'Country': 'Country',
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 'not a number',
            'Longitude': 0.0,
            'UTC_Offset': 0,
            'Temperatures': list(range(1, 41)),
        }
        
        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)

    def test_invalid_utc_offset_type(self) -> None:
        """TrackSchema should validate UTC_Offset as integer."""
        # Arrange
        data = {
            'ID': 1,
            'City': 'Test',
            'Country': 'Country',
            'Track_Name': 'Track',
            'Region': 'Region',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'UTC_Offset': 'invalid',
            'Temperatures': list(range(1, 41)),
        }
        
        # Act & Assert
        with pytest.raises(ValidationError):
            TrackSchema(**data)


class TestTrackSchemaRealData:
    """Integration tests with real F1 track data."""

    def test_melbourne_track(self) -> None:
        """TrackSchema should validate real Melbourne track data."""
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
            'Temperatures': '[18.8, 20.2, 18.7, 17.0, 16.1, 15.8, 16.5, 14.5, 13.9, 12.8, 12.3, 10.7, 11.8, 10.9, 10.4, 9.5, 10.2, 9.3, 9.6, 9.8, 9.8, 9.9, 10.4, 10.8, 11.3, 12.0, 12.3, 12.5, 13.0, 12.4, 14.1, 13.3, 14.0, 15.0, 14.0, 16.1, 14.7, 15.3, 17.4, 18.1]',
        }
        
        # Act
        track = TrackSchema(**data)
        
        # Assert
        assert track.id == 1
        assert track.city == 'Melbourne'
        assert len(track.temperatures) == 40

    def test_lusail_track(self) -> None:
        """TrackSchema should validate real Lusail track data."""
        # Arrange
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
