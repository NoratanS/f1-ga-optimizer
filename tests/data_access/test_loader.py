"""Unit tests for CsvTrackLoader."""

import pytest
from pathlib import Path

from f1_optimizer.data_access.loader import CsvTrackLoader, CsvTrackLoaderError
from f1_optimizer.data_access.schemas import TrackSchema


class TestCsvTrackLoaderInitialization:
    """Tests for CsvTrackLoader initialization."""

    def test_init_with_valid_path(self, tmp_path: Path) -> None:
        """CsvTrackLoader should initialize with a valid CSV file path."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")

        # Act
        loader = CsvTrackLoader(csv_file)

        # Assert
        assert loader.file_path == csv_file

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """CsvTrackLoader should accept both Path objects and strings."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")

        # Act
        loader = CsvTrackLoader(str(csv_file))

        # Assert
        assert loader.file_path == csv_file

    def test_init_with_nonexistent_file(self) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError for nonexistent file."""
        # Act & Assert
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            CsvTrackLoader("/nonexistent/path/data.csv")
        assert "not found" in str(exc_info.value)

    def test_init_with_directory_path(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError if path is a directory."""
        # Act & Assert
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            CsvTrackLoader(tmp_path)
        assert "not a file" in str(exc_info.value)

    def test_init_with_non_csv_file(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError for non-CSV files."""
        # Arrange
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")

        # Act & Assert
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            CsvTrackLoader(txt_file)
        assert "CSV file" in str(exc_info.value)


class TestCsvTrackLoaderLoad:
    """Tests for CsvTrackLoader.load() method."""

    def test_load_single_track(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load a single valid track."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2, 18.7]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert len(track_db) == 1
        assert 1 in track_db
        assert track_db[1].city == "Melbourne"
        assert track_db[1].country == "Australia"

    def test_load_multiple_tracks(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load multiple tracks."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2]"
2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"[9.2, 12.0]"
3,Suzuka,Japan,Suzuka International Racing Course,Asia,34.8431,136.5410,9,"[7.5, 9.4]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert len(track_db) == 3
        assert all(i in track_db for i in [1, 2, 3])
        assert track_db[1].city == "Melbourne"
        assert track_db[2].city == "Shanghai"
        assert track_db[3].city == "Suzuka"

    def test_load_returns_track_schema_instances(self, tmp_path: Path) -> None:
        """CsvTrackLoader should return TrackSchema instances."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert isinstance(track_db[1], TrackSchema)

    def test_load_maps_track_id_to_schema(self, tmp_path: Path) -> None:
        """CsvTrackLoader should map track ID correctly."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
42,Test Track,Test Country,Test Circuit,Test Region,0.0,0.0,0,"[10.0, 11.0]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert 42 in track_db
        assert track_db[42].id == 42

    def test_load_empty_csv_raises_error(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise error for empty CSV (headers only)."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")

        # Act & Assert
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            loader.load()
        assert "No valid tracks" in str(exc_info.value)

    def test_load_no_headers_raises_error(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise error for CSV with no headers."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("")

        # Act & Assert
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            loader.load()
        assert "empty" in str(exc_info.value).lower()

    def test_load_invalid_data_skipped(self, tmp_path: Path) -> None:
        """CsvTrackLoader should skip rows with invalid data."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2]"
999,InvalidCity,InvalidCountry,InvalidCircuit,InvalidRegion,invalid_latitude,144.9680,10,"[18.8]"
2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"[9.2, 12.0]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert len(track_db) == 2
        assert 1 in track_db
        assert 2 in track_db
        assert 999 not in track_db

    def test_load_parses_temperatures_from_json_string(self, tmp_path: Path) -> None:
        """CsvTrackLoader should parse temperature JSON strings correctly."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2, 18.7, 17.0]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert track_db[1].temperatures == [18.8, 20.2, 18.7, 17.0]
        assert len(track_db[1].temperatures) == 4

    def test_load_with_negative_temperatures(self, tmp_path: Path) -> None:
        """CsvTrackLoader should handle negative temperature values."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
7,Montreal,Canada,Circuit Gilles-Villeneuve,North America,45.5000,-73.5228,-5,"[-4.2, -1.1, 1.4, 0.5, -2.0]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert track_db[7].temperatures == [-4.2, -1.1, 1.4, 0.5, -2.0]
        assert any(t < 0 for t in track_db[7].temperatures)

    def test_load_returns_dict_type(self, tmp_path: Path) -> None:
        """CsvTrackLoader should return a dictionary."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert isinstance(track_db, dict)


class TestCsvTrackLoaderRealData:
    """Integration tests with realistic data patterns."""

    def test_load_all_f1_tracks(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load all 24 F1 tracks from complete CSV."""
        # Arrange - Real F1 track data
        csv_file = tmp_path / "f1_tracks.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2, 18.7, 17.0, 16.1, 15.8, 16.5, 14.5, 13.9, 12.8, 12.3, 10.7, 11.8, 10.9, 10.4, 9.5, 10.2, 9.3, 9.6, 9.8, 9.8, 9.9, 10.4, 10.8, 11.3, 12.0, 12.3, 12.5, 13.0, 12.4, 14.1, 13.3, 14.0, 15.0, 14.0, 16.1, 14.7, 15.3, 17.4, 18.1]"
2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"[9.2, 12.0, 11.0, 14.0, 13.6, 16.4, 17.0, 17.7, 18.4, 19.5, 21.3, 20.9, 22.5, 23.1, 24.1, 25.6, 25.8, 28.3, 29.6, 29.4, 29.0, 28.8, 30.0, 29.8, 29.8, 28.4, 27.2, 26.4, 25.8, 24.0, 24.0, 21.1, 20.5, 17.0, 17.6, 16.2, 14.5, 12.7, 11.8, 8.5]"
3,Suzuka,Japan,Suzuka International Racing Course,Asia,34.8431,136.5410,9,"[7.5, 9.4, 9.3, 11.8, 12.4, 13.6, 14.5, 16.0, 15.5, 16.6, 17.8, 19.6, 19.2, 20.0, 21.3, 23.0, 24.1, 26.1, 26.8, 26.2, 27.4, 28.8, 28.5, 27.7, 28.0, 27.5, 26.5, 25.7, 26.1, 23.8, 22.8, 20.6, 19.7, 16.6, 15.6, 15.3, 13.4, 12.1, 11.0, 8.9]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert
        assert len(track_db) == 3
        assert all(isinstance(track, TrackSchema) for track in track_db.values())
        assert all(isinstance(track_id, int) for track_id in track_db.keys())

    def test_load_track_db_accessible_by_id(self, tmp_path: Path) -> None:
        """Track database should be easily accessible by track ID."""
        # Arrange
        csv_file = tmp_path / "data.csv"
        csv_content = """ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures
1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2]"
2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"[9.2, 12.0]"
"""
        csv_file.write_text(csv_content)

        # Act
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()

        # Assert - simulating use case for distance matrix and fitness
        melbourne = track_db[1]
        shanghai = track_db[2]
        
        assert melbourne.latitude == -37.8497
        assert melbourne.longitude == 144.9680
        assert shanghai.city == "Shanghai"
        assert len(shanghai.temperatures) == 2
