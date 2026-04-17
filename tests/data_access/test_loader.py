import pytest
from pathlib import Path

from f1_optimizer.data_access.loader import CsvTrackLoader, CsvTrackLoaderError
from f1_optimizer.data_access.schemas import TrackSchema


class TestCsvTrackLoaderInitialization:
    """Tests for CsvTrackLoader initialization."""

    def test_init_valid_csv_file(self, tmp_path: Path) -> None:
        """CsvTrackLoader should initialize with a valid CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")
        
        loader = CsvTrackLoader(csv_file)
        assert loader.file_path == csv_file

    def test_init_accepts_string_or_path(self, tmp_path: Path) -> None:
        """CsvTrackLoader should accept both Path objects and strings."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")
        
        loader = CsvTrackLoader(str(csv_file))
        assert loader.file_path == csv_file

    def test_init_rejects_nonexistent_file(self) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError for nonexistent file."""
        with pytest.raises(CsvTrackLoaderError, match="not found"):
            CsvTrackLoader("/nonexistent/path/data.csv")

    def test_init_rejects_directory(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError if path is a directory."""
        with pytest.raises(CsvTrackLoaderError, match="not a file"):
            CsvTrackLoader(tmp_path)

    def test_init_rejects_non_csv_file(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise CsvTrackLoaderError for non-CSV files."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")
        
        with pytest.raises(CsvTrackLoaderError, match="CSV file"):
            CsvTrackLoader(txt_file)


class TestCsvTrackLoaderLoad:
    """Tests for CsvTrackLoader.load() method.
    
    All tests verify the all-or-nothing data integrity approach:
    - No partial loading
    - Any error causes complete failure
    - Error messages direct users to fix the source data
    """

    def test_load_single_valid_track(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load a single valid track."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert len(track_db) == 1
        assert isinstance(track_db[1], TrackSchema)
        assert track_db[1].city == "Melbourne"
        assert track_db[1].id == 1

    def test_load_multiple_tracks(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load multiple valid tracks."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            f'2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"{temps}"\n'
            f'3,Suzuka,Japan,Suzuka International Racing Course,Asia,34.8431,136.5410,9,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert len(track_db) == 3
        assert all(i in track_db for i in [1, 2, 3])

    def test_load_parses_temperature_json(self, tmp_path: Path) -> None:
        """CsvTrackLoader should parse temperature JSON strings correctly."""
        csv_file = tmp_path / "data.csv"
        temps = [15.5, 16.2, 17.1, 18.0, 19.5, 20.1, 21.3, 22.0, 23.5, 24.1, 
                 25.2, 26.0, 27.5, 28.1, 29.2, 30.0, 31.5, 32.1, 33.2, 34.0,
                 35.5, 36.1, 37.2, 38.0, 39.5, 40.1, 41.2, 42.0, 43.5, 44.1,
                 45.2, 46.0, 47.5, 48.1, 49.2, 50.0, 51.5, 52.1, 53.2, 54.0]
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Test,Country,Circuit,Region,0.0,0.0,0,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert len(track_db[1].temperatures) == 40
        assert all(isinstance(t, float) for t in track_db[1].temperatures)

    def test_load_accepts_negative_temperatures(self, tmp_path: Path) -> None:
        """CsvTrackLoader should handle negative temperature values."""
        csv_file = tmp_path / "data.csv"
        temps = [-5.0, -4.0, -3.0, -2.0, -1.0] + list(range(0, 35))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'7,Montreal,Canada,Circuit Gilles-Villeneuve,North America,45.5000,-73.5228,-5,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert any(t < 0 for t in track_db[7].temperatures)

    def test_load_rejects_empty_csv(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise error for empty CSV (headers only)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n")
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="No valid tracks"):
            loader.load()

    def test_load_rejects_empty_file(self, tmp_path: Path) -> None:
        """CsvTrackLoader should raise error for completely empty file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("")
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)empty"):
            loader.load()

    def test_load_rejects_invalid_row_bad_data(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject invalid data (no partial loading)."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            '999,Bad,City,Circuit,Region,invalid_latitude,144.9680,10,"[1,2,3]"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)invalid"):
            loader.load()

    def test_load_rejects_missing_required_field(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject rows with missing required fields."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            '1,Melbourne,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[1,2,3]"\n'  # Missing Country
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)invalid"):
            loader.load()

    def test_load_rejects_too_few_temperatures(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject rows with fewer than 40 temperatures."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            '1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[1,2,3]"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)invalid"):
            loader.load()

    def test_load_rejects_too_many_temperatures(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject rows with more than 40 temperatures."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 51))  # 50 temperatures
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)invalid"):
            loader.load()

    def test_load_rejects_empty_temperature_list(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject rows with empty temperature lists."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            '1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[]"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)invalid"):
            loader.load()

    def test_load_returns_dict_type(self, tmp_path: Path) -> None:
        """CsvTrackLoader should return a dictionary."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert isinstance(track_db, dict)


class TestCsvTrackLoaderDuplicateIds:
    """Tests for handling duplicate track IDs (all-or-nothing validation)."""

    def test_load_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        """CsvTrackLoader should reject CSV with duplicate track IDs."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            f'1,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)duplicate"):
            loader.load()

    def test_load_rejects_multiple_duplicate_ids(self, tmp_path: Path) -> None:
        """CsvTrackLoader should detect multiple sets of duplicate IDs."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            f'1,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"{temps}"\n'
            f'2,Suzuka,Japan,Suzuka International Racing Course,Asia,34.8431,136.5410,9,"{temps}"\n'
            f'2,Bahrain,Bahrain,Bahrain International Circuit,Middle East,26.0325,50.5106,3,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError, match="(?i)duplicate"):
            loader.load()

    def test_duplicate_error_includes_conflicting_ids(self, tmp_path: Path) -> None:
        """Error message should list which IDs are duplicated."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'5,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            f'5,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        with pytest.raises(CsvTrackLoaderError) as exc_info:
            loader.load()
        assert "5" in str(exc_info.value)


class TestCsvTrackLoaderRealData:
    """Integration tests with realistic F1 track data."""

    def test_load_real_f1_tracks(self, tmp_path: Path) -> None:
        """CsvTrackLoader should load real F1 track data correctly."""
        csv_file = tmp_path / "f1_tracks.csv"
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            '1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"[18.8, 20.2, 18.7, 17.0, 16.1, 15.8, 16.5, 14.5, 13.9, 12.8, 12.3, 10.7, 11.8, 10.9, 10.4, 9.5, 10.2, 9.3, 9.6, 9.8, 9.8, 9.9, 10.4, 10.8, 11.3, 12.0, 12.3, 12.5, 13.0, 12.4, 14.1, 13.3, 14.0, 15.0, 14.0, 16.1, 14.7, 15.3, 17.4, 18.1]"\n'
            '2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"[9.2, 12.0, 11.0, 14.0, 13.6, 16.4, 17.0, 17.7, 18.4, 19.5, 21.3, 20.9, 22.5, 23.1, 24.1, 25.6, 25.8, 28.3, 29.6, 29.4, 29.0, 28.8, 30.0, 29.8, 29.8, 28.4, 27.2, 26.4, 25.8, 24.0, 24.0, 21.1, 20.5, 17.0, 17.6, 16.2, 14.5, 12.7, 11.8, 8.5]"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        assert len(track_db) == 2
        assert all(isinstance(track, TrackSchema) for track in track_db.values())
        assert track_db[1].latitude == -37.8497
        assert len(track_db[1].temperatures) == 40

    def test_load_track_db_accessible_by_id(self, tmp_path: Path) -> None:
        """Track database should be easily accessible by track ID."""
        csv_file = tmp_path / "data.csv"
        temps = list(range(1, 41))
        csv_file.write_text(
            "ID,City,Country,Track_Name,Region,Latitude,Longitude,UTC_Offset,Temperatures\n"
            f'1,Melbourne,Australia,Albert Park Circuit,Oceania,-37.8497,144.9680,10,"{temps}"\n'
            f'2,Shanghai,China,Shanghai International Circuit,Asia,31.3389,121.2200,8,"{temps}"\n'
        )
        
        loader = CsvTrackLoader(csv_file)
        track_db = loader.load()
        
        # Simulating use case for distance matrix and fitness
        melbourne = track_db[1]
        shanghai = track_db[2]
        
        assert melbourne.latitude == -37.8497
        assert melbourne.longitude == 144.9680
        assert shanghai.city == "Shanghai"
        assert len(shanghai.temperatures) == 40
