"""Data loading and validation for F1 track information."""

import csv
import logging
from pathlib import Path

from pydantic import ValidationError

from .schemas import TrackSchema

logger = logging.getLogger(__name__)


class CsvTrackLoaderError(Exception):
    """Base exception for CSV track loading errors."""

    pass


class CsvTrackLoader:
    """Loads and validates F1 track data from CSV files.

    This loader reads track data from a CSV file and validates each record
    using the TrackSchema. It returns a track database (dictionary) mapping
    track IDs to validated TrackSchema instances, which can be used for
    distance matrix calculations and fitness function evaluation in the
    genetic algorithm.

    Attributes:
        file_path: Path to the CSV file containing track data.

    Example:
        >>> loader = CsvTrackLoader('data/data.csv')
        >>> track_db = loader.load()
        >>> melbourne_track = track_db[1]
        >>> print(melbourne_track.city)
        Melbourne
    """

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the loader with a path to the CSV file.

        Args:
            file_path: Path to the CSV file (string or Path object).

        Raises:
            CsvTrackLoaderError: If the file path is invalid or file doesn't exist.
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise CsvTrackLoaderError(
                f"Track data file not found at: {self.file_path.resolve()}"
            )

        if not self.file_path.is_file():
            raise CsvTrackLoaderError(f"Path is not a file: {self.file_path}")

        if self.file_path.suffix.lower() != '.csv':
            raise CsvTrackLoaderError(
                f"Expected CSV file, got: {self.file_path.suffix}"
            )

    def load(self) -> dict[int, TrackSchema]:
        """Load and validate track data from the CSV file.

        Reads the CSV file, validates each row against TrackSchema, and returns
        a dictionary mapping track IDs to TrackSchema instances.

        Returns:
            A dictionary mapping track ID (int) to TrackSchema instance.
            Example: {1: TrackSchema(...), 2: TrackSchema(...), ...}

        Raises:
            CsvTrackLoaderError: If the file cannot be read or contains invalid data.

        Example:
            >>> loader = CsvTrackLoader('data/data.csv')
            >>> tracks = loader.load()
            >>> print(f"Loaded {len(tracks)} tracks")
            Loaded 24 tracks
        """
        track_db: dict[int, TrackSchema] = {}
        invalid_rows: list[tuple[int, str]] = []

        try:
            with self.file_path.open('r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                if reader.fieldnames is None:
                    raise CsvTrackLoaderError("CSV file is empty or has no headers")

                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    try:
                        track = TrackSchema(**row)
                        track_db[track.id] = track
                        logger.debug(
                            f"Loaded track {track.id}: {track.city} ({track.country})"
                        )
                    except ValidationError as e:
                        error_msg = f"Row {row_num}: {e.error_count()} validation error(s)"
                        invalid_rows.append((row_num, error_msg))
                        logger.warning(f"Skipped {error_msg}")

        except FileNotFoundError as e:
            raise CsvTrackLoaderError(f"Could not open file: {self.file_path}") from e
        except (csv.Error, UnicodeDecodeError) as e:
            raise CsvTrackLoaderError(f"Error reading CSV file: {e}") from e

        if not track_db:
            raise CsvTrackLoaderError(
                "No valid tracks loaded from CSV. "
                f"{len(invalid_rows)} rows failed validation."
            )

        if invalid_rows:
            logger.warning(
                f"Loaded {len(track_db)} tracks with {len(invalid_rows)} invalid rows"
            )

        logger.info(
            f"Successfully loaded {len(track_db)} tracks from {self.file_path.name}"
        )
        return track_db

