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
    """Loads and validates F1 track data from CSV files with all-or-nothing validation."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize with CSV file path.
        
        Args:
            file_path: Path to CSV file (string or Path).
            
        Raises:
            CsvTrackLoaderError: If file doesn't exist, is not a file, or isn't CSV.
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
        """Load and validate all tracks from CSV.
        
        Returns:
            dict[int, TrackSchema]: Tracks keyed by ID.
            
        Raises:
            CsvTrackLoaderError: If any validation, duplicate ID, or file read error occurs.
        """
        track_db: dict[int, TrackSchema] = {}
        invalid_rows: list[tuple[int, str]] = []
        duplicate_ids: list[int] = []
        seen_ids: dict[int, int] = {}  # Maps ID to first row number where it appeared

        try:
            with self.file_path.open('r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                if reader.fieldnames is None:
                    raise CsvTrackLoaderError("CSV file is empty or has no headers")

                for row_num, row in enumerate(reader, start=2):
                    self._validate_and_add_track(
                        row_num, row, track_db, invalid_rows, duplicate_ids, seen_ids
                    )

        except FileNotFoundError as e:
            raise CsvTrackLoaderError(f"Could not open file: {self.file_path}") from e
        except (csv.Error, UnicodeDecodeError) as e:
            raise CsvTrackLoaderError(f"Error reading CSV file: {e}") from e

        # Fail if any data quality issues found
        if duplicate_ids:
            raise CsvTrackLoaderError(
                f"Found {len(duplicate_ids)} duplicate track ID(s): {duplicate_ids}. "
                "Please fix the data source."
            )

        if invalid_rows:
            error_details = "\n".join(
                f"  Row {row_num}: {error_msg}" for row_num, error_msg in invalid_rows
            )
            raise CsvTrackLoaderError(
                f"Found {len(invalid_rows)} invalid row(s) in CSV:\n{error_details}"
            )

        if not track_db:
            raise CsvTrackLoaderError("No valid tracks loaded from CSV.")

        logger.info(
            f"Successfully loaded {len(track_db)} tracks from {self.file_path.name}"
        )
        return track_db

    def _validate_and_add_track(
        self,
        row_num: int,
        row: dict,
        track_db: dict[int, TrackSchema],
        invalid_rows: list[tuple[int, str]],
        duplicate_ids: list[int],
        seen_ids: dict[int, int],
    ) -> None:
        """Validate and add a single track row, handling errors and duplicates.
        
        Args:
            row_num: The row number in the CSV (1-indexed header, so data starts at 2).
            row: The raw row dict from csv.DictReader.
            track_db: The accumulating valid tracks dict.
            invalid_rows: List of (row_num, error_msg) tuples for failed validations.
            duplicate_ids: List of IDs that appear more than once.
            seen_ids: Maps ID to first row number where it appeared (valid or invalid).
        """
        # Extract ID from raw row to check duplicates even if validation fails
        try:
            row_id = int(row.get('ID', -1))
        except (ValueError, TypeError):
            row_id = None
        
        # Check if this ID was seen before (in valid or invalid rows)
        if row_id is not None:
            if row_id in seen_ids:
                if row_id not in duplicate_ids:
                    duplicate_ids.append(row_id)
                logger.error(
                    f"Row {row_num}: Duplicate ID {row_id} (first seen in row {seen_ids[row_id]})"
                )
            else:
                seen_ids[row_id] = row_num
        
        try:
            track = TrackSchema(**row)
            # Only add to track_db if it's not a duplicate
            if track.id not in duplicate_ids:
                track_db[track.id] = track
                logger.debug(f"Loaded {track.id}: {track.city} ({track.country})")

        except ValidationError as e:
            
            error_details = []
            for err in e.errors():
                # Handle cases where loc might be empty (root-level validators)
                if err['loc']:
                    field = err['loc'][0]
                else:
                    field = "root"
                error_details.append(f"{field}: {err['msg']}")
            
            error_summary = "; ".join(error_details)
            invalid_rows.append((row_num, error_summary))
            logger.error(f"Row {row_num}: {error_summary}")

