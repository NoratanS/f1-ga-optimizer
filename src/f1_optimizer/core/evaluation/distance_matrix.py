import numpy as np
from typing import TYPE_CHECKING

from .utils import haversine_distance

if TYPE_CHECKING:
    from f1_optimizer.data_access.schemas import TrackSchema


class DistanceMatrix:
    """Pre-computed cache for O(1) fitness evaluation lookups.

    Pre-computes distance, temperature, UTC offset, and region data from track_db
    into numpy arrays. Uses 0-based indexing internally; 1-based race IDs handled
    automatically via accessor methods.
    """

    def __init__(self, track_db: dict[int, "TrackSchema"]):
        """Initialize from track database. Race IDs must be consecutive (1-N)."""
        self._validate_track_db(track_db)

        # Derive dimensions from actual data (no magic numbers)
        self.num_races = len(track_db)
        self.num_weeks = len(next(iter(track_db.values())).temperatures)

        # Pre-compute all matrices
        self.distance_matrix = self._compute_distance_matrix(track_db)
        self.temperature_matrix = self._compute_temperature_matrix(track_db)
        self.utc_offsets = self._compute_utc_offsets(track_db)
        self.utc_difference_matrix = self._compute_utc_difference_matrix()
        self.regions = self._extract_regions(track_db)

    def _validate_track_db(self, track_db: dict[int, "TrackSchema"]) -> None:
        """Validate track_db has consecutive race IDs and consistent temperature counts."""
        if not track_db:
            raise ValueError("track_db cannot be empty")

        n_races = len(track_db)
        
        # Algorithmic requirement: race IDs must be consecutive integers 1 to N
        expected_ids = set(range(1, n_races + 1))
        actual_ids = set(track_db.keys())
        if expected_ids != actual_ids:
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            msg = f"Race IDs must be consecutive integers from 1 to {n_races}. "
            if missing:
                msg += f"Missing: {missing}. "
            if extra:
                msg += f"Extra: {extra}."
            raise ValueError(msg)

        # Check all tracks have consistent season length
        first_track = next(iter(track_db.values()))
        expected_weeks = len(first_track.temperatures)
        
        for race_id, track in track_db.items():
            if len(track.temperatures) != expected_weeks:
                raise ValueError(
                    f"Track {race_id} has {len(track.temperatures)} temperatures, "
                    f"but expected {expected_weeks} (all tracks must have same season length)"
                )

    def _compute_distance_matrix(self, track_db: dict[int, "TrackSchema"]) -> np.ndarray:
        """Pre-compute haversine distances between all track pairs."""
        D = np.zeros((self.num_races, self.num_races), dtype=np.float32)

        for race_id_1 in range(1, self.num_races + 1):
            track1 = track_db[race_id_1]
            for race_id_2 in range(1, self.num_races + 1):
                track2 = track_db[race_id_2]

                dist = haversine_distance(
                    track1.latitude,
                    track1.longitude,
                    track2.latitude,
                    track2.longitude,
                )

                # Store in 0-indexed array
                D[race_id_1 - 1, race_id_2 - 1] = dist

        return D

    def _compute_temperature_matrix(
        self, track_db: dict[int, "TrackSchema"]
    ) -> np.ndarray:
        """Pre-compute temperatures for all tracks and weeks."""
        T = np.zeros((self.num_races, self.num_weeks), dtype=np.float32)

        for race_id in range(1, self.num_races + 1):
            track = track_db[race_id]
            temps = np.array(track.temperatures, dtype=np.float32)
            T[race_id - 1, :] = temps

        return T

    def _compute_utc_offsets(self, track_db: dict[int, "TrackSchema"]) -> np.ndarray:
        """Extract UTC offsets for all tracks."""
        utc = np.zeros(self.num_races, dtype=np.int8)

        for race_id in range(1, self.num_races + 1):
            track = track_db[race_id]
            utc[race_id - 1] = track.utc_offset

        return utc

    def _compute_utc_difference_matrix(self) -> np.ndarray:
        """Pre-compute shortest circular UTC distances between all track pairs."""
        UTC_diff = np.zeros((self.num_races, self.num_races), dtype=np.int8)

        for i in range(self.num_races):
            for j in range(self.num_races):
                utc_i = self.utc_offsets[i]
                utc_j = self.utc_offsets[j]
                # Shortest circular distance
                diff = min(abs(utc_j - utc_i), 24 - abs(utc_j - utc_i))
                UTC_diff[i, j] = diff

        return UTC_diff

    def _extract_regions(self, track_db: dict[int, "TrackSchema"]) -> list[str]:
        """Extract region assignment for all tracks."""
        regions = [""] * self.num_races

        for race_id in range(1, self.num_races + 1):
            track = track_db[race_id]
            regions[race_id - 1] = track.region

        return regions

    # ===========================
    # Accessor Methods
    # ===========================

    def get_distance(self, race_id_1: int, race_id_2: int) -> float:
        """Get distance between two tracks in kilometers."""
        return float(self.distance_matrix[race_id_1 - 1, race_id_2 - 1])

    def get_temperature(self, race_id: int, week: int) -> float:
        """Get temperature at a track during a specific week in °C."""
        return float(self.temperature_matrix[race_id - 1, week])

    def get_utc_offset(self, race_id: int) -> int:
        """Get UTC offset for a track in hours."""
        return int(self.utc_offsets[race_id - 1])

    def get_region(self, race_id: int) -> str:
        """Get region assignment for a track."""
        return self.regions[race_id - 1]

    # ===========================
    # Vectorized Operations
    # ===========================

    def get_all_distances_from(self, race_id: int) -> np.ndarray:
        """Get distances from a track to all others."""
        return self.distance_matrix[race_id - 1, :]

    def get_all_temperatures_for_track(self, race_id: int) -> np.ndarray:
        """Get all temperatures for a track across all weeks."""
        return self.temperature_matrix[race_id - 1, :]

    def get_temperatures_at_week(self, week: int) -> np.ndarray:
        """Get temperatures at all tracks during a specific week."""
        return self.temperature_matrix[:, week]

    def get_circular_utc_distances(self, race_id_1: int, race_id_2: int) -> int:
        """Get shortest timezone distance between two tracks (0-12 hours)."""
        return int(self.utc_difference_matrix[race_id_1 - 1, race_id_2 - 1])

    def get_races_by_region(self, region: str) -> set[int]:
        """Get all race IDs in a specific region."""
        return {race_id for race_id in range(1, self.num_races + 1) 
                if self.get_region(race_id) == region}
