"""Tests for DistanceMatrix class."""

import pytest
import numpy as np

from f1_optimizer.core.evaluation.distance_matrix import DistanceMatrix


class TestDistanceMatrixInitialization:
    """Tests for DistanceMatrix initialization."""

    def test_initialization_with_valid_track_db(self, track_db):
        """Test initialization with valid track database."""
        # Act
        dm = DistanceMatrix(track_db)

        # Assert
        assert dm.num_races == len(track_db)
        assert dm.num_weeks > 0
        assert dm.distance_matrix is not None
        assert dm.temperature_matrix is not None
        assert dm.utc_offsets is not None
        assert dm.utc_difference_matrix is not None
        assert len(dm.regions) == dm.num_races

    def test_num_races_and_weeks_derived_from_data(self, track_db):
        """Test that dimensions are derived from data, not hardcoded."""
        # Act
        dm = DistanceMatrix(track_db)

        # Assert
        expected_races = len(track_db)
        first_track = next(iter(track_db.values()))
        expected_weeks = len(first_track.temperatures)
        
        assert dm.num_races == expected_races
        assert dm.num_weeks == expected_weeks

    @pytest.mark.parametrize(
        "matrix_attr,expected_shape_func",
        [
            ("distance_matrix", lambda dm: (dm.num_races, dm.num_races)),
            ("temperature_matrix", lambda dm: (dm.num_races, dm.num_weeks)),
            ("utc_offsets", lambda dm: (dm.num_races,)),
            ("utc_difference_matrix", lambda dm: (dm.num_races, dm.num_races)),
        ],
        ids=["distance", "temperature", "utc_offsets", "utc_difference"],
    )
    def test_matrix_shapes(self, track_db, matrix_attr, expected_shape_func):
        """Test all matrices have correct shapes."""
        # Act
        dm = DistanceMatrix(track_db)
        matrix = getattr(dm, matrix_attr)

        # Assert
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == expected_shape_func(dm)

    def test_regions_list_length(self, track_db):
        """Test regions list has correct length."""
        # Act
        dm = DistanceMatrix(track_db)

        # Assert
        assert len(dm.regions) == dm.num_races
        assert isinstance(dm.regions, list)


class TestDistanceMatrixGetters:
    """Tests for DistanceMatrix getter methods."""

    def test_get_distance(self, distance_matrix):
        """Test get_distance returns valid symmetric distance."""
        # Act
        distance = distance_matrix.get_distance(1, 2)

        # Assert
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0
        assert distance == distance_matrix.get_distance(2, 1)  # Symmetric

    def test_get_distance_same_track_is_zero(self, distance_matrix):
        """Test distance from track to itself is zero."""
        # Act
        distance = distance_matrix.get_distance(1, 1)

        # Assert
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_get_temperature(self, distance_matrix):
        """Test get_temperature returns valid temperature."""
        # Act
        temp = distance_matrix.get_temperature(1, 0)

        # Assert
        assert isinstance(temp, (float, np.floating))
        assert -50 < temp < 60  # Reasonable range

    def test_get_utc_properties(self, distance_matrix):
        """Test UTC offset and circular distance methods."""
        # Act
        offset = distance_matrix.get_utc_offset(1)
        utc_diff = distance_matrix.get_circular_utc_distances(1, 2)

        # Assert
        assert isinstance(offset, (int, np.integer))
        assert -12 <= offset <= 14
        assert 0 <= utc_diff <= 12

    def test_get_region(self, distance_matrix):
        """Test get_region returns valid region string."""
        # Act
        region = distance_matrix.get_region(1)

        # Assert
        assert isinstance(region, str)
        assert len(region) > 0


class TestDistanceMatrixProperties:
    """Tests for DistanceMatrix mathematical properties."""

    def test_distance_matrix_is_symmetric(self, distance_matrix):
        """Test distance matrix is symmetric."""
        # Act & Assert
        for i in range(1, min(distance_matrix.num_races, 6)):
            for j in range(i + 1, min(distance_matrix.num_races, 6)):
                assert distance_matrix.get_distance(i, j) == pytest.approx(
                    distance_matrix.get_distance(j, i)
                )

    def test_distance_matrix_is_non_negative(self, distance_matrix):
        """Test all distances are non-negative."""
        # Act & Assert
        for i in range(1, distance_matrix.num_races + 1):
            for j in range(1, distance_matrix.num_races + 1):
                assert distance_matrix.get_distance(i, j) >= -0.01

    def test_utc_difference_matrix_is_symmetric(self, distance_matrix):
        """Test UTC difference matrix is symmetric."""
        # Act & Assert
        for i in range(1, min(distance_matrix.num_races, 6)):
            for j in range(i + 1, min(distance_matrix.num_races, 6)):
                assert distance_matrix.get_circular_utc_distances(i, j) == \
                       distance_matrix.get_circular_utc_distances(j, i)

    def test_utc_differences_are_bounded(self, distance_matrix):
        """Test UTC differences are bounded by 12 hours."""
        # Act & Assert
        for i in range(1, distance_matrix.num_races + 1):
            for j in range(1, distance_matrix.num_races + 1):
                utc_diff = distance_matrix.get_circular_utc_distances(i, j)
                assert 0 <= utc_diff <= 12


class TestDistanceMatrixRegions:
    """Tests for region-based data access."""

    def test_get_races_by_region(self, distance_matrix):
        """Test get_races_by_region returns set of valid race IDs."""
        # Act
        races = distance_matrix.get_races_by_region("Europe")

        # Assert
        assert isinstance(races, set)
        for race_id in races:
            assert 1 <= race_id <= distance_matrix.num_races

    def test_get_middle_east_races(self, distance_matrix):
        """Test Middle East region detection works."""
        # Act
        middle_east_races = distance_matrix.get_races_by_region("Middle East")

        # Assert
        assert isinstance(middle_east_races, set)
        assert len(middle_east_races) > 0  # Should exist in F1 calendar

    def test_nonexistent_region_returns_empty_set(self, distance_matrix):
        """Test getting races by non-existent region returns empty set."""
        # Act
        races = distance_matrix.get_races_by_region("Narnia")

        # Assert
        assert isinstance(races, set)
        assert len(races) == 0

    def test_all_races_assigned_to_regions(self, distance_matrix):
        """Test that all races belong to some region."""
        # Act
        all_regions = set(distance_matrix.regions)
        all_races_in_regions = set()
        for region in all_regions:
            all_races_in_regions.update(distance_matrix.get_races_by_region(region))

        # Assert
        expected_races = set(range(1, distance_matrix.num_races + 1))
        assert all_races_in_regions == expected_races


class TestDistanceMatrixValidation:
    """Tests for DistanceMatrix validation."""

    def test_reject_empty_track_db(self):
        """Test initialization rejects empty track database."""
        # Act & Assert
        with pytest.raises(ValueError, match="track_db cannot be empty"):
            DistanceMatrix({})

    def test_reject_non_consecutive_race_ids(self, track_db):
        """Test initialization rejects non-consecutive race IDs."""
        # Arrange
        invalid_db = {1: track_db[1], 3: track_db[3]}

        # Act & Assert
        with pytest.raises(ValueError, match="consecutive integers"):
            DistanceMatrix(invalid_db)

    def test_reject_inconsistent_temperature_counts(self, track_db):
        """Test initialization rejects inconsistent temperature vectors."""
        # Arrange
        invalid_db = dict(track_db)
        first_track = invalid_db[1]
        bad_track = first_track.model_copy(
            update={"temperatures": first_track.temperatures[:-1]}
        )
        invalid_db[1] = bad_track

        # Act & Assert
        with pytest.raises(ValueError, match="same season length|consistent"):
            DistanceMatrix(invalid_db)
