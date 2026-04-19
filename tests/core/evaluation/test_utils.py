import math

from f1_optimizer.core.evaluation.utils import (
    haversine_distance,
    circular_utc_distance,
    extract_race_sequence,
)


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_coordinates(self):
        """Distance between same coordinates should be ~0."""
        distance = haversine_distance(0, 0, 0, 0)
        assert distance < 0.01  # Allow tiny floating point error

    def test_known_distance_london_paris(self):
        """Test with known distance between London and Paris."""
        # London: 51.5074°N, 0.1278°W
        # Paris: 48.8566°N, 2.3522°E
        # Approximate distance: 343 km
        distance = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        assert 340 < distance < 350

    def test_equator_spacing(self):
        """Test distance along the equator."""
        # 1 degree on equator ≈ 111 km
        distance = haversine_distance(0, 0, 0, 1)
        assert 110 < distance < 112

    def test_antipodal_points(self):
        """Distance between antipodal points should be ~half Earth's circumference."""
        # North pole to South pole
        distance = haversine_distance(90, 0, -90, 0)
        expected = math.pi * 6371  # π * R
        assert expected - 100 < distance < expected + 100

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        distance = haversine_distance(-10, -20, -11, -21)
        assert distance > 0  # Should be positive


class TestCircularUtcDistance:
    """Tests for circular_utc_distance function."""

    def test_same_utc(self):
        """Distance between same UTC should be 0."""
        assert circular_utc_distance(5, 5) == 0

    def test_adjacent_utc(self):
        """Distance between adjacent UTC offsets."""
        assert circular_utc_distance(5, 6) == 1

    def test_opposite_sides(self):
        """Distance between opposite sides of the circle."""
        assert circular_utc_distance(0, 12) == 12

    def test_circular_wraparound_forward(self):
        """Test wraparound: 23 to 1 should be 2 hours."""
        assert circular_utc_distance(23, 1) == 2

    def test_negative_utc(self):
        """Test with negative UTC offsets."""
        assert circular_utc_distance(-5, -4) == 1

    def test_negative_to_positive(self):
        """Test transition from negative to positive UTC."""
        # -1 to 1 is 2 hours forward or 22 backward → min is 2
        assert circular_utc_distance(-1, 1) == 2

    def test_out_of_range_positive(self):
        """Test UTC values outside 0-23 range (positive)."""
        # 25 should normalize to 1
        assert circular_utc_distance(25, 2) == 1

    def test_out_of_range_negative(self):
        """Test UTC values outside 0-23 range (negative)."""
        # -1 should normalize to 23
        assert circular_utc_distance(-1, 0) == 1


class TestExtractRaceSequence:
    """Tests for extract_race_sequence function."""

    def test_empty_chromosome(self):
        """Empty chromosome should return empty sequences."""
        omega, tau = extract_race_sequence([])
        assert omega == []
        assert tau == []

    def test_all_valid_races(self):
        """Chromosome with only valid race IDs."""
        chromosome = [1, 2, 3]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [1, 2, 3]
        assert tau == [0, 1, 2]

    def test_all_breaks(self):
        """Chromosome with only breaks (≥100)."""
        chromosome = [100, 101, 102]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == []
        assert tau == []

    def test_mixed_races_and_breaks(self):
        """Chromosome with mixed valid races and breaks."""
        chromosome = [1, 100, 2, 101, 3]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [1, 2, 3]
        assert tau == [0, 2, 4]

    def test_single_valid_race(self):
        """Chromosome with single valid race ID."""
        chromosome = [5]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [5]
        assert tau == [0]

    def test_race_id_zero(self):
        """Race ID 0 should be valid."""
        chromosome = [0, 1, 2]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [0, 1, 2]
        assert tau == [0, 1, 2]

    def test_race_id_99(self):
        """Race ID 99 should be valid (boundary)."""
        chromosome = [99]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [99]
        assert tau == [0]

    def test_break_at_100(self):
        """Race ID 100 should be a break (boundary)."""
        chromosome = [1, 100, 2]
        omega, tau = extract_race_sequence(chromosome)
        assert omega == [1, 2]
        assert tau == [0, 2]


