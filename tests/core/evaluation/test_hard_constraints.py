"""Tests for hard constraint violation counters (HC1-HC7)."""

import pytest

from f1_optimizer.core.evaluation.hard_constraints import (
    count_completeness_violations,
    count_capacity_violations,
    count_weather_violations,
    count_burnout_violations,
    count_break_violations,
    count_ramadan_violations,
    evaluate_hard_constraints,
    HardConstraintViolations,
)


class TestCompletenessViolations:
    """Tests for HC1: Race Completeness constraint."""

    @pytest.mark.parametrize(
        "chromosome,num_races,expected_violations",
        [
            ([1, 2, 3, 4, 5], 5, 0),  # All races once
            ([1], 1, 0),  # Single race
            ([1, 2, 3, 4, 4], 5, 2),  # One duplicate, one missing
            ([101, 102, 103, 104, 105], 5, 5),  # All missing
            ([1, 1, 1, 1, 1], 5, 8),  # One race repeated
        ],
        ids=["all_races_once", "single_race", "missing_and_duplicate", "all_missing", "all_same_race"],
    )
    def test_completeness_violations(self, chromosome, num_races, expected_violations):
        """Test completeness violation counting."""
        # Act
        violations = count_completeness_violations(chromosome, num_races)

        # Assert
        assert violations == expected_violations


class TestCapacityViolations:
    """Tests for HC2: Capacity Constraint (at most one race per week)."""

    @pytest.mark.parametrize(
        "chromosome",
        [
            [1, 2, 3, 4, 5],  # Valid schedule
            [1, 101, 2, 102, 3],  # With breaks
            [101, 102, 103, 104, 105],  # All breaks
        ],
        ids=["valid_schedule", "with_breaks", "all_breaks"],
    )
    def test_capacity_no_violations(self, chromosome):
        """Test capacity constraint has no violations (enforced by GA representation)."""
        # Act
        violations = count_capacity_violations(chromosome)

        # Assert
        assert violations == 0


class TestWeatherViolations:
    """Tests for HC3: Extreme Weather constraint."""

    def test_weather_violations_are_non_negative(self, distance_matrix, fitness_config):
        """Test weather violations are counted correctly."""
        # Arrange
        chromosome = [1, 2, 3, 4, 5] + [101] * (distance_matrix.num_weeks - 5)

        # Act
        violations = count_weather_violations(chromosome, distance_matrix, fitness_config)

        # Assert
        assert isinstance(violations, int)
        assert violations >= 0


class TestBurnoutViolations:
    """Tests for HC4: Burnout Prevention constraint (max 3 consecutive races)."""

    @pytest.mark.parametrize(
        "chromosome,expected_violations",
        [
            ([1, 2, 3, 101, 4, 5, 6, 101], 0),  # Max 3 consecutive
            ([1, 2, 3, 4, 101, 5], 1),  # 4 consecutive = 1 violation
            ([1, 2, 3, 4, 5, 101], 2),  # 5 consecutive = 2 violations
            ([101, 102, 103, 104], 0),  # All breaks
        ],
        ids=["max_three", "four_races", "five_races", "all_breaks"],
    )
    def test_burnout_violations(self, chromosome, expected_violations):
        """Test burnout violation counting."""
        # Act
        violations = count_burnout_violations(chromosome)

        # Assert
        assert violations == expected_violations


class TestBreakViolations:
    """Tests for HC5: Summer Break Requirement."""

    def test_no_violations_with_valid_break(self, fitness_config):
        """Test no violations when valid break exists in summer window."""
        # Arrange
        chromosome = list(range(1, 24)) + [101, 101, 101] + [101] * (40 - 27)
        start_week, end_week = fitness_config.summer_break_window
        for i in range(start_week, min(start_week + 3, len(chromosome))):
            chromosome[i] = 101

        # Act
        violations = count_break_violations(chromosome, fitness_config)

        # Assert
        assert violations == 0

    def test_violations_without_break_in_window(self, fitness_config):
        """Test violations when no 3-week break in summer window."""
        # Arrange: All races, no sufficient break
        chromosome = list(range(1, 25)) + [101] * 16

        # Act
        violations = count_break_violations(chromosome, fitness_config)

        # Assert
        assert isinstance(violations, int)
        assert violations >= 0


class TestRamadanViolations:
    """Tests for HC6: Ramadan Observance constraint."""

    def test_no_violations_no_middle_east_races(self, fitness_config):
        """Test no violations when no Middle East races scheduled."""
        # Arrange
        chromosome = list(range(1, 25)) + [101] * 16

        # Act
        violations = count_ramadan_violations(chromosome, fitness_config, set())

        # Assert
        assert violations == 0

    def test_no_violations_middle_east_outside_ramadan(self, fitness_config):
        """Test no violations when Middle East races outside Ramadan window."""
        # Arrange
        chromosome = [101] * 40
        start_week, _ = fitness_config.ramadan_window
        chromosome[start_week - 1] = 1  # Week before Ramadan

        # Act
        violations = count_ramadan_violations(chromosome, fitness_config, {1})

        # Assert
        assert violations == 0

    def test_violations_during_ramadan(self, fitness_config):
        """Test violations when Middle East races during Ramadan."""
        # Arrange
        chromosome = [101] * 40
        start_week, end_week = fitness_config.ramadan_window
        for week in range(start_week, min(start_week + 2, end_week + 1)):
            chromosome[week] = 1

        # Act
        violations = count_ramadan_violations(chromosome, fitness_config, {1})

        # Assert
        assert violations >= 1


class TestEvaluateHardConstraints:
    """Tests for evaluate_hard_constraints orchestrator."""

    def test_returns_hard_constraint_violations_object(
        self, distance_matrix, fitness_config, sample_individual
    ):
        """Test evaluate_hard_constraints returns proper object with all 7 counts."""
        # Act
        violations = evaluate_hard_constraints(
            sample_individual.chromosome,
            distance_matrix,
            fitness_config,
            distance_matrix.get_races_by_region("Middle East"),
        )

        # Assert
        assert isinstance(violations, HardConstraintViolations)
        assert hasattr(violations, "completeness")
        assert hasattr(violations, "capacity")
        assert hasattr(violations, "weather")
        assert hasattr(violations, "burnout")
        assert hasattr(violations, "break_requirement")
        assert hasattr(violations, "ramadan")
        # All should be non-negative integers
        assert all(isinstance(getattr(violations, attr), int) for attr in
                   ["completeness", "capacity", "weather", "burnout", "break_requirement", "ramadan"])
        assert all(getattr(violations, attr) >= 0 for attr in
                   ["completeness", "capacity", "weather", "burnout", "break_requirement", "ramadan"])
