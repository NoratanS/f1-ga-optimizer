"""Tests for soft constraint penalty calculators (SC1-SC5)."""

import pytest

from f1_optimizer.core.evaluation.soft_constraints import (
    calc_distance_penalty,
    calc_environment_penalty,
    calc_jetlag_penalty,
    calc_headers_penalty,
    calc_logistics_penalty,
    evaluate_soft_constraints,
    SoftConstraintPenalties,
)


class TestDistancePenalty:
    """Tests for SC1: Distance Penalty."""

    def test_zero_penalty_no_races(self, distance_matrix, fitness_config):
        """Test zero penalty when no races scheduled."""
        # Arrange
        chromosome = [101] * distance_matrix.num_weeks

        # Act
        penalty = calc_distance_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_penalty_for_consecutive_races(self, distance_matrix, fitness_config):
        """Test penalty proportional to distance for consecutive races."""
        # Arrange
        chromosome = [1, 2] + [101] * (distance_matrix.num_weeks - 2)

        # Act
        penalty = calc_distance_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        distance = distance_matrix.get_distance(1, 2)
        expected = fitness_config.K_dist * distance
        assert penalty == pytest.approx(expected)

    def test_penalty_ignores_breaks(self, distance_matrix, fitness_config):
        """Test that breaks don't add to distance penalty."""
        # Arrange
        chromosome = [1, 101, 2] + [101] * (distance_matrix.num_weeks - 3)

        # Act
        penalty = calc_distance_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        distance_12 = distance_matrix.get_distance(1, 2)
        expected = fitness_config.K_dist * distance_12
        assert penalty == pytest.approx(expected)

    def test_penalty_is_non_negative(self, distance_matrix, fitness_config, sample_individual):
        """Test penalty is non-negative."""
        # Act
        penalty = calc_distance_penalty(
            sample_individual.chromosome, distance_matrix, fitness_config
        )

        # Assert
        assert penalty >= 0


class TestEnvironmentPenalty:
    """Tests for SC2: Environmental Penalty."""

    def test_zero_penalty_no_races(self, distance_matrix, fitness_config):
        """Test zero penalty when no races scheduled."""
        # Arrange
        chromosome = [101] * distance_matrix.num_weeks

        # Act
        penalty = calc_environment_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_penalty_is_non_negative(self, distance_matrix, fitness_config):
        """Test environment penalty is non-negative for any schedule."""
        # Arrange
        chromosome = list(range(1, min(25, distance_matrix.num_races + 1))) + \
                     [101] * (distance_matrix.num_weeks - 24)

        # Act
        penalty = calc_environment_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty >= 0

    def test_penalty_changes_with_schedule(self, distance_matrix, fitness_config):
        """Test penalty varies with different schedules."""
        # Arrange
        chromosome1 = [1] + [101] * (distance_matrix.num_weeks - 1)
        chromosome2 = [1, 1, 1] + [101] * (distance_matrix.num_weeks - 3)

        # Act
        penalty1 = calc_environment_penalty(chromosome1, distance_matrix, fitness_config)
        penalty2 = calc_environment_penalty(chromosome2, distance_matrix, fitness_config)

        # Assert
        assert penalty1 >= 0
        assert penalty2 >= 0


class TestJetlagPenalty:
    """Tests for SC3: Jet Lag Penalty (UTC jump fatigue)."""

    def test_zero_penalty_no_races(self, distance_matrix, fitness_config):
        """Test zero penalty when no races scheduled."""
        # Arrange
        chromosome = [101] * distance_matrix.num_weeks

        # Act
        penalty = calc_jetlag_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_single_race_zero_penalty(self, distance_matrix, fitness_config):
        """Test zero penalty with single race."""
        # Arrange
        chromosome = [1] + [101] * (distance_matrix.num_weeks - 1)

        # Act
        penalty = calc_jetlag_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_penalty_uses_squared_utc_difference(self, distance_matrix, fitness_config):
        """Test penalty for consecutive races uses squared UTC difference."""
        # Arrange
        chromosome = [1, 2] + [101] * (distance_matrix.num_weeks - 2)

        # Act
        penalty = calc_jetlag_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        utc_diff = distance_matrix.get_circular_utc_distances(1, 2)
        expected = fitness_config.K_utc * (utc_diff ** 2)
        assert penalty == pytest.approx(expected)

    def test_penalty_is_non_negative(self, distance_matrix, fitness_config, sample_individual):
        """Test penalty is non-negative."""
        # Act
        penalty = calc_jetlag_penalty(
            sample_individual.chromosome, distance_matrix, fitness_config
        )

        # Assert
        assert penalty >= 0


class TestHeadersPenalty:
    """Tests for SC4: Consecutive Races Penalty (double/triple headers)."""

    def test_zero_penalty_no_consecutive_races(self, distance_matrix, fitness_config):
        """Test zero penalty when no back-to-back races."""
        # Arrange
        chromosome = [1, 101, 2, 101, 3] + [101] * (distance_matrix.num_weeks - 5)

        # Act
        penalty = calc_headers_penalty(chromosome, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_zero_penalty_single_race(self, distance_matrix, fitness_config):
        """Test zero penalty with single race."""
        # Arrange
        chromosome = [1] + [101] * (distance_matrix.num_weeks - 1)

        # Act
        penalty = calc_headers_penalty(chromosome, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_penalty_counts_headers(self, distance_matrix, fitness_config):
        """Test penalty counts double and triple headers."""
        # Arrange: Double header (2 consecutive)
        chromosome = [1, 2] + [101] * (distance_matrix.num_weeks - 2)

        # Act
        penalty = calc_headers_penalty(chromosome, fitness_config)

        # Assert
        expected = fitness_config.K_double * 1  # One double header
        assert penalty == pytest.approx(expected)

    def test_penalty_is_non_negative(self, distance_matrix, fitness_config, sample_individual):
        """Test penalty is non-negative."""
        # Act
        penalty = calc_headers_penalty(sample_individual.chromosome, fitness_config)

        # Assert
        assert penalty >= 0


class TestLogisticsPenalty:
    """Tests for SC5: Logistical Complexity Penalty (bounce back, fly-away)."""

    def test_zero_penalty_no_races(self, distance_matrix, fitness_config):
        """Test zero penalty when no races scheduled."""
        # Arrange
        chromosome = [101] * distance_matrix.num_weeks

        # Act
        penalty = calc_logistics_penalty(chromosome, distance_matrix, fitness_config)

        # Assert
        assert penalty == pytest.approx(0.0)

    def test_penalty_is_non_negative(self, distance_matrix, fitness_config, sample_individual):
        """Test logistics penalty is non-negative."""
        # Act
        penalty = calc_logistics_penalty(
            sample_individual.chromosome, distance_matrix, fitness_config
        )

        # Assert
        assert penalty >= 0

    def test_penalty_different_schedules(self, distance_matrix, fitness_config):
        """Test penalty varies with different schedules."""
        # Arrange
        chromosome1 = [1, 2, 1] + [101] * (distance_matrix.num_weeks - 3)  # Bounce back
        chromosome2 = [1, 2, 3] + [101] * (distance_matrix.num_weeks - 3)  # Sequential

        # Act
        penalty1 = calc_logistics_penalty(chromosome1, distance_matrix, fitness_config)
        penalty2 = calc_logistics_penalty(chromosome2, distance_matrix, fitness_config)

        # Assert
        assert penalty1 >= 0
        assert penalty2 >= 0


class TestEvaluateSoftConstraints:
    """Tests for evaluate_soft_constraints orchestrator."""

    def test_returns_soft_constraint_penalties_object(
        self, distance_matrix, fitness_config, sample_individual
    ):
        """Test evaluate_soft_constraints returns proper object with all 5 penalties."""
        # Act
        penalties = evaluate_soft_constraints(
            sample_individual.chromosome,
            distance_matrix,
            fitness_config,
        )

        # Assert
        assert isinstance(penalties, SoftConstraintPenalties)
        assert hasattr(penalties, "distance")
        assert hasattr(penalties, "environment")
        assert hasattr(penalties, "jetlag")
        assert hasattr(penalties, "headers")
        assert hasattr(penalties, "logistics")
        # All should be non-negative floats
        for attr in ["distance", "environment", "jetlag", "headers", "logistics"]:
            value = getattr(penalties, attr)
            assert isinstance(value, (float, int))
            assert value >= 0

    def test_penalties_are_non_negative(self, distance_matrix, fitness_config, sample_individual):
        """Test all penalties in result are non-negative."""
        # Act
        penalties = evaluate_soft_constraints(
            sample_individual.chromosome, distance_matrix, fitness_config
        )

        # Assert
        assert penalties.distance >= 0
        assert penalties.environment >= 0
        assert penalties.jetlag >= 0
        assert penalties.headers >= 0
        assert penalties.logistics >= 0
