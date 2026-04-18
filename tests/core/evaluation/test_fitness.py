"""Tests for FitnessConfig and FitnessEvaluator."""

import pytest

from f1_optimizer.core.evaluation.fitness import FitnessConfig, FitnessEvaluator
from f1_optimizer.core.models import Individual


class TestFitnessConfigValidation:
    """Tests for FitnessConfig validation."""

    def test_default_config_is_valid(self):
        """Test default FitnessConfig is valid and positive."""
        # Act
        config = FitnessConfig()

        # Assert
        assert config is not None
        assert config.K_dist > 0
        assert config.K_env > 0
        assert config.K_utc > 0
        assert config.K_double > 0
        assert config.K_triple > 0
        assert config.K_bounce > 0
        assert config.K_flyaway > 0
        assert config.M > 0

    def test_weights_valid_and_sum_to_one(self):
        """Test all weights are in [0, 1] and sum to ~1.0."""
        # Act
        config = FitnessConfig()

        # Assert
        for w in [config.w_dist, config.w_env, config.w_log, config.w_hum]:
            assert 0 <= w <= 1
        total = config.w_dist + config.w_env + config.w_log + config.w_hum
        assert 0.95 <= total <= 1.05

    def test_temperature_bounds_ordered(self):
        """Test temperature bounds are logically ordered."""
        # Act
        config = FitnessConfig()

        # Assert
        assert config.T_opt_min < config.T_opt_max
        assert config.T_hard_min < config.T_hard_max
        assert config.T_hard_min <= config.T_opt_min
        assert config.T_opt_max <= config.T_hard_max

    @pytest.mark.parametrize(
        "invalid_coeff,coeff_name",
        [
            (-1.0, "K_dist"),
            (-1.0, "K_env"),
            (-1.0, "K_utc"),
        ],
        ids=["negative_distance", "negative_environment", "negative_jetlag"],
    )
    def test_reject_negative_penalty_coefficients(self, invalid_coeff, coeff_name):
        """Test rejection of negative penalty coefficients."""
        # Act & Assert
        with pytest.raises(ValueError):
            FitnessConfig(**{coeff_name: invalid_coeff})

    @pytest.mark.parametrize(
        "invalid_weight",
        [1.5, -0.5],
        ids=["above_one", "negative"],
    )
    def test_reject_invalid_weights(self, invalid_weight):
        """Test rejection of weights outside [0, 1]."""
        # Act & Assert
        with pytest.raises(ValueError):
            FitnessConfig(w_dist=invalid_weight)

    @pytest.mark.parametrize(
        "invalid_m",
        [-1000, 0],
        ids=["negative", "zero"],
    )
    def test_reject_invalid_big_m(self, invalid_m):
        """Test rejection of non-positive Big-M constant."""
        # Act & Assert
        with pytest.raises(ValueError):
            FitnessConfig(M=invalid_m)

    def test_custom_coefficients(self):
        """Test creating config with custom coefficients."""
        # Act
        config = FitnessConfig(K_dist=0.5, K_env=0.3, M=10000)

        # Assert
        assert config.K_dist == 0.5
        assert config.K_env == 0.3
        assert config.M == 10000


class TestFitnessEvaluatorInitialization:
    """Tests for FitnessEvaluator initialization."""

    def test_initialization_stores_inputs(self, distance_matrix, fitness_config):
        """Test initialization properly stores distance matrix and config."""
        # Act
        evaluator = FitnessEvaluator(distance_matrix, fitness_config)

        # Assert
        assert evaluator.distance_matrix is distance_matrix
        assert evaluator.config is fitness_config

    def test_auto_detects_middle_east_races(self, distance_matrix, fitness_config):
        """Test auto-detection of Middle East races from data."""
        # Act
        evaluator = FitnessEvaluator(distance_matrix, fitness_config)

        # Assert
        assert isinstance(evaluator.middle_east_races, set)
        for race_id in evaluator.middle_east_races:
            assert 1 <= race_id <= distance_matrix.num_races
        # Should match direct detection from data
        assert evaluator.middle_east_races == distance_matrix.get_races_by_region("Middle East")


class TestFitnessEvaluatorEvaluateIndividual:
    """Tests for FitnessEvaluator.evaluate_individual method."""

    def test_returns_non_negative_float_fitness(self, fitness_evaluator, sample_individual):
        """Test evaluate_individual returns non-negative fitness score."""
        # Act
        fitness = fitness_evaluator.evaluate_individual(sample_individual)

        # Assert
        assert isinstance(fitness, (float, int))
        assert fitness >= 0

    def test_fitness_is_deterministic(self, fitness_evaluator, sample_individual):
        """Test fitness is deterministic (same result on repeated calls)."""
        # Act
        fitness1 = fitness_evaluator.evaluate_individual(sample_individual)
        fitness2 = fitness_evaluator.evaluate_individual(sample_individual)

        # Assert
        assert fitness1 == fitness2

    def test_different_chromosomes_comparable(self, fitness_evaluator, distance_matrix):
        """Test different chromosomes have comparable fitness scores."""
        # Arrange
        indiv1 = Individual(
            chromosome=list(range(1, 25)) + [101] * 16,
            fitness=None
        )
        indiv2 = Individual(
            chromosome=list(range(24, 0, -1)) + [101] * 16,
            fitness=None
        )

        # Act
        fitness1 = fitness_evaluator.evaluate_individual(indiv1)
        fitness2 = fitness_evaluator.evaluate_individual(indiv2)

        # Assert
        assert fitness1 >= 0
        assert fitness2 >= 0
        assert isinstance(fitness1, (float, int))
        assert isinstance(fitness2, (float, int))

    def test_all_breaks_has_violations(self, fitness_evaluator, distance_matrix):
        """Test individual with all breaks has high fitness (max violations)."""
        # Arrange
        all_breaks = Individual(chromosome=[101] * distance_matrix.num_weeks, fitness=None)

        # Act
        fitness = fitness_evaluator.evaluate_individual(all_breaks)

        # Assert
        assert fitness > 0  # Should have completeness violations


class TestFitnessEvaluatorPopulation:
    """Tests for FitnessEvaluator.evaluate_population method."""

    def test_evaluates_all_individuals_in_population(self, fitness_evaluator, distance_matrix):
        """Test evaluate_population modifies all individuals in place."""
        # Arrange
        population = [
            Individual(chromosome=[1, 2, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
            Individual(chromosome=[5, 4, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
            Individual(chromosome=[1, 1, 1] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
        ]

        # Act
        fitness_evaluator.evaluate_population(population)

        # Assert
        for individual in population:
            assert individual.fitness is not None
            assert individual.fitness >= 0
            assert isinstance(individual.fitness, (float, int))

    def test_population_evaluation_is_deterministic(self, fitness_evaluator, distance_matrix):
        """Test population evaluation is deterministic."""
        # Arrange
        pop1 = [
            Individual(chromosome=[1, 2, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
            Individual(chromosome=[5, 4, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
        ]
        pop2 = [
            Individual(chromosome=[1, 2, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
            Individual(chromosome=[5, 4, 3] + [101] * (distance_matrix.num_weeks - 3), fitness=None),
        ]

        # Act
        fitness_evaluator.evaluate_population(pop1)
        fitness_evaluator.evaluate_population(pop2)

        # Assert
        for ind1, ind2 in zip(pop1, pop2):
            assert ind1.fitness == ind2.fitness
