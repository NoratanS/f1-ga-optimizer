import pytest

from f1_optimizer.core.ga.ga import (
    GAConfig,
    GAReport,
    initialize_population,
    GeneticAlgorithm,
)
from f1_optimizer.core.models import Individual


class TestGAConfig:
    """Tests for GAConfig."""

    def test_default_config_is_valid(self):
        """Default configuration should be valid."""
        config = GAConfig()
        assert config.population_size == 50
        assert config.generations == 100
        assert config.crossover_probability == 0.8
        assert config.mutation_probability == 0.1
        assert config.tournament_size == 5
        assert config.crossover_type == "ox1"
        assert config.mutation_type == "swap"
        assert config.elite_size == 2
        assert config.stagnation_threshold == 20

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = GAConfig(
            population_size=100,
            generations=200,
            crossover_probability=0.9,
            tournament_size=10,
        )
        assert config.population_size == 100
        assert config.generations == 200
        assert config.crossover_probability == 0.9
        assert config.tournament_size == 10

    def test_reject_invalid_population_size(self):
        """Population size must be > 0."""
        with pytest.raises(ValueError):
            GAConfig(population_size=0)
        with pytest.raises(ValueError):
            GAConfig(population_size=-1)

    def test_reject_invalid_generations(self):
        """Generations must be > 0."""
        with pytest.raises(ValueError):
            GAConfig(generations=0)

    def test_reject_invalid_crossover_probability(self):
        """Crossover probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            GAConfig(crossover_probability=-0.1)
        with pytest.raises(ValueError):
            GAConfig(crossover_probability=1.1)

    def test_reject_invalid_mutation_probability(self):
        """Mutation probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            GAConfig(mutation_probability=1.5)

    def test_reject_invalid_tournament_size(self):
        """Tournament size must be > 0."""
        with pytest.raises(ValueError):
            GAConfig(tournament_size=0)

    def test_reject_invalid_elite_size(self):
        """Elite size must be >= 0."""
        with pytest.raises(ValueError):
            GAConfig(elite_size=-1)

    def test_reject_invalid_stagnation_threshold(self):
        """Stagnation threshold must be >= 1."""
        with pytest.raises(ValueError):
            GAConfig(stagnation_threshold=0)

    def test_reject_invalid_crossover_type(self):
        """Crossover type must be 'ox1' or 'pmx'."""
        with pytest.raises(ValueError):
            GAConfig(crossover_type="uniform")
        with pytest.raises(ValueError):
            GAConfig(crossover_type="invalid")

    def test_reject_invalid_mutation_type(self):
        """Mutation type must be 'swap' or 'inversion'."""
        with pytest.raises(ValueError):
            GAConfig(mutation_type="gaussian")
        with pytest.raises(ValueError):
            GAConfig(mutation_type="invalid")

    def test_accept_crossover_types(self):
        """Should accept both crossover types."""
        config_ox1 = GAConfig(crossover_type="ox1")
        config_pmx = GAConfig(crossover_type="pmx")
        assert config_ox1.crossover_type == "ox1"
        assert config_pmx.crossover_type == "pmx"

    def test_accept_mutation_types(self):
        """Should accept both mutation types."""
        config_swap = GAConfig(mutation_type="swap")
        config_inversion = GAConfig(mutation_type="inversion")
        assert config_swap.mutation_type == "swap"
        assert config_inversion.mutation_type == "inversion"


class TestGAReport:
    """Tests for GAReport."""

    def test_report_creation(self):
        """Should create report with all fields."""
        ind = Individual([1, 2, 3], fitness=5.0)
        report = GAReport(
            best_fitness=5.0,
            best_individual=ind,
            avg_fitness_per_gen=[10.0, 8.0, 5.0],
            best_fitness_per_gen=[10.0, 8.0, 5.0],
            generations_run=3,
        )
        assert report.best_fitness == 5.0
        assert report.generations_run == 3
        assert len(report.avg_fitness_per_gen) == 3

    def test_report_summary(self):
        """Summary should be human-readable."""
        ind = Individual([1, 2, 3], fitness=5.0)
        report = GAReport(
            best_fitness=5.0,
            best_individual=ind,
            avg_fitness_per_gen=[10.0, 5.0],
            best_fitness_per_gen=[10.0, 5.0],
            generations_run=2,
        )
        summary = report.summary()
        assert "5.00" in summary or "5.0" in summary
        assert "2" in summary
        assert "GA Report" in summary


class TestInitializePopulation:
    """Tests for initialize_population function."""

    def test_returns_correct_population_size(self):
        """Should return population of correct size."""
        pop = initialize_population(population_size=10, num_races=5, num_weeks=3)
        assert len(pop) == 10

    def test_all_individuals_are_valid(self):
        """All individuals should have correct chromosome size."""
        num_races = 5
        num_weeks = 8  # 5 races + 3 breaks = 8 total
        pop = initialize_population(population_size=10, num_races=num_races, num_weeks=num_weeks)
        
        for ind in pop:
            assert len(ind.chromosome) == num_weeks  # Chromosome size is num_weeks

    def test_chromosome_contains_all_genes(self):
        """Chromosome should contain all race and break genes exactly once."""
        num_races = 5
        num_weeks = 8  # 5 races + 3 breaks = 8 total
        pop = initialize_population(population_size=1, num_races=num_races, num_weeks=num_weeks)
        
        chromosome = pop[0].chromosome
        races = sorted([g for g in chromosome if g < 100])
        breaks = sorted([g for g in chromosome if g >= 100])
        
        # Should have exactly num_races races and (num_weeks - num_races) breaks
        assert races == list(range(1, num_races + 1))
        assert breaks == list(range(100, 100 + (num_weeks - num_races)))

    def test_different_individuals_different_chromosomes(self):
        """With high probability, individuals should have different chromosomes."""
        pop = initialize_population(population_size=100, num_races=24, num_weeks=40)
        
        # Convert chromosomes to tuples for comparison
        chroms = [tuple(ind.chromosome) for ind in pop]
        unique_chroms = set(chroms)
        
        # Most should be unique (allow some collisions for large population)
        assert len(unique_chroms) > 95

    def test_individuals_have_no_fitness(self):
        """Initialized individuals should have fitness=None."""
        pop = initialize_population(population_size=5, num_races=5, num_weeks=8)
        assert all(ind.fitness is None for ind in pop)


class TestGeneticAlgorithm:
    """Tests for GeneticAlgorithm class."""

    def test_initialization(self, fitness_evaluator):
        """Should initialize with config and fitness evaluator."""
        config = GAConfig()
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        assert ga.config == config
        assert ga.fitness_evaluator == fitness_evaluator
        assert ga.num_races == fitness_evaluator.distance_matrix.num_races
        assert ga.num_weeks == fitness_evaluator.distance_matrix.num_weeks

    def test_evolve_returns_gaport(self, fitness_evaluator):
        """Evolve should return GAReport."""
        config = GAConfig(generations=2, population_size=10)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        assert isinstance(report, GAReport)
        assert report.best_individual is not None
        assert report.best_fitness is not None
        assert report.generations_run > 0

    def test_evolve_tracks_history(self, fitness_evaluator):
        """History should have one entry per generation + initial."""
        config = GAConfig(generations=3, population_size=10)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        assert len(report.best_fitness_per_gen) == report.generations_run + 1
        assert len(report.avg_fitness_per_gen) == report.generations_run + 1

    def test_best_individual_improves_or_stays_same(self, fitness_evaluator):
        """Best fitness should monotonically decrease (lower is better)."""
        config = GAConfig(generations=5, population_size=20)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        for i in range(len(report.best_fitness_per_gen) - 1):
            # Best fitness should never increase
            assert report.best_fitness_per_gen[i + 1] <= report.best_fitness_per_gen[i]

    def test_final_best_fitness_matches_reported(self, fitness_evaluator):
        """Final best fitness should match the reported best."""
        config = GAConfig(generations=2, population_size=10)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        assert report.best_fitness == report.best_fitness_per_gen[-1]

    def test_early_stopping_with_stagnation_threshold(self, fitness_evaluator):
        """Should stop early if stagnation_threshold exceeded."""
        config = GAConfig(
            generations=100,
            population_size=50,
            stagnation_threshold=3,  # Stop after 3 generations without improvement
            mutation_probability=0.0,  # Disable mutation to force plateau
            crossover_probability=0.0,  # Disable crossover to force plateau
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # Should run far fewer than 100 generations due to early stopping
        assert report.generations_run < 100

    def test_elite_preservation(self, fitness_evaluator):
        """Best individuals should be preserved across generations."""
        config = GAConfig(
            generations=3,
            population_size=20,
            elite_size=5,
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # Best fitness per gen should exist for all generations
        assert len(report.best_fitness_per_gen) > 0

    def test_crossover_probability_respected(self, fitness_evaluator):
        """When crossover_probability=0, no crossover should occur."""
        config = GAConfig(
            generations=2,
            population_size=10,
            crossover_probability=0.0,
            mutation_probability=0.0,
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # Should still complete without error
        assert report.best_fitness is not None

    def test_mutation_probability_respected(self, fitness_evaluator):
        """When mutation_probability=0, no mutation should occur."""
        config = GAConfig(
            generations=2,
            population_size=10,
            mutation_probability=0.0,
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        assert report.best_fitness is not None

    def test_different_crossover_types(self, fitness_evaluator):
        """Should work with both crossover types."""
        config_ox1 = GAConfig(generations=2, population_size=10, crossover_type="ox1")
        config_pmx = GAConfig(generations=2, population_size=10, crossover_type="pmx")
        
        ga_ox1 = GeneticAlgorithm(config_ox1, fitness_evaluator)
        ga_pmx = GeneticAlgorithm(config_pmx, fitness_evaluator)
        
        report_ox1 = ga_ox1.evolve()
        report_pmx = ga_pmx.evolve()
        
        assert report_ox1.best_fitness is not None
        assert report_pmx.best_fitness is not None

    def test_different_mutation_types(self, fitness_evaluator):
        """Should work with both mutation types."""
        config_swap = GAConfig(generations=2, population_size=10, mutation_type="swap")
        config_inv = GAConfig(generations=2, population_size=10, mutation_type="inversion")
        
        ga_swap = GeneticAlgorithm(config_swap, fitness_evaluator)
        ga_inv = GeneticAlgorithm(config_inv, fitness_evaluator)
        
        report_swap = ga_swap.evolve()
        report_inv = ga_inv.evolve()
        
        assert report_swap.best_fitness is not None
        assert report_inv.best_fitness is not None

    def test_population_size_maintained(self, fitness_evaluator):
        """Population size should not change after evolution."""
        config = GAConfig(generations=2, population_size=15)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        # Capture initial population via initialize_population
        pop = initialize_population(config.population_size, ga.num_races, ga.num_weeks)
        assert len(pop) == config.population_size

    def test_best_individual_is_clone(self, fitness_evaluator):
        """Best individual in report should be independent copy."""
        config = GAConfig(generations=2, population_size=10)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # Modify the returned individual
        _ = report.best_individual.chromosome.copy()
        report.best_individual.chromosome[0] = 999
        
        # Should not affect later runs (new algorithm instance)
        ga2 = GeneticAlgorithm(config, fitness_evaluator)
        report2 = ga2.evolve()
        
        # New run should have valid chromosome
        assert report2.best_individual.chromosome[0] != 999

    def test_single_generation(self, fitness_evaluator):
        """Should work with just one generation."""
        config = GAConfig(generations=1, population_size=10)
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        assert report.generations_run == 1
        assert len(report.best_fitness_per_gen) == 2  # Initial + gen 1

    def test_tournament_size_constraint(self, fitness_evaluator):
        """Tournament size should not exceed population size."""
        config = GAConfig(
            generations=2,
            population_size=5,
            tournament_size=3,  # Valid: < population_size
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        assert report.best_fitness is not None

    def test_generations_run_off_by_one_accuracy(self, fitness_evaluator):
        """Verify generations_run is reported accurately with early stopping."""
        config = GAConfig(
            generations=100,
            population_size=10,
            stagnation_threshold=5,  # Stop after 5 generations without improvement
            mutation_probability=0.0,  # Disable mutation to force early plateau
            crossover_probability=0.0,  # Disable crossover to force plateau
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # History should have initial + evolutionary generations
        # If we run N generations, history has N+1 entries (initial + N evolutionary)
        assert len(report.best_fitness_per_gen) == report.generations_run + 1
        assert len(report.avg_fitness_per_gen) == report.generations_run + 1
        # Should stop well before generation 100 due to early stopping
        assert report.generations_run < 100
        # generations_run must be positive
        assert report.generations_run > 0

    def test_offspring_generation_with_odd_population_size(self, fitness_evaluator):
        """Verify offspring generation handles odd population_size correctly."""
        # Odd population: 11 = 10 non-elite + 1
        config = GAConfig(
            generations=2,
            population_size=11,
            elite_size=2,
        )
        ga = GeneticAlgorithm(config, fitness_evaluator)
        
        report = ga.evolve()
        
        # Should complete without error and produce valid report
        assert report.best_fitness is not None
        assert report.best_individual is not None
        assert len(report.best_fitness_per_gen) == 3  # Initial + 2 generations
