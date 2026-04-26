"""Shared fixtures for CLI tests."""

import pytest

from f1_optimizer.core.evaluation.fitness import FitnessBreakdown
from f1_optimizer.core.ga.ga import GAConfig, GAReport
from f1_optimizer.core.models import Individual


@pytest.fixture
def sorted_chromosome(track_db):
    """Valid chromosome with races 1..N in order, followed by break genes."""
    num_races = len(track_db)
    return list(range(1, num_races + 1)) + list(range(100, 100 + (40 - num_races)))


@pytest.fixture
def minimal_ga_config():
    """Minimal GA config for fast integration tests."""
    return GAConfig(population_size=10, generations=5, stagnation_threshold=3, elite_size=2)


@pytest.fixture
def sample_ga_report(track_db):
    """Manually constructed GAReport for unit testing reports and schedule."""
    num_races = len(track_db)
    chromosome = list(range(1, num_races + 1)) + list(range(100, 100 + (40 - num_races)))
    individual = Individual(chromosome=chromosome, fitness=50000.0)
    breakdown = FitnessBreakdown(
        total_fitness=50000.0,
        distance_penalty=12500.0,   # w_dist=0.25, K_dist=1.0 → 50,000 km
        environment_penalty=12500.0,
        jetlag_penalty=6250.0,
        headers_penalty=6250.0,
        logistics_penalty=12500.0,
        hard_violations_penalty=0.0,
    )
    return GAReport(
        best_fitness=50000.0,
        best_individual=individual,
        best_fitness_per_gen=[60000.0, 55000.0, 50000.0],
        best_breakdown_per_gen=[breakdown, breakdown, breakdown],
        generations_run=2,
        fitness_breakdown=breakdown,
    )
