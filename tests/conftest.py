"""Pytest configuration and shared fixtures for all tests."""

import pytest
from pathlib import Path

from f1_optimizer.data_access.loader import CsvTrackLoader
from f1_optimizer.core.evaluation.distance_matrix import DistanceMatrix
from f1_optimizer.core.evaluation.fitness import FitnessConfig, FitnessEvaluator
from f1_optimizer.core.models import Individual


@pytest.fixture(scope="session")
def data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def track_db(data_dir):
    """Load track database from CSV (session-scoped for efficiency)."""
    loader = CsvTrackLoader(data_dir / "data.csv")
    return loader.load()


@pytest.fixture(scope="session")
def distance_matrix(track_db):
    """Create pre-computed distance matrix (session-scoped)."""
    return DistanceMatrix(track_db)


@pytest.fixture
def fitness_config():
    """Create default FitnessConfig for testing."""
    return FitnessConfig()


@pytest.fixture
def fitness_evaluator(distance_matrix, fitness_config):
    """Create FitnessEvaluator instance."""
    return FitnessEvaluator(distance_matrix, fitness_config)


@pytest.fixture
def sample_individual():
    """Create a sample valid individual for testing."""
    # All races scheduled exactly once (races 1-24) with breaks (101+)
    chromosome = list(range(1, 25)) + [101] * 16  # 24 races + 16 breaks = 40 weeks
    return Individual(chromosome=chromosome, fitness=None)


@pytest.fixture
def invalid_individual_no_races():
    """Create individual with no races scheduled (all breaks)."""
    return Individual(chromosome=[101] * 40, fitness=None)


@pytest.fixture
def invalid_individual_missing_race():
    """Create individual missing race 1 (scheduled race 2-24 but missing 1)."""
    chromosome = list(range(2, 25)) + [1] + [101] * 16
    return Individual(chromosome=chromosome, fitness=None)


@pytest.fixture
def invalid_individual_duplicate_race():
    """Create individual with race 1 scheduled twice."""
    chromosome = [1, 1] + list(range(2, 24)) + [101] * 14
    return Individual(chromosome=chromosome, fitness=None)


@pytest.fixture
def individual_extreme_temps(distance_matrix):
    """Create individual scheduled during extreme temperatures if possible.
    
    This schedules races at weeks with highest/lowest temperatures from track_db.
    """
    # Find race-week pairs with extreme temperatures
    extreme_weeks = []
    
    for race_id in range(1, distance_matrix.num_races + 1):
        for week in range(distance_matrix.num_weeks):
            temp = distance_matrix.get_temperature(race_id, week)
            extreme_weeks.append((race_id, week, temp))
    
    # Sort by temperature (to find extremes)
    extreme_weeks.sort(key=lambda x: x[2])
    
    # Create chromosome by scheduling races at extreme temperature weeks
    chromosome = [101] * distance_matrix.num_weeks
    
    # Schedule first N races at weeks with extreme temps
    n_races = min(distance_matrix.num_races, 24)
    for i, (race_id, week, temp) in enumerate(extreme_weeks[:n_races]):
        if chromosome[week] == 101:  # Only if week is empty
            chromosome[week] = race_id
        else:
            # Find next empty week
            for w in range(distance_matrix.num_weeks):
                if chromosome[w] == 101:
                    chromosome[w] = race_id
                    break
    
    # Fill remaining races
    scheduled = {chr for chr in chromosome if chr < 100}
    for race_id in range(1, distance_matrix.num_races + 1):
        if race_id not in scheduled:
            for week in range(distance_matrix.num_weeks):
                if chromosome[week] == 101:
                    chromosome[week] = race_id
                    scheduled.add(race_id)
                    break
    
    return Individual(chromosome=chromosome[:distance_matrix.num_weeks], fitness=None)


@pytest.fixture
def individual_back_to_back():
    """Create individual with all races scheduled back-to-back (consecutive weeks)."""
    # Schedule races 1-24 in consecutive weeks, then breaks
    chromosome = list(range(1, 25)) + [101] * 16
    return Individual(chromosome=chromosome, fitness=None)


@pytest.fixture
def individual_consecutive_races():
    """Create individual with 4 consecutive race weeks (violates burnout HC5)."""
    # Schedule 4 races in weeks 0-3, then pattern that avoids more consecutive races
    chromosome = [1, 2, 3, 4, 101] + list(range(5, 24)) + [101] * (40 - 24)
    return Individual(chromosome=chromosome, fitness=None)
