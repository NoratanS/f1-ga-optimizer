import pytest

from f1_optimizer.core.models import Individual
from f1_optimizer.core.selection import tournament_selection


class TestTournamentSelection:
    """Test cases for tournament_selection function."""

    def test_select_returns_best_from_entire_population(self):
        """When tournament is entire population, should return best individual."""
        pop = [
            Individual([1, 2, 3], fitness=100.0),
            Individual([4, 5, 6], fitness=50.0),
            Individual([7, 8, 9], fitness=75.0),
            Individual([10, 11, 12], fitness=25.0),  # best
        ]
        winner = tournament_selection(pop, tournament_size=len(pop))
        assert winner.fitness == 25.0

    def test_select_returns_tournament_member(self):
        """Selected individual must be from tournament sample."""
        pop = [
            Individual([1], fitness=10.0),
            Individual([2], fitness=20.0),
            Individual([3], fitness=30.0),
        ]
        winner = tournament_selection(pop, tournament_size=2)
        assert winner in pop

    def test_error_tournament_size_exceeds_population(self):
        """Should raise error if tournament_size > population size."""
        pop = [Individual([1], fitness=1.0), Individual([2], fitness=2.0)]
        with pytest.raises(ValueError, match="cannot exceed population size"):
            tournament_selection(pop, tournament_size=3)

    def test_error_unevaluated_individuals(self):
        """Should raise error if any tournament member has fitness=None."""
        pop = [
            Individual([1], fitness=10.0),
            Individual([2], fitness=None),
            Individual([3], fitness=30.0),
        ]
        with pytest.raises(ValueError, match="unevaluated"):
            tournament_selection(pop, tournament_size=len(pop))
