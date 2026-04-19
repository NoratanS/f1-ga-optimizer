from f1_optimizer.core.models import Individual
from f1_optimizer.core.mutation import swap_mutation


class TestSwapMutation:
    """Test cases for swap_mutation function."""

    def test_swap_mutation_returns_new_individual(self):
        """Mutation should return a new Individual instance."""
        ind = Individual([1, 2, 3, 4, 5], fitness=10.0)
        mutated = swap_mutation(ind, num_swaps=1)
        assert isinstance(mutated, Individual)
        assert mutated is not ind

    def test_chromosome_length_preserved(self):
        """Mutated individual should have same chromosome length."""
        ind = Individual(list(range(10)), fitness=5.0)
        mutated = swap_mutation(ind, num_swaps=3)
        assert len(mutated.chromosome) == len(ind.chromosome)

    def test_original_unchanged(self):
        """Original individual should not be modified."""
        original_chrom = [1, 2, 3, 4, 5]
        ind = Individual(original_chrom.copy(), fitness=10.0)
        _ = swap_mutation(ind, num_swaps=2)
        assert ind.chromosome == original_chrom

    def test_mutation_preserves_elements(self):
        """Swaps should not add/remove elements, only rearrange."""
        original_chrom = [10, 20, 30, 40, 50]
        ind = Individual(original_chrom.copy(), fitness=5.0)
        mutated = swap_mutation(ind, num_swaps=3)
        assert sorted(mutated.chromosome) == sorted(original_chrom)
