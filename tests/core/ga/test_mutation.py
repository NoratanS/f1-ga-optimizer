from unittest.mock import patch


from f1_optimizer.core.models import Individual
from f1_optimizer.core.ga.mutation import swap_mutation, inversion_mutation


class TestSwapMutation:
    """Test cases for swap_mutation function."""

    @patch("f1_optimizer.core.ga.mutation.random.sample")
    def test_swap_mutation_mocked_swap(self, mock_sample):
        """Test swap_mutation with mocked random.sample for deterministic output."""
        # Mock one call to random.sample: swap positions (0, 2)
        mock_sample.return_value = (0, 2)
        
        ind = Individual([1, 2, 3, 4, 5])
        mutated = swap_mutation(ind)
        
        # After swap (0,2): [3, 2, 1, 4, 5]
        assert mutated.chromosome == [3, 2, 1, 4, 5]

    def test_swap_mutation_returns_new_individual(self):
        """Mutation should return a new Individual instance."""
        ind = Individual([1, 2, 3, 4, 5], fitness=10.0)
        mutated = swap_mutation(ind)
        assert isinstance(mutated, Individual)
        assert mutated is not ind

    def test_swap_mutation_fitness_reset(self):
        """Mutation should reset fitness to None."""
        ind = Individual([1, 2, 3, 4, 5], fitness=10.0)
        mutated = swap_mutation(ind)
        assert mutated.fitness is None

    def test_chromosome_length_preserved(self):
        """Mutated individual should have same chromosome length."""
        ind = Individual(list(range(10)), fitness=5.0)
        mutated = swap_mutation(ind)
        assert len(mutated.chromosome) == len(ind.chromosome)

    def test_original_unchanged(self):
        """Original individual should not be modified."""
        original_chrom = [1, 2, 3, 4, 5]
        ind = Individual(original_chrom.copy(), fitness=10.0)
        _ = swap_mutation(ind)
        assert ind.chromosome == original_chrom

    def test_mutation_preserves_elements(self):
        """Swap should not add/remove elements, only rearrange."""
        original_chrom = [10, 20, 30, 40, 50]
        ind = Individual(original_chrom.copy(), fitness=5.0)
        mutated = swap_mutation(ind)
        assert sorted(mutated.chromosome) == sorted(original_chrom)


class TestInversionMutation:
    """Test cases for inversion_mutation function."""

    @patch("f1_optimizer.core.ga.mutation.random.randint")
    def test_inversion_mutation_mocked_positions(self, mock_randint):
        """Test inversion_mutation with mocked positions for deterministic output."""
        # Mock randint calls: i=1, j=4 (for segment reversal)
        mock_randint.side_effect = [1, 4]
        
        ind = Individual([1, 2, 3, 4, 5])
        mutated = inversion_mutation(ind)
        
        # Reverse segment [1:5]: [1, 5, 4, 3, 2]
        assert mutated.chromosome == [1, 5, 4, 3, 2]

    def test_inversion_mutation_returns_new_individual(self):
        """Inversion should return a new Individual instance."""
        ind = Individual([1, 2, 3, 4, 5], fitness=10.0)
        mutated = inversion_mutation(ind)
        assert isinstance(mutated, Individual)
        assert mutated is not ind

    def test_inversion_mutation_fitness_reset(self):
        """Inversion should reset fitness to None."""
        ind = Individual([1, 2, 3, 4, 5], fitness=10.0)
        mutated = inversion_mutation(ind)
        assert mutated.fitness is None

    def test_chromosome_length_preserved(self):
        """Inverted individual should have same chromosome length."""
        ind = Individual(list(range(10)), fitness=5.0)
        mutated = inversion_mutation(ind)
        assert len(mutated.chromosome) == len(ind.chromosome)

    def test_original_unchanged(self):
        """Original individual should not be modified."""
        original_chrom = [1, 2, 3, 4, 5]
        ind = Individual(original_chrom.copy(), fitness=10.0)
        _ = inversion_mutation(ind)
        assert ind.chromosome == original_chrom

    def test_inversion_preserves_elements(self):
        """Inversions should not add/remove elements, only rearrange."""
        original_chrom = [10, 20, 30, 40, 50]
        ind = Individual(original_chrom.copy(), fitness=5.0)
        mutated = inversion_mutation(ind)
        assert sorted(mutated.chromosome) == sorted(original_chrom)
