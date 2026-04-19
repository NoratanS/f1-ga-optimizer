from unittest.mock import patch


from f1_optimizer.core.ga.crossover import crossover_ox1, crossover_pmx
from f1_optimizer.core.models import Individual


class TestCrossoverOX1:
    """Test cases for Order Crossover 1 (OX1)."""

    @patch("f1_optimizer.core.ga.crossover.random.randint")
    def test_ox1_mocked_crossover_point(self, mock_randint):
        """Test OX1 with mocked cut points for deterministic output."""
        # Mock randint calls: point1=1, point2=3
        mock_randint.side_effect = [1, 3]
        
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([3, 4, 5, 1, 2])
        
        child1, child2 = crossover_ox1(parent1, parent2)
        
        # Child1: segment [1:3] from parent1 = [2,3]
        # Positions [1:3] = [2,3]
        # Fill rest from parent2 starting at pos 3: [1,2] -> skip 2,3 -> [1]
        # Fill: pos 3=1, pos 4=4, pos 0=5
        # Result: [5, 2, 3, 1, 4]
        assert child1.chromosome == [5, 2, 3, 1, 4]
        
        # Child2: segment [1:3] from parent2 = [4,5]
        # Positions [1:3] = [4,5]
        # Fill rest from parent1 starting at pos 3: [4,5] -> skip 4,5 -> [1,2,3]
        # Fill: pos 3=1, pos 4=2, pos 0=3
        # Result: [3, 4, 5, 1, 2]
        assert child2.chromosome == [3, 4, 5, 1, 2]

    def test_ox1_returns_individuals(self):
        """Crossover should return two new Individual instances."""
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([5, 4, 3, 2, 1])
        
        child1, child2 = crossover_ox1(parent1, parent2)
        
        assert isinstance(child1, Individual)
        assert isinstance(child2, Individual)
        assert child1 is not parent1
        assert child2 is not parent2

    def test_ox1_chromosome_length_preserved(self):
        """Children should have same chromosome length as parents."""
        parent1 = Individual(list(range(10)))
        parent2 = Individual(list(reversed(range(10))))
        
        child1, child2 = crossover_ox1(parent1, parent2)
        
        assert len(child1.chromosome) == len(parent1.chromosome)
        assert len(child2.chromosome) == len(parent2.chromosome)

    def test_ox1_preserves_all_elements(self):
        """Children should contain all same elements as parents (permutation)."""
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([5, 4, 3, 2, 1])
        
        child1, child2 = crossover_ox1(parent1, parent2)
        
        assert sorted(child1.chromosome) == sorted(parent1.chromosome)
        assert sorted(child2.chromosome) == sorted(parent2.chromosome)

    def test_ox1_fitness_reset(self):
        """Children should have fitness=None (unevaluated)."""
        parent1 = Individual([1, 2, 3, 4, 5], fitness=10.0)
        parent2 = Individual([5, 4, 3, 2, 1], fitness=20.0)
        
        child1, child2 = crossover_ox1(parent1, parent2)
        
        assert child1.fitness is None
        assert child2.fitness is None

    def test_ox1_parents_unchanged(self):
        """Parents should not be modified by crossover."""
        orig_parent1 = [1, 2, 3, 4, 5]
        orig_parent2 = [5, 4, 3, 2, 1]
        
        parent1 = Individual(orig_parent1.copy())
        parent2 = Individual(orig_parent2.copy())
        
        crossover_ox1(parent1, parent2)
        
        assert parent1.chromosome == orig_parent1
        assert parent2.chromosome == orig_parent2


class TestCrossoverPMX:
    """Test cases for Partially Mapped Crossover (PMX)."""

    @patch("f1_optimizer.core.ga.crossover.random.randint")
    def test_pmx_mocked_segment_points(self, mock_randint):
        """Test PMX with mocked segment points for deterministic output."""
        # Mock calls: first for point1, second for point2
        mock_randint.side_effect = [1, 3]
        
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([3, 4, 5, 1, 2])
        
        child1, child2 = crossover_pmx(parent1, parent2)
        
        # Segment [1:3]: parent1=[2,3], parent2=[4,5]
        # Mappings: 2->4, 3->5 (and reverse 4->2, 5->3)
        # Child1: gets segment [4,5] from parent2 at [1:3]
        #   - pos 0: p1[0]=1 not in mapping -> use 1
        #   - pos 3: p1[3]=4 maps to 2 -> use 2
        #   - pos 4: p1[4]=5 maps to 3 -> use 3
        # Expected: [1, 4, 5, 2, 3]
        assert child1.chromosome == [1, 4, 5, 2, 3]
        
        # Child2: gets segment [2,3] from parent1 at [1:3]
        #   - pos 0: p2[0]=3 maps to 5 -> use 5
        #   - pos 3: p2[3]=1 not in mapping -> use 1
        #   - pos 4: p2[4]=2 maps to 4 -> use 4
        # Expected: [5, 2, 3, 1, 4]
        assert child2.chromosome == [5, 2, 3, 1, 4]

    def test_pmx_returns_individuals(self):
        """PMX should return two new Individual instances."""
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([5, 4, 3, 2, 1])
        
        child1, child2 = crossover_pmx(parent1, parent2)
        
        assert isinstance(child1, Individual)
        assert isinstance(child2, Individual)
        assert child1 is not parent1
        assert child2 is not parent2

    def test_pmx_chromosome_length_preserved(self):
        """Children should have same chromosome length as parents."""
        parent1 = Individual(list(range(10)))
        parent2 = Individual(list(reversed(range(10))))
        
        child1, child2 = crossover_pmx(parent1, parent2)
        
        assert len(child1.chromosome) == len(parent1.chromosome)
        assert len(child2.chromosome) == len(parent2.chromosome)

    def test_pmx_preserves_all_elements(self):
        """Children should contain all same elements as parents (permutation)."""
        parent1 = Individual([1, 2, 3, 4, 5])
        parent2 = Individual([5, 4, 3, 2, 1])
        
        child1, child2 = crossover_pmx(parent1, parent2)
        
        assert sorted(child1.chromosome) == sorted(parent1.chromosome)
        assert sorted(child2.chromosome) == sorted(parent2.chromosome)

    def test_pmx_fitness_reset(self):
        """Children should have fitness=None (unevaluated)."""
        parent1 = Individual([1, 2, 3, 4, 5], fitness=10.0)
        parent2 = Individual([5, 4, 3, 2, 1], fitness=20.0)
        
        child1, child2 = crossover_pmx(parent1, parent2)
        
        assert child1.fitness is None
        assert child2.fitness is None

    def test_pmx_parents_unchanged(self):
        """Parents should not be modified by crossover."""
        orig_parent1 = [1, 2, 3, 4, 5]
        orig_parent2 = [5, 4, 3, 2, 1]
        
        parent1 = Individual(orig_parent1.copy())
        parent2 = Individual(orig_parent2.copy())
        
        crossover_pmx(parent1, parent2)
        
        assert parent1.chromosome == orig_parent1
        assert parent2.chromosome == orig_parent2
