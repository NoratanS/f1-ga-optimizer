from f1_optimizer.core.models import Individual

class TestIndividualInitialization:
    """Tests for Individual initialization."""

    def test_init_with_chromosome_and_fitness(self):
        # Arrange
        chromosome = [1, 2, 3, 4, 5]
        fitness = 85.5

        # Act
        individual = Individual(chromosome=chromosome, fitness=fitness)

        # Assert
        assert individual.chromosome == chromosome
        assert individual.fitness == fitness

    def test_init_with_chromosome_only(self):
        # Arrange
        chromosome = [1, 2, 3, 4, 5]

        # Act
        individual = Individual(chromosome=chromosome)

        # Assert
        assert individual.chromosome == chromosome
        assert individual.fitness is None

    def test_init_converts_chromosome_to_list(self):
        # Arrange
        chromosome = (1, 2, 3, 4, 5)  # tuple input

        # Act
        individual = Individual(chromosome=chromosome)

        # Assert
        assert isinstance(individual.chromosome, list)
        assert individual.chromosome == [1, 2, 3, 4, 5]

    def test_init_creates_defensive_copy_of_chromosome(self):
        # Arrange
        original_chromosome = [1, 2, 3, 4, 5]

        # Act
        individual = Individual(chromosome=original_chromosome)
        original_chromosome[0] = 999

        # Assert
        assert individual.chromosome[0] == 1  # unaffected by external change
        assert individual.chromosome != original_chromosome


class TestIndividualClone:
    """Tests for Individual clone method (deep copy for genetic operations)."""

    def test_clone_creates_new_instance(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.5)

        # Act
        cloned = individual.clone()

        # Assert
        assert cloned is not individual
        assert id(cloned) != id(individual)

    def test_clone_copies_chromosome(self):
        # Arrange
        chromosome = [1, 2, 3, 4, 5]
        individual = Individual(chromosome=chromosome, fitness=85.5)

        # Act
        cloned = individual.clone()

        # Assert
        assert cloned.chromosome == individual.chromosome
        assert cloned.chromosome == chromosome

    def test_clone_creates_independent_chromosome(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.5)

        # Act
        cloned = individual.clone()
        cloned.chromosome[0] = 999

        # Assert
        assert individual.chromosome[0] == 1  # original unaffected
        assert cloned.chromosome[0] == 999  # clone modified

    def test_clone_preserves_fitness(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.5)

        # Act
        cloned = individual.clone()

        # Assert
        assert cloned.fitness == individual.fitness
        assert cloned.fitness == 85.5

    def test_clone_preserves_unassessed_fitness(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5])

        # Act
        cloned = individual.clone()

        # Assert
        assert cloned.fitness is None
        assert cloned.fitness == individual.fitness

    def test_clone_fitness_independence(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.5)

        # Act
        cloned = individual.clone()
        cloned.fitness = 95.0

        # Assert
        assert individual.fitness == 85.5  # original unaffected
        assert cloned.fitness == 95.0  # clone modified


class TestIndividualRepr:
    """Tests for Individual __repr__ method."""

    def test_repr_unassessed_individual(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5])

        # Act
        repr_str = repr(individual)

        # Assert
        assert "Individual" in repr_str
        assert "Unassessed" in repr_str
        assert "[1, 2, 3, 4, 5]" in repr_str

    def test_repr_assessed_individual(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.5)

        # Act
        repr_str = repr(individual)

        # Assert
        assert "Individual" in repr_str
        assert "85.50" in repr_str
        assert "[1, 2, 3, 4, 5]" in repr_str

    def test_repr_with_large_fitness_value(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=1234567.89)

        # Act
        repr_str = repr(individual)

        # Assert
        assert "1,234,567.89" in repr_str  # comma-formatted

    def test_repr_with_zero_fitness(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5], fitness=0.0)

        # Act
        repr_str = repr(individual)

        # Assert
        assert "0.00" in repr_str
        assert "Unassessed" not in repr_str

    def test_repr_shows_first_five_chromosome_elements(self):
        # Arrange
        chromosome = list(range(1, 26))  # 25 elements
        individual = Individual(chromosome=chromosome, fitness=85.5)

        # Act
        repr_str = repr(individual)

        # Assert
        assert "[1, 2, 3, 4, 5]" in repr_str
        assert "..." in repr_str  # indicates truncation

    def test_repr_short_chromosome(self):
        # Arrange
        chromosome = [1, 2]
        individual = Individual(chromosome=chromosome, fitness=85.5)

        # Act
        repr_str = repr(individual)

        # Assert
        assert "[1, 2]" in repr_str


class TestIndividualGeneticAlgorithmContext:
    """Tests for Individual behavior in genetic algorithm scenarios."""

    def test_multiple_individuals_are_independent(self):
        # Arrange
        ind1 = Individual(chromosome=[1, 2, 3], fitness=80.0)
        ind2 = Individual(chromosome=[4, 5, 6], fitness=90.0)

        # Act
        ind1.fitness = 85.0

        # Assert
        assert ind1.fitness == 85.0
        assert ind2.fitness == 90.0

    def test_population_of_individuals(self):
        # Arrange
        population = [
            Individual(chromosome=[1, 2, 3, 4, 5]),
            Individual(chromosome=[6, 7, 8, 9, 10]),
            Individual(chromosome=[11, 12, 13, 14, 15]),
        ]

        # Act
        for i, individual in enumerate(population):
            individual.fitness = 80.0 + i * 5

        # Assert
        assert len(population) == 3
        assert population[0].fitness == 80.0
        assert population[1].fitness == 85.0
        assert population[2].fitness == 90.0

    def test_individual_fitness_update_for_evaluation(self):
        # Arrange
        individual = Individual(chromosome=[1, 2, 3, 4, 5])
        assert individual.fitness is None

        # Act
        individual.fitness = 87.5

        # Assert
        assert individual.fitness == 87.5

    def test_clone_for_mutation_operations(self):
        # Arrange
        parent = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.0)

        # Act
        mutant = parent.clone()
        mutant.chromosome[0] = 999  # simulate mutation
        mutant.fitness = None  # reset fitness for re-evaluation

        # Assert
        assert parent.chromosome[0] == 1  # parent unchanged
        assert mutant.chromosome[0] == 999  # mutant has mutation
        assert parent.fitness == 85.0  # parent fitness unchanged
        assert mutant.fitness is None  # mutant needs re-evaluation

    def test_clone_for_crossover_operations(self):
        # Arrange
        parent1 = Individual(chromosome=[1, 2, 3, 4, 5], fitness=85.0)
        parent2 = Individual(chromosome=[6, 7, 8, 9, 10], fitness=90.0)

        # Act
        # Simulate crossover: child takes first 2 from parent1, rest from parent2
        child_chromosome = parent1.chromosome[:2] + parent2.chromosome[2:]
        child = Individual(chromosome=child_chromosome, fitness=None)

        # Assert
        assert child.chromosome == [1, 2, 8, 9, 10]
        assert child.fitness is None  # needs evaluation
        assert parent1.chromosome == [1, 2, 3, 4, 5]  # parent1 unchanged
        assert parent2.chromosome == [6, 7, 8, 9, 10]  # parent2 unchanged
