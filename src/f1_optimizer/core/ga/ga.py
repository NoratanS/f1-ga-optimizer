"""Genetic Algorithm for F1 calendar optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict

from f1_optimizer.core.models import Individual
from f1_optimizer.core.ga.selection import tournament_selection
from f1_optimizer.core.ga.mutation import swap_mutation, inversion_mutation
from f1_optimizer.core.ga.crossover import crossover_ox1, crossover_pmx
from f1_optimizer.core.evaluation.fitness import FitnessEvaluator


class GAConfig(BaseModel):
    """Configuration for Genetic Algorithm."""
    
    population_size: int = Field(default=50, gt=0, description="Population size")
    generations: int = Field(default=100, gt=0, description="Number of generations")
    crossover_probability: float = Field(default=0.8, ge=0, le=1, description="Crossover probability")
    mutation_probability: float = Field(default=0.1, ge=0, le=1, description="Mutation probability per individual")
    tournament_size: int = Field(default=5, gt=0, description="Tournament selection size")
    crossover_type: str = Field(default="ox1", description="Crossover type: 'ox1' or 'pmx'")
    mutation_type: str = Field(default="swap", description="Mutation type: 'swap' or 'inversion'")
    elite_size: int = Field(default=2, ge=0, description="Number of elite individuals to preserve")
    patience: int = Field(default=20, ge=1, description="Early stopping patience (generations without improvement)")
    
    model_config = ConfigDict(validate_assignment=True)


@dataclass
class GAReport:
    """Report from GA execution."""
    
    best_fitness: float
    best_individual: Individual
    avg_fitness_per_gen: list[float]
    best_fitness_per_gen: list[float]
    generations_run: int
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"GA Report:\n"
            f"  Best Fitness: {self.best_fitness:.2f}\n"
            f"  Generations Run: {self.generations_run}\n"
            f"  Best Individual: {self.best_individual}"
        )


def initialize_population(population_size: int, num_races: int, num_weeks: int) -> list[Individual]:
    """Initialize population with random permutations.
    
    Args:
        population_size: Number of individuals
        num_races: Number of races (1 to num_races)
        num_weeks: Total number of weeks in the schedule
        
    Returns:
        List of initialized individuals with random chromosomes of size num_weeks
    """
    population = []
    # Chromosome: races 1..num_races + breaks 100..100+(num_weeks-num_races)-1
    # Total chromosome size: num_weeks
    num_breaks = num_weeks - num_races
    genes = list(range(1, num_races + 1)) + list(range(100, 100 + num_breaks))
    
    for _ in range(population_size):
        chromosome = genes.copy()
        random.shuffle(chromosome)
        population.append(Individual(chromosome, fitness=None))
    
    return population


class GeneticAlgorithm:
    """Genetic Algorithm for F1 calendar optimization."""
    
    def __init__(
        self,
        config: GAConfig,
        fitness_evaluator: FitnessEvaluator,
    ):
        """Initialize GA.
        
        Args:
            config: GA configuration
            fitness_evaluator: FitnessEvaluator instance with distance matrix
        """
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        
        # Extract constants from distance matrix (no track_db needed)
        self.num_races = fitness_evaluator.distance_matrix.num_races
        self.num_weeks = fitness_evaluator.distance_matrix.num_weeks
    
    def evolve(self) -> GAReport:
        """Run GA evolution loop and return report.
        
        Returns:
            GAReport with best individual, fitness history, and execution stats
        """
        # Initialize population
        population = initialize_population(self.config.population_size, self.num_races, self.num_weeks)
        
        # Evaluate initial population
        self.fitness_evaluator.evaluate_population(population)
        
        # Sort population for elite extraction in loop
        population = sorted(population, key=lambda ind: ind.fitness)
        
        # Track best
        best_individual = population[0].clone()
        best_fitness = population[0].fitness
        generations_without_improvement = 0
        
        # History tracking
        best_fitness_per_gen = [best_fitness]
        avg_fitness_per_gen = [sum(ind.fitness for ind in population) / len(population)]
        
        # Select operators based on config
        crossover_fn = crossover_ox1 if self.config.crossover_type == "ox1" else crossover_pmx
        mutation_fn = swap_mutation if self.config.mutation_type == "swap" else inversion_mutation
        
        # Main loop
        for generation in range(self.config.generations):
            # Check early stopping
            if generations_without_improvement >= self.config.patience:
                break
            
            # Elitism: preserve best individuals (population is already sorted from previous iteration)
            elite = population[:self.config.elite_size]
            
            # Generate offspring through selection, crossover, mutation
            offspring = []
            while len(offspring) < self.config.population_size - self.config.elite_size:
                # Selection
                parent1 = tournament_selection(population, self.config.tournament_size)
                parent2 = tournament_selection(population, self.config.tournament_size)
                
                # Crossover
                if random.random() < self.config.crossover_probability:
                    child1, child2 = crossover_fn(parent1, parent2)
                else:
                    child1, child2 = parent1.clone(), parent2.clone()
                
                # Mutation
                if random.random() < self.config.mutation_probability:
                    child1 = mutation_fn(child1)
                if random.random() < self.config.mutation_probability:
                    child2 = mutation_fn(child2)
                
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            self.fitness_evaluator.evaluate_population(offspring)
            
            # Merge elite + offspring, sort, and trim to population size
            population = sorted(elite + offspring, key=lambda ind: ind.fitness)[:self.config.population_size]
            
            # Track best (first element after sorting)
            if population[0].fitness < best_fitness:
                best_individual = population[0].clone()
                best_fitness = population[0].fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # History
            best_fitness_per_gen.append(best_fitness)
            avg_fitness_per_gen.append(sum(ind.fitness for ind in population) / len(population))
        
        return GAReport(
            best_fitness=best_fitness,
            best_individual=best_individual,
            best_fitness_per_gen=best_fitness_per_gen,
            avg_fitness_per_gen=avg_fitness_per_gen,
            generations_run=generation + 1 if generation >= 0 else 0,
        )
