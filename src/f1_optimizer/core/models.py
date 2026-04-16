from typing import List, Optional

class Individual:
    """Represents an individual in the population (one potential f1 calendar solution)."""
    def __init__(self, chromosome: List[int], fitness: Optional[float] = None):
        """Initializes an individual with a chromosome and an optional fitness value.
        Args:
            chromosome (List[int]): A list of race ID (1-24) and breaks (101+) representing the calendar.
            fitness (Optional[float]): The fitness value of the individual. Defaults to None, which means it has not been evaluated yet."""
        self.chromosome: List[int] = list(chromosome)
        self.fitness: Optional[float] = fitness
    
    def clone(self) -> 'Individual':
        """Creates a deep copy of the individual for use in genetic operations.
        Returns:
            Individual: A new instance of Individual with the same chromosome and fitness."""
        return Individual(chromosome=self.chromosome, fitness=self.fitness)

    def __repr__(self):
        fit_str = f"{self.fitness:,.2f}" if self.fitness is not None else "Unassessed"
        return f"<Individual | Fitness: {fit_str} | Chromosome[:5]: {self.chromosome[:5]}...>"