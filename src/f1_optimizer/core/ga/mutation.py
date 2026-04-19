import random

from f1_optimizer.core.models import Individual


def swap_mutation(individual: Individual) -> Individual:
    """Clone individual and perform one random position swap.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        New Individual with one position swapped and fitness=None
    """
    mutated = individual.clone()
    n = len(mutated.chromosome)
    i, j = random.sample(range(n), 2)
    mutated.chromosome[i], mutated.chromosome[j] = mutated.chromosome[j], mutated.chromosome[i]
    mutated.fitness = None
    return mutated


def inversion_mutation(individual: Individual) -> Individual:
    """Clone individual and perform one random segment reversal.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        New Individual with one reversed segment and fitness=None
    """
    mutated = individual.clone()
    n = len(mutated.chromosome)
    
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    # Reverse the segment [i:j+1]
    mutated.chromosome[i:j+1] = reversed(mutated.chromosome[i:j+1])
    
    mutated.fitness = None
    return mutated
