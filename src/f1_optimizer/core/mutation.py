import random

from f1_optimizer.core.models import Individual


def swap_mutation(individual: Individual, num_swaps: int) -> Individual:
    """Clone individual and perform num_swaps random position swaps."""
    mutated = individual.clone()
    for _ in range(num_swaps):
        i, j = random.sample(range(len(mutated.chromosome)), 2)
        mutated.chromosome[i], mutated.chromosome[j] = mutated.chromosome[j], mutated.chromosome[i]
    return mutated
