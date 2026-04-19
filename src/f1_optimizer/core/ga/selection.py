import random

from f1_optimizer.core.models import Individual


def tournament_selection(population: list[Individual], tournament_size: int) -> Individual:
    """Select best individual from random tournament sample."""
    if tournament_size > len(population):
        raise ValueError(
            f"tournament_size ({tournament_size}) cannot exceed population size ({len(population)})"
        )
    tournament = random.sample(population, tournament_size)
    if any(ind.fitness is None for ind in tournament):
        raise ValueError("Cannot select from unevaluated individuals (fitness is None)")
    return min(tournament, key=lambda ind: ind.fitness)
