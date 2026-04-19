import random

from f1_optimizer.core.models import Individual


def crossover_ox1(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    """Order Crossover 1 (OX1) - Davies version for permutations.
    
    Selects a segment from parent1 between two cut points, preserves it in child1,
    then fills remaining positions from parent2 in order (skipping genes already in segment).
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        
    Returns:
        Tuple of two children individuals
    """
    n = len(parent1.chromosome)
    
    # Select two cut points
    point1 = random.randint(0, n - 2)
    point2 = random.randint(point1 + 1, n - 1)
    
    # Child1: copy segment [point1:point2] from parent1
    child1_chrom = [None] * n
    segment = parent1.chromosome[point1:point2]
    for i, gene in enumerate(segment):
        child1_chrom[point1 + i] = gene
    
    # Fill remaining positions from parent2 (in order, wrapping, skipping used genes)
    used_genes = set(segment)
    fill_pos = point2
    for gene in parent2.chromosome[point2:] + parent2.chromosome[:point2]:
        if gene not in used_genes:
            child1_chrom[fill_pos % n] = gene
            fill_pos += 1
    
    # Child2: same process with parent roles swapped
    child2_chrom = [None] * n
    segment = parent2.chromosome[point1:point2]
    for i, gene in enumerate(segment):
        child2_chrom[point1 + i] = gene
    
    used_genes = set(segment)
    fill_pos = point2
    for gene in parent1.chromosome[point2:] + parent1.chromosome[:point2]:
        if gene not in used_genes:
            child2_chrom[fill_pos % n] = gene
            fill_pos += 1
    
    return (
        Individual(child1_chrom, fitness=None),
        Individual(child2_chrom, fitness=None),
    )


def crossover_pmx(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    """Partially Mapped Crossover (PMX) - two-point crossover with mapping.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        
    Returns:
        Tuple of two children individuals
    """
    n = len(parent1.chromosome)
    point1 = random.randint(0, n - 2)
    point2 = random.randint(point1 + 1, n - 1)
    
    p1 = parent1.chromosome
    p2 = parent2.chromosome
    
    # Initialize children with -1 (placeholder)
    child1 = [-1] * n
    child2 = [-1] * n
    
    # Copy segment from parents (child1 gets from parent2, child2 from parent1)
    for i in range(point1, point2):
        child1[i] = p2[i]
        child2[i] = p1[i]
    
    # Create bidirectional mappings for the segment
    mapping1 = {}  # Maps p1[segment] -> p2[segment]
    mapping2 = {}  # Maps p2[segment] -> p1[segment]
    for i in range(point1, point2):
        mapping1[p1[i]] = p2[i]
        mapping2[p2[i]] = p1[i]
    
    # Fill remaining positions for child1 (from parent1, following mapping2 for conflicts)
    for i in range(n):
        if i < point1 or i >= point2:
            gene = p1[i]
            while gene in mapping2:
                gene = mapping2[gene]
            child1[i] = gene
    
    # Fill remaining positions for child2 (from parent2, following mapping1 for conflicts)
    for i in range(n):
        if i < point1 or i >= point2:
            gene = p2[i]
            while gene in mapping1:
                gene = mapping1[gene]
            child2[i] = gene
    
    return (
        Individual(child1, fitness=None),
        Individual(child2, fitness=None),
    )
