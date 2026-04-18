from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from .utils import extract_race_sequence
from .distance_matrix import DistanceMatrix

if TYPE_CHECKING:
    from .fitness import FitnessConfig


class SoftConstraintPenalties(BaseModel):
    """Immutable result object for all SC1-SC5 penalty values."""
    
    model_config = ConfigDict(frozen=True)

    distance: float = Field(default=0.0, ge=0.0, description="Z_dist distance penalty")
    environment: float = Field(default=0.0, ge=0.0, description="Z_env environment penalty")
    jetlag: float = Field(default=0.0, ge=0.0, description="Z_utc jetlag penalty")
    headers: float = Field(default=0.0, ge=0.0, description="Z_headers schedule intensity penalty")
    logistics: float = Field(default=0.0, ge=0.0, description="Z_log logistics penalty")
    
    @property
    def total(self) -> float:
        """Sum of all penalties."""
        return sum(self.model_dump().values())


# ============================================================================
# Soft Constraint Penalty Functions (SC1-SC5)
# ============================================================================


def calc_distance_penalty(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> float:
    """SC1: Penalize total travel distance."""
    omega, _ = extract_race_sequence(chromosome)
    
    total_distance = 0.0
    for k in range(len(omega) - 1):
        distance = distance_matrix.get_distance(omega[k], omega[k + 1])
        total_distance += distance
    
    return config.K_dist * total_distance


def calc_environment_penalty(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> float:
    """SC2: Penalize races outside optimal temperature window."""
    total_penalty = 0.0
    
    for week, race_id in enumerate(chromosome):
        if race_id < 100:  # Valid race scheduled
            temp = distance_matrix.get_temperature(race_id, week)
            
            # Temperature deviation from optimal window
            if temp < config.T_opt_min:
                deviation = config.T_opt_min - temp
            elif temp > config.T_opt_max:
                deviation = temp - config.T_opt_max
            else:
                deviation = 0.0
            
            total_penalty += deviation
    
    return config.K_env * total_penalty


def calc_jetlag_penalty(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> float:
    """SC3: Penalize timezone jumps between consecutive races."""
    omega, _ = extract_race_sequence(chromosome)
    
    total_penalty = 0.0
    for k in range(len(omega) - 1):
        utc_diff = distance_matrix.get_circular_utc_distances(omega[k], omega[k + 1])
        total_penalty += utc_diff ** 2
    
    return config.K_utc * total_penalty


def calc_headers_penalty(
    chromosome: list[int],
    config: FitnessConfig
) -> float:
    """SC4: Penalize double and triple header weeks."""
    # Build y_t vector
    y = [1 if chromosome[t] < 100 else 0 for t in range(len(chromosome))]
    
    # Count double-headers
    n_double = 0
    for t in range(len(y) - 1):
        if y[t] == 1 and y[t + 1] == 1:
            n_double += 1
    
    # Count triple-headers
    n_triple = 0
    for t in range(len(y) - 2):
        if y[t] == 1 and y[t + 1] == 1 and y[t + 2] == 1:
            n_triple += 1
    
    return (config.K_double * n_double) + (config.K_triple * n_triple)


def calc_logistics_penalty(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> float:
    """SC5: Penalize region bouncing and inter-continental back-to-backs."""
    omega, tau = extract_race_sequence(chromosome)
    
    # Count region bounces (any consecutive race change)
    p_bounce = 0
    for k in range(len(omega) - 1):
        region_k = distance_matrix.get_region(omega[k])
        region_k1 = distance_matrix.get_region(omega[k + 1])
        if region_k != region_k1:
            p_bounce += 1
    
    # Count flyaway back-to-backs (back-to-back with region change)
    p_flyaway = 0
    for k in range(len(omega) - 1):
        # Check if consecutive weeks AND different regions
        if tau[k + 1] - tau[k] == 1:
            region_k = distance_matrix.get_region(omega[k])
            region_k1 = distance_matrix.get_region(omega[k + 1])
            if region_k != region_k1:
                p_flyaway += 1
    
    return (config.K_bounce * p_bounce) + (config.K_flyaway * p_flyaway)


def evaluate_soft_constraints(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> SoftConstraintPenalties:
    """Evaluate all SC1-SC5 constraints and return penalty values."""
    return SoftConstraintPenalties(
        distance=calc_distance_penalty(chromosome, distance_matrix, config),
        environment=calc_environment_penalty(chromosome, distance_matrix, config),
        jetlag=calc_jetlag_penalty(chromosome, distance_matrix, config),
        headers=calc_headers_penalty(chromosome, config),
        logistics=calc_logistics_penalty(chromosome, distance_matrix, config),
    )
