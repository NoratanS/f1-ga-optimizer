from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from .utils import extract_race_sequence
from .distance_matrix import DistanceMatrix

if TYPE_CHECKING:
    from .fitness import FitnessConfig


class HardConstraintViolations(BaseModel):
    """Immutable result object for all HC1-HC7 violation counts."""
    
    model_config = ConfigDict(frozen=True)

    completeness: int = Field(default=0, ge=0, description="HC1 violation count")
    capacity: int = Field(default=0, ge=0, description="HC2 violation count")
    weather: int = Field(default=0, ge=0, description="HC3 violation count")
    freight: int = Field(default=0, ge=0, description="HC4 violation count")
    burnout: int = Field(default=0, ge=0, description="HC5 violation count")
    break_requirement: int = Field(default=0, ge=0, description="HC6 violation count")
    ramadan: int = Field(default=0, ge=0, description="HC7 violation count")

    @property
    def total(self) -> int:
        """Sum of all violation counts."""
        return sum(self.model_dump().values())


# ============================================================================
# Hard Constraint Evaluation Functions (HC1-HC7)
# ============================================================================


def count_completeness_violations(chromosome: list[int], num_races: int) -> int:
    """HC1: Count races not scheduled exactly once."""
    race_counts = [0] * (num_races + 1)  # Index 0 unused, 1 to num_races
    
    for race_id in chromosome:
        if race_id < 100:  # Valid race ID
            race_counts[race_id] += 1
    
    # Count violations: sum of |1 - count| for each race
    violations = sum(abs(1 - race_counts[i]) for i in range(1, num_races + 1))
    return violations


def count_capacity_violations(chromosome: list[int]) -> int:
    """HC2: Count weeks with more than one race."""
    violations = 0
    
    for week in range(len(chromosome)):
        race_id = chromosome[week]
        # Count races scheduled in same week (multiple races if race_id < 100)
        # In this GA representation, max 1 per week by design
        # But we still check: if somehow duplicate, count excess
        races_in_week = sum(1 for w in range(len(chromosome)) if chromosome[w] == race_id and race_id < 100)
        if races_in_week > 1:
            violations += races_in_week - 1
            break  # Avoid double-counting same race
    
    # Simpler approach: check each week position
    violations = 0
    for week in range(len(chromosome)):
        if chromosome[week] < 100:  # If race scheduled this week
            # Count how many races in this week
            races_this_week = sum(1 for race_id in [chromosome[week]] if race_id < 100)
            if races_this_week > 1:
                violations += races_this_week - 1
    
    return violations


def count_weather_violations(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> int:
    """HC3: Count races outside hard temperature bounds."""
    violations = 0
    
    for week, race_id in enumerate(chromosome):
        if race_id < 100:  # Valid race scheduled this week
            temp = distance_matrix.get_temperature(race_id, week)
            if temp < config.T_hard_min or temp > config.T_hard_max:
                violations += 1
    
    return violations


def count_freight_violations(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig
) -> int:
    """HC4: Count back-to-back races exceeding L_max distance."""
    omega, tau = extract_race_sequence(chromosome)
    violations = 0
    
    for k in range(len(omega) - 1):
        # Check if races k and k+1 are back-to-back (consecutive weeks)
        if tau[k + 1] - tau[k] == 1:
            distance = distance_matrix.get_distance(omega[k], omega[k + 1])
            if distance > config.L_max:
                violations += 1
    
    return violations


def count_burnout_violations(chromosome: list[int]) -> int:
    """HC5: Count windows with 4+ consecutive race weeks."""
    # Build y_t vector (1 if race in week, 0 if break)
    y = [1 if chromosome[t] < 100 else 0 for t in range(len(chromosome))]
    
    violations = 0
    
    # Check all windows of 4 consecutive weeks
    for t in range(len(y) - 3):
        races_in_window = y[t] + y[t + 1] + y[t + 2] + y[t + 3]
        if races_in_window > 3:
            violations += races_in_window - 3
    
    return violations


def count_break_violations(
    chromosome: list[int],
    config: FitnessConfig
) -> int:
    """HC6: Check for 3 consecutive off-weeks in summer window."""
    # Build y_t vector
    y = [1 if chromosome[t] < 100 else 0 for t in range(len(chromosome))]
    
    start_week, end_week = config.summer_break_window
    
    # Check for 3 consecutive off-weeks within summer window
    for t in range(start_week, min(end_week - 2, len(y) - 2)):
        if y[t] + y[t + 1] + y[t + 2] == 0:
            return 0  # Found valid break
    
    return 1  # No valid break found


def count_ramadan_violations(
    chromosome: list[int],
    config: FitnessConfig,
    middle_east_races: set[int]
) -> int:
    """HC7: Count Middle East races during Ramadan window."""
    if not middle_east_races:
        return 0  # No Middle East races in dataset
    
    violations = 0
    start_week, end_week = config.ramadan_window
    
    for week in range(start_week, min(end_week + 1, len(chromosome))):
        race_id = chromosome[week]
        if race_id in middle_east_races:
            violations += 1
    
    return violations


def evaluate_hard_constraints(
    chromosome: list[int],
    distance_matrix: DistanceMatrix,
    config: FitnessConfig,
    middle_east_races: set[int]
) -> HardConstraintViolations:
    """Evaluate all HC1-HC7 constraints and return violation counts."""
    return HardConstraintViolations(
        completeness=count_completeness_violations(chromosome, distance_matrix.num_races),
        capacity=count_capacity_violations(chromosome),
        weather=count_weather_violations(chromosome, distance_matrix, config),
        freight=count_freight_violations(chromosome, distance_matrix, config),
        burnout=count_burnout_violations(chromosome),
        break_requirement=count_break_violations(chromosome, config),
        ramadan=count_ramadan_violations(chromosome, config, middle_east_races),
    )
