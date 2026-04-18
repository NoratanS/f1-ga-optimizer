from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .hard_constraints import evaluate_hard_constraints
from .soft_constraints import evaluate_soft_constraints
from f1_optimizer.core.models import Individual
from .distance_matrix import DistanceMatrix


class FitnessConfig(BaseModel):
    """Penalty coefficients and constraint thresholds for fitness evaluation."""
    
    # Penalty coefficients (K_* parameters from constraints.md)
    K_dist: float = Field(default=1.0, gt=0, description="Penalty per km of travel")
    K_env: float = Field(default=1000.0, gt=0, description="Penalty per °C temp deviation")
    K_utc: float = Field(default=500.0, gt=0, description="Base penalty for timezone shifts")
    K_double: float = Field(default=5000.0, gt=0, description="Penalty for double-header week")
    K_triple: float = Field(default=15000.0, gt=0, description="Penalty for triple-header week")
    K_bounce: float = Field(default=10000.0, gt=0, description="Penalty for region change")
    K_flyaway: float = Field(default=50000.0, gt=0, description="Penalty for inter-continental back-to-back")
    M: float = Field(default=1000000.0, gt=0, description="Big-M multiplier for hard violations")
    
    # Optimization weights (must sum to ~1.0)
    w_dist: float = Field(default=0.25, ge=0, le=1, description="Weight for distance penalty")
    w_env: float = Field(default=0.25, ge=0, le=1, description="Weight for environment penalty")
    w_log: float = Field(default=0.25, ge=0, le=1, description="Weight for logistics penalty")
    w_hum: float = Field(default=0.25, ge=0, le=1, description="Weight for human fatigue penalties")
    
    # Temperature thresholds
    T_hard_min: float = Field(default=5.0, description="Hard constraint minimum temperature (°C)")
    T_hard_max: float = Field(default=40.0, description="Hard constraint maximum temperature (°C)")
    T_opt_min: float = Field(default=18.0, description="Soft constraint minimum temperature (°C)")
    T_opt_max: float = Field(default=28.0, description="Soft constraint maximum temperature (°C)")
    
    # Back-to-back distance limit
    L_max: float = Field(default=8000.0, gt=0, description="Max back-to-back distance")

    # Seasonal windows (0-indexed week ranges, inclusive)
    summer_break_window: tuple[int, int] = Field(
        default=(17, 26), description="Summer break window (0-indexed)"
    )
    ramadan_window: tuple[int, int] = Field(
        default=(6, 10), description="Ramadan observance window (0-indexed)"
    )

    @field_validator("w_dist", "w_env", "w_log", "w_hum", mode="after")
    @classmethod
    def validate_weights_individual(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("Individual weights must be in range [0, 1]")
        return v

    @field_validator("T_hard_min", "T_hard_max", "T_opt_min", "T_opt_max")
    @classmethod
    def validate_temperature_order(cls, v: float) -> float:
        return v

    def validate_temperatures(self) -> None:
        """Raise ValueError if temperature bounds are illogical."""
        if self.T_hard_min >= self.T_hard_max:
            raise ValueError(
                f"Hard temp min ({self.T_hard_min}) must be < max ({self.T_hard_max})"
            )
        if self.T_opt_min >= self.T_opt_max:
            raise ValueError(
                f"Optimal temp min ({self.T_opt_min}) must be < max ({self.T_opt_max})"
            )
        if self.T_opt_min < self.T_hard_min or self.T_opt_max > self.T_hard_max:
            raise ValueError(
                f"Optimal window [{self.T_opt_min}, {self.T_opt_max}] must be "
                f"within hard bounds [{self.T_hard_min}, {self.T_hard_max}]"
            )

    def validate_weights_sum(self) -> None:
        """Raise ValueError if weights don't sum to ~1.0."""
        total = self.w_dist + self.w_env + self.w_log + self.w_hum
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f} "
                f"(w_dist={self.w_dist}, w_env={self.w_env}, "
                f"w_log={self.w_log}, w_hum={self.w_hum})"
            )

    def validate_seasonal_windows(self, num_weeks: int) -> None:
        """Raise ValueError if windows exceed season bounds."""
        for window_name, (start, end) in [
            ("summer_break", self.summer_break_window),
            ("ramadan", self.ramadan_window),
        ]:
            if not (0 <= start < end < num_weeks):
                raise ValueError(
                    f"{window_name} window ({start}, {end}) invalid for season of {num_weeks} weeks"
                )


# ============================================================================
# Fitness Evaluator - Orchestrates HC + SC into Total Fitness Score
# ============================================================================


class FitnessEvaluator:
    """Combines hard and soft constraints: F = Z + (M * V_total)."""
    
    def __init__(
        self,
        distance_matrix: DistanceMatrix,
        config: FitnessConfig
    ):
        """Initialize with distance matrix and fitness config."""
        self.distance_matrix = distance_matrix
        self.config = config
        
        # Auto-detect Middle East races from data (not from config)
        self.middle_east_races = distance_matrix.get_races_by_region("Middle East")
        
        # Validate config at initialization time
        self.config.validate_weights_sum()
        self.config.validate_temperatures()
        self.config.validate_seasonal_windows(distance_matrix.num_weeks)
    
    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate fitness score: F = Z + (M * V_total)."""
        
        # Evaluate hard constraints
        hc_violations = evaluate_hard_constraints(
            individual.chromosome,
            self.distance_matrix,
            self.config,
            self.middle_east_races
        )
        
        # Evaluate soft constraints
        sc_penalties = evaluate_soft_constraints(
            individual.chromosome,
            self.distance_matrix,
            self.config
        )
        
        # Compute weighted soft objective Z
        Z = (
            self.config.w_dist * sc_penalties.distance +
            self.config.w_env * sc_penalties.environment +
            self.config.w_log * sc_penalties.logistics +
            self.config.w_hum * (sc_penalties.jetlag + sc_penalties.headers)
        )
        
        # Apply Big-M penalty for hard violations
        F = Z + (self.config.M * hc_violations.total)
        
        return F
    
    def evaluate_population(self, population: list[Individual]) -> None:
        """Evaluate fitness for all individuals in-place."""
        for individual in population:
            individual.fitness = self.evaluate_individual(individual)
