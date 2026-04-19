"""Configuration loading and management for F1 GA Optimizer."""

from pathlib import Path
import tomllib

from f1_optimizer.core.ga.ga import GAConfig
from f1_optimizer.core.evaluation.fitness import FitnessConfig


class ConfigLoadError(Exception):
    """Error loading or parsing config file."""
    pass


def load_config_file(config_path: Path | None) -> tuple[FitnessConfig, GAConfig]:
    """Load and merge configuration from TOML file with defaults.
    
    Args:
        config_path: Path to TOML config file, or None for defaults only
        
    Returns:
        (FitnessConfig, GAConfig) instances with merged values
        
    Raises:
        ConfigLoadError: If config file is invalid or can't be parsed
    """
    # Start with defaults
    fitness_config = FitnessConfig()
    ga_config = GAConfig()
    
    if config_path is None:
        return fitness_config, ga_config
    
    # Validate file exists
    if not config_path.exists():
        raise ConfigLoadError(f"Config file not found: {config_path}")
    
    if not config_path.is_file():
        raise ConfigLoadError(f"Config path is not a file: {config_path}")
    
    # Parse TOML
    try:
        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigLoadError(f"Invalid TOML syntax in {config_path}: {e}")
    except Exception as e:
        raise ConfigLoadError(f"Failed to read {config_path}: {e}")
    
    # Extract sections
    fitness_section = config_dict.get("fitness", {})
    ga_section = config_dict.get("ga", {})
    
    # Create updated configs with merged values
    try:
        fitness_config = FitnessConfig(**fitness_section)
    except Exception as e:
        raise ConfigLoadError(f"Invalid fitness config: {e}")
    
    try:
        ga_config = GAConfig(**ga_section)
    except Exception as e:
        raise ConfigLoadError(f"Invalid GA config: {e}")
    
    return fitness_config, ga_config


def create_example_config() -> str:
    """Generate an example config file as string."""
    return """# F1 GA Optimizer Configuration

[fitness]
# Penalty coefficients (K_* parameters)
K_dist = 1.0           # Penalty per km of travel
K_env = 1000.0         # Penalty per °C temperature deviation
K_utc = 500.0          # Base penalty for timezone shifts
K_double = 5000.0      # Penalty for double-header week
K_triple = 15000.0     # Penalty for triple-header week
K_bounce = 10000.0     # Penalty for region change
K_flyaway = 50000.0    # Penalty for inter-continental back-to-back
M = 1000000.0          # Big-M multiplier for hard constraint violations

# Optimization weights (should sum to ~1.0)
w_dist = 0.25          # Weight for distance penalty
w_env = 0.25           # Weight for environment penalty
w_log = 0.25           # Weight for logistics penalty
w_hum = 0.25           # Weight for human fatigue penalties

# Temperature thresholds (°C)
T_hard_min = 5.0       # Hard constraint minimum
T_hard_max = 40.0      # Hard constraint maximum
T_opt_min = 18.0       # Soft constraint minimum (optimal)
T_opt_max = 28.0       # Soft constraint maximum (optimal)

# Back-to-back distance limit (km)
L_max = 8000.0

# Seasonal windows (0-indexed week ranges, inclusive)
summer_break_window = [17, 26]    # Summer break window
ramadan_window = [6, 10]          # Ramadan observance window

[ga]
# Population and evolution
population_size = 50       # Number of individuals per generation
generations = 100          # Number of generations to run
stagnation_threshold = 20  # Early stopping patience (gens without improvement)
elite_size = 2             # Number of elite individuals to preserve

# Operators
crossover_type = "ox1"     # "ox1" or "pmx"
mutation_type = "swap"     # "swap" or "inversion"

# Probabilities
crossover_probability = 0.8  # Chance of crossover per pair
mutation_probability = 0.1   # Chance of mutation per individual

# Selection
tournament_size = 5        # Tournament selection pool size
"""
