"""Configuration loading for F1 GA Optimizer."""

from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

from f1_optimizer.core.evaluation.fitness import FitnessConfig
from f1_optimizer.core.ga.ga import GAConfig


class ConfigLoadError(Exception):
    """Error loading or parsing config file."""


def load_fitness_config(config_path: Path | None) -> FitnessConfig:
    """Load FitnessConfig from a flat TOML file, or return defaults."""
    if config_path is None:
        return FitnessConfig()
    raw = _load_toml(config_path)
    try:
        return FitnessConfig(**raw)
    except Exception as e:
        raise ConfigLoadError(f"Invalid fitness config: {e}") from e


def load_ga_config(config_path: Path | None) -> GAConfig:
    """Load GAConfig from a flat TOML file, or return defaults.

    Raises ConfigLoadError if the file contains list values — use expand_ga_configs for multi-config.
    """
    if config_path is None:
        return GAConfig()
    raw = _load_toml(config_path)
    list_fields = [k for k, v in raw.items() if isinstance(v, list)]
    if list_fields:
        raise ConfigLoadError(
            f"Config has list values for: {list_fields}. Use --multi-config to run permutations."
        )
    try:
        return GAConfig(**raw)
    except Exception as e:
        raise ConfigLoadError(f"Invalid GA config: {e}") from e


def expand_ga_configs(config_path: Path | None) -> list[tuple[str, GAConfig]]:
    """Parse a flat GA config TOML supporting list values, expand into all permutations.

    Returns a list of (config_name, GAConfig) tuples.
    """
    if config_path is None:
        return [("config_1", GAConfig())]
    raw = _load_toml(config_path)
    scalars = {k: v for k, v in raw.items() if not isinstance(v, list)}
    lists = {k: v for k, v in raw.items() if isinstance(v, list)}

    field_names = list(lists.keys())
    field_values = list(lists.values())
    configs: list[tuple[str, GAConfig]] = []
    for i, combo in enumerate(itertools.product(*field_values)):
        combo_dict = {**scalars, **dict(zip(field_names, combo))}
        name_parts = [f"config_{i + 1}"]
        for field, val in zip(field_names, combo):
            name_parts.append(f"{_abbr(field)}{val}")
        name = "_".join(name_parts)
        try:
            ga_config = GAConfig(**combo_dict)
        except Exception as e:
            raise ConfigLoadError(f"Invalid config for {name}: {e}") from e
        configs.append((name, ga_config))
    return configs


_ABBR: dict[str, str] = {
    "population_size": "ps",
    "generations": "g",
    "crossover_probability": "cp",
    "mutation_probability": "mp",
    "tournament_size": "ts",
    "elite_size": "es",
    "stagnation_threshold": "st",
    "crossover_type": "cx",
    "mutation_type": "mt",
}


def _abbr(field: str) -> str:
    return _ABBR.get(field, field[:2])


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigLoadError(f"Config file not found: {path}")
    except IsADirectoryError:
        raise ConfigLoadError(f"Config path is not a file: {path}")
    except tomllib.TOMLDecodeError as e:
        raise ConfigLoadError(f"Invalid TOML in {path}: {e}") from e
    except Exception as e:
        raise ConfigLoadError(f"Failed to read {path}: {e}") from e


def create_example_fitness_config() -> str:
    return """\
# F1 GA Optimizer - Fitness Configuration
K_dist = 1.0
K_env = 1000.0
K_utc = 500.0
K_double = 5000.0
K_triple = 15000.0
K_bounce = 10000.0
K_flyaway = 50000.0
M = 1000000.0

w_dist = 0.25
w_env = 0.25
w_log = 0.25
w_hum = 0.25

T_hard_min = 5.0
T_hard_max = 40.0
T_opt_min = 18.0
T_opt_max = 28.0

summer_break_window = [17, 26]
ramadan_window = [6, 10]
"""


def create_example_ga_config() -> str:
    return """\
# F1 GA Optimizer - GA Configuration
population_size = 100
generations = 1000
crossover_probability = 0.8
mutation_probability = 0.1
crossover_type = "ox1"
mutation_type = "swap"
tournament_size = 5
stagnation_threshold = 50
elite_size = 2
"""
