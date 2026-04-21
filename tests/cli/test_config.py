"""Tests for CLI configuration loading."""

import pytest

from f1_optimizer.cli.config import (
    ConfigLoadError,
    expand_ga_configs,
    load_fitness_config,
    load_ga_config,
)
from f1_optimizer.core.evaluation.fitness import FitnessConfig
from f1_optimizer.core.ga.ga import GAConfig


class TestLoadFitnessConfig:
    """Tests for load_fitness_config."""

    def test_returns_defaults_when_no_path(self):
        # Act
        config = load_fitness_config(None)

        # Assert
        assert isinstance(config, FitnessConfig)
        assert config == FitnessConfig()

    def test_loads_overridden_field_from_flat_toml(self, tmp_path):
        # Arrange
        toml_file = tmp_path / "fitness.toml"
        toml_file.write_text("K_dist = 2.0\n")

        # Act
        config = load_fitness_config(toml_file)

        # Assert
        assert config.K_dist == 2.0

    def test_raises_for_missing_file(self, tmp_path):
        # Arrange
        missing = tmp_path / "nonexistent.toml"

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="not found"):
            load_fitness_config(missing)

    def test_raises_for_invalid_toml_syntax(self, tmp_path):
        # Arrange
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("not valid toml ===")

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="(?i)toml"):
            load_fitness_config(bad_toml)

    def test_raises_for_invalid_field_value(self, tmp_path):
        # Arrange — K_dist must be > 0
        toml_file = tmp_path / "fitness.toml"
        toml_file.write_text("K_dist = -1.0\n")

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="(?i)fitness"):
            load_fitness_config(toml_file)


class TestLoadGaConfig:
    """Tests for load_ga_config."""

    def test_returns_defaults_when_no_path(self):
        # Act
        config = load_ga_config(None)

        # Assert
        assert isinstance(config, GAConfig)
        assert config == GAConfig()

    def test_loads_overridden_fields_from_flat_toml(self, tmp_path):
        # Arrange
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text('population_size = 200\ncrossover_type = "pmx"\n')

        # Act
        config = load_ga_config(toml_file)

        # Assert
        assert config.population_size == 200
        assert config.crossover_type == "pmx"

    def test_raises_for_missing_file(self, tmp_path):
        # Arrange
        missing = tmp_path / "nonexistent.toml"

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="not found"):
            load_ga_config(missing)

    def test_raises_for_list_values_with_multi_config_hint(self, tmp_path):
        # Arrange — list values require --multi-config flag
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text("population_size = [50, 100]\n")

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="--multi-config"):
            load_ga_config(toml_file)

    def test_raises_for_invalid_crossover_type(self, tmp_path):
        # Arrange
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text('crossover_type = "unknown"\n')

        # Act & Assert
        with pytest.raises(ConfigLoadError, match="(?i)GA config"):
            load_ga_config(toml_file)


class TestExpandGaConfigs:
    """Tests for expand_ga_configs."""

    def test_returns_single_default_config_when_no_path(self):
        # Act
        configs = expand_ga_configs(None)

        # Assert
        assert len(configs) == 1
        name, ga_config = configs[0]
        assert name == "config_1"
        assert isinstance(ga_config, GAConfig)

    def test_returns_single_config_for_flat_toml(self, tmp_path):
        # Arrange
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text("population_size = 100\n")

        # Act
        configs = expand_ga_configs(toml_file)

        # Assert
        assert len(configs) == 1
        assert configs[0][1].population_size == 100

    def test_expands_two_list_fields_into_cartesian_product(self, tmp_path):
        # Arrange — 2 × 2 = 4 permutations
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text("population_size = [50, 100]\ncrossover_probability = [0.7, 0.9]\n")

        # Act
        configs = expand_ga_configs(toml_file)

        # Assert
        assert len(configs) == 4
        pop_sizes = {ga_config.population_size for _, ga_config in configs}
        assert pop_sizes == {50, 100}

    def test_config_names_contain_field_abbreviations(self, tmp_path):
        # Arrange
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text("population_size = [50, 100]\n")

        # Act
        configs = expand_ga_configs(toml_file)

        # Assert
        names = [name for name, _ in configs]
        assert "config_1_ps50" in names
        assert "config_2_ps100" in names

    def test_raises_for_invalid_value_in_permutation(self, tmp_path):
        # Arrange — "invalid" is not a valid crossover_type
        toml_file = tmp_path / "ga.toml"
        toml_file.write_text('crossover_type = ["ox1", "invalid"]\n')

        # Act & Assert
        with pytest.raises(ConfigLoadError):
            expand_ga_configs(toml_file)
