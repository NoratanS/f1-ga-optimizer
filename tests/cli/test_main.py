"""Tests for CLI main commands."""

import tomllib

from typer.testing import CliRunner

from f1_optimizer.cli.main import app

runner = CliRunner()


class TestOptimizeCommand:
    """Tests for the optimize command."""

    def test_single_run_exits_zero_and_creates_output_files(self, data_dir, tmp_path):
        # Arrange
        config_file = tmp_path / "ga.toml"
        config_file.write_text("population_size = 10\ngenerations = 5\nstagnation_threshold = 3\n")
        output_dir = tmp_path / "out"

        # Act
        result = runner.invoke(app, [
            "optimize", str(data_dir / "data.csv"),
            "--config", str(config_file),
            "--output", str(output_dir),
        ])

        # Assert
        assert result.exit_code == 0
        assert (output_dir / "config.toml").exists()
        assert (output_dir / "schedule.md").exists()
        assert (output_dir / "plot.png").exists()

    def test_multi_config_creates_per_config_dirs_and_summary(self, data_dir, tmp_path):
        # Arrange
        config_file = tmp_path / "ga.toml"
        config_file.write_text(
            "population_size = [10, 20]\ngenerations = 5\nstagnation_threshold = 3\n"
        )
        output_dir = tmp_path / "out"

        # Act
        result = runner.invoke(app, [
            "optimize", str(data_dir / "data.csv"),
            "--config", str(config_file),
            "--multi-config",
            "--output", str(output_dir),
        ])

        # Assert
        assert result.exit_code == 0
        assert (output_dir / "config_1_ps10").is_dir()
        assert (output_dir / "config_2_ps20").is_dir()
        assert (output_dir / "summary.md").exists()

    def test_exits_one_for_list_values_without_multi_config_flag(self, data_dir, tmp_path):
        # Arrange
        config_file = tmp_path / "ga.toml"
        config_file.write_text("population_size = [10, 20]\n")

        # Act
        result = runner.invoke(app, [
            "optimize", str(data_dir / "data.csv"),
            "--config", str(config_file),
            "--output", str(tmp_path / "out"),
        ])

        # Assert
        assert result.exit_code == 1

    def test_exits_one_for_invalid_toml_syntax(self, data_dir, tmp_path):
        # Arrange
        config_file = tmp_path / "bad.toml"
        config_file.write_text("not valid ===")

        # Act
        result = runner.invoke(app, [
            "optimize", str(data_dir / "data.csv"),
            "--config", str(config_file),
            "--output", str(tmp_path / "out"),
        ])

        # Assert
        assert result.exit_code == 1


class TestGenerateConfigCommand:
    """Tests for the generate-config command."""

    def test_creates_both_config_files(self, tmp_path):
        # Act
        result = runner.invoke(app, ["generate-config", "--output-dir", str(tmp_path)])

        # Assert
        assert result.exit_code == 0
        assert (tmp_path / "fitness_config.toml").exists()
        assert (tmp_path / "ga_config.toml").exists()

    def test_generated_files_are_valid_flat_toml(self, tmp_path):
        # Act
        runner.invoke(app, ["generate-config", "--output-dir", str(tmp_path)])

        # Assert — both files parse without error and have no section headers
        for filename in ["fitness_config.toml", "ga_config.toml"]:
            content = (tmp_path / filename).read_text()
            with open(tmp_path / filename, "rb") as f:
                parsed = tomllib.load(f)
            assert len(parsed) > 0
            # No TOML table/section headers (lines starting with "[")
            assert not any(line.lstrip().startswith("[") for line in content.splitlines())
