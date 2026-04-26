"""Tests for CLI report file writers."""

import tomllib
from dataclasses import replace

from f1_optimizer.cli.reports import write_config_toml, write_schedule_md, write_summary_md
from f1_optimizer.core.ga.ga import GAConfig


class TestWriteConfigToml:
    """Tests for write_config_toml."""

    def test_creates_file(self, tmp_path):
        # Act
        write_config_toml(GAConfig(), tmp_path / "config.toml")

        # Assert
        assert (tmp_path / "config.toml").exists()

    def test_file_has_no_section_headers(self, tmp_path):
        # Act
        write_config_toml(GAConfig(), tmp_path / "config.toml")

        # Assert
        content = (tmp_path / "config.toml").read_text()
        assert "[" not in content

    def test_file_round_trips_through_tomllib(self, tmp_path):
        # Arrange
        config = GAConfig(population_size=150, crossover_type="pmx")
        output = tmp_path / "config.toml"

        # Act
        write_config_toml(config, output)

        # Assert
        with open(output, "rb") as f:
            parsed = tomllib.load(f)
        assert parsed["population_size"] == 150
        assert parsed["crossover_type"] == "pmx"


class TestWriteScheduleMd:
    """Tests for write_schedule_md."""

    def test_creates_file(self, tmp_path, sample_ga_report, track_db, fitness_evaluator):
        # Act
        write_schedule_md(sample_ga_report, track_db, GAConfig(), fitness_evaluator, tmp_path / "schedule.md")

        # Assert
        assert (tmp_path / "schedule.md").exists()

    def test_contains_total_fitness_score(self, tmp_path, sample_ga_report, track_db, fitness_evaluator):
        # Act
        write_schedule_md(sample_ga_report, track_db, GAConfig(), fitness_evaluator, tmp_path / "schedule.md")

        # Assert
        content = (tmp_path / "schedule.md").read_text()
        assert "50,000.00" in content

    def test_contains_total_distance_in_km(self, tmp_path, sample_ga_report, track_db, fitness_evaluator):
        # Act
        write_schedule_md(sample_ga_report, track_db, GAConfig(), fitness_evaluator, tmp_path / "schedule.md")

        # Assert
        content = (tmp_path / "schedule.md").read_text()
        assert "50,000 km" in content

    def test_contains_race_schedule_section_with_rows(self, tmp_path, sample_ga_report, track_db, fitness_evaluator):
        # Act
        write_schedule_md(sample_ga_report, track_db, GAConfig(), fitness_evaluator, tmp_path / "schedule.md")

        # Assert
        content = (tmp_path / "schedule.md").read_text()
        assert "## Race Schedule" in content
        assert "| 1 |" in content


class TestWriteSummaryMd:
    """Tests for write_summary_md."""

    def test_creates_file(self, tmp_path, sample_ga_report):
        # Act
        write_summary_md([("config_1", GAConfig(), sample_ga_report)], tmp_path / "summary.md")

        # Assert
        assert (tmp_path / "summary.md").exists()

    def test_contains_markdown_table_headers(self, tmp_path, sample_ga_report):
        # Act
        write_summary_md([("config_1", GAConfig(), sample_ga_report)], tmp_path / "summary.md")

        # Assert
        content = (tmp_path / "summary.md").read_text()
        assert "| Rank |" in content
        assert "| Best Fitness |" in content

    def test_ranks_configs_ascending_by_fitness(self, tmp_path, sample_ga_report):
        # Arrange — second report with higher (worse) fitness, all other fields unchanged
        worse_report = replace(sample_ga_report, best_fitness=99999.0)
        results = [
            ("config_worse", GAConfig(), worse_report),
            ("config_best", GAConfig(), sample_ga_report),
        ]

        # Act
        write_summary_md(results, tmp_path / "summary.md", runs=1)

        # Assert — best config appears before worse config in the file
        content = (tmp_path / "summary.md").read_text()
        assert content.index("config_best") < content.index("config_worse")

    def test_names_best_configuration_at_bottom(self, tmp_path, sample_ga_report):
        # Act
        write_summary_md(
            [("my_config", GAConfig(), sample_ga_report)], tmp_path / "summary.md"
        )

        # Assert
        content = (tmp_path / "summary.md").read_text()
        assert "Best configuration" in content
        assert "my_config" in content
