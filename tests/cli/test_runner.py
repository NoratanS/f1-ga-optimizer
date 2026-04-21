"""Tests for CLI runner orchestration."""

from f1_optimizer.cli.runner import run_multi_config, run_single_config
from f1_optimizer.core.ga.ga import GAReport


class TestRunSingleConfig:
    """Tests for run_single_config."""

    def test_creates_required_output_files(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Act
        run_single_config(minimal_ga_config, fitness_evaluator, track_db, tmp_path)

        # Assert
        assert (tmp_path / "config.toml").exists()
        assert (tmp_path / "schedule.md").exists()
        assert (tmp_path / "plot.png").exists()

    def test_returns_ga_report_with_breakdown(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Act
        report = run_single_config(minimal_ga_config, fitness_evaluator, track_db, tmp_path)

        # Assert
        assert isinstance(report, GAReport)
        assert report.best_fitness is not None
        assert report.fitness_breakdown is not None

    def test_multi_run_returns_a_valid_report(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Act — 3 runs; result is the best of them
        report = run_single_config(minimal_ga_config, fitness_evaluator, track_db, tmp_path, runs=3)

        # Assert
        assert isinstance(report, GAReport)
        assert report.best_fitness > 0


class TestRunMultiConfig:
    """Tests for run_multi_config."""

    def test_creates_per_config_directories(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Arrange
        configs = [("config_a", minimal_ga_config), ("config_b", minimal_ga_config)]

        # Act
        run_multi_config(configs, fitness_evaluator, track_db, tmp_path)

        # Assert
        assert (tmp_path / "config_a").is_dir()
        assert (tmp_path / "config_b").is_dir()

    def test_creates_summary_md(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Act
        run_multi_config([("config_1", minimal_ga_config)], fitness_evaluator, track_db, tmp_path)

        # Assert
        assert (tmp_path / "summary.md").exists()

    def test_returns_one_result_tuple_per_config(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Arrange
        configs = [("config_a", minimal_ga_config), ("config_b", minimal_ga_config)]

        # Act
        results = run_multi_config(configs, fitness_evaluator, track_db, tmp_path)

        # Assert
        assert len(results) == 2
        for name, ga_config, report in results:
            assert isinstance(name, str)
            assert isinstance(report, GAReport)

    def test_on_config_start_callback_receives_correct_args(self, tmp_path, minimal_ga_config, fitness_evaluator, track_db):
        # Arrange
        configs = [("config_1", minimal_ga_config), ("config_2", minimal_ga_config)]
        calls: list[tuple] = []

        # Act
        run_multi_config(
            configs,
            fitness_evaluator,
            track_db,
            tmp_path,
            on_config_start=lambda name, idx, total: calls.append((name, idx, total)),
        )

        # Assert
        assert calls == [("config_1", 0, 2), ("config_2", 1, 2)]
