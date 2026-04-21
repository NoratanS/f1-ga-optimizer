"""Run orchestration for single and multi-config GA runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from f1_optimizer.core.evaluation.fitness import FitnessEvaluator
from f1_optimizer.core.ga.ga import GAConfig, GAReport, GeneticAlgorithm
from f1_optimizer.data_access.schemas import TrackSchema
from f1_optimizer.cli.reports import write_config_toml, write_schedule_md, write_summary_md
from f1_optimizer.visualize.plots import plot_fitness_curve


def run_single_config(
    ga_config: GAConfig,
    fitness_evaluator: FitnessEvaluator,
    tracks: dict[int, TrackSchema],
    output_dir: Path,
    runs: int = 1,
) -> GAReport:
    """Run GA (possibly N times), write best result files to output_dir."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    best_report = GeneticAlgorithm(ga_config, fitness_evaluator).evolve()
    for _ in range(runs - 1):
        report = GeneticAlgorithm(ga_config, fitness_evaluator).evolve()
        if report.best_fitness < best_report.best_fitness:
            best_report = report

    output_dir.mkdir(parents=True, exist_ok=True)
    write_config_toml(ga_config, output_dir / "config.toml")
    write_schedule_md(best_report, tracks, ga_config, fitness_evaluator.config, output_dir / "schedule.md")
    plot_fitness_curve(best_report, output_dir / "plot.png")
    return best_report


def run_multi_config(
    configs: list[tuple[str, GAConfig]],
    fitness_evaluator: FitnessEvaluator,
    tracks: dict[int, TrackSchema],
    output_dir: Path,
    runs: int = 1,
    on_config_start: Callable[[str, int, int], None] | None = None,
    on_config_done: Callable[[str, GAReport], None] | None = None,
) -> list[tuple[str, GAConfig, GAReport]]:
    """Run GA for each config permutation, write per-config results and summary.md."""
    results: list[tuple[str, GAConfig, GAReport]] = []
    for i, (name, ga_config) in enumerate(configs):
        if on_config_start:
            on_config_start(name, i, len(configs))
        report = run_single_config(ga_config, fitness_evaluator, tracks, output_dir / name, runs)
        results.append((name, ga_config, report))
        if on_config_done:
            on_config_done(name, report)

    write_summary_md(results, output_dir / "summary.md", runs)
    return results
