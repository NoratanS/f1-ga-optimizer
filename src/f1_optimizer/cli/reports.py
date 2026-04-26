"""File report writers for GA optimization results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from f1_optimizer.core.evaluation.fitness import FitnessEvaluator
from f1_optimizer.core.ga.ga import GAConfig, GAReport
from f1_optimizer.core.models import Individual
from f1_optimizer.data_access.schemas import TrackSchema
from f1_optimizer.cli.schedule import format_schedule
from f1_optimizer.cli.baselines import REAL_2026_CALENDAR


def write_config_toml(ga_config: GAConfig, path: Path) -> None:
    """Write GAConfig as a flat TOML file."""
    lines = []
    for field, value in ga_config.model_dump().items():
        if isinstance(value, str):
            lines.append(f'{field} = "{value}"')
        elif value is None:
            lines.append(f"{field} = null")
        else:
            lines.append(f"{field} = {value}")
    path.write_text("\n".join(lines) + "\n")


def _improvement(ga: float, real: float) -> str:
    if real == 0:
        return "-"
    return f"{(ga - real) / real * 100:+.1f}%"


def write_schedule_md(
    report: GAReport,
    tracks: dict[int, TrackSchema],
    ga_config: GAConfig,
    fitness_evaluator: FitnessEvaluator,
    path: Path,
) -> None:
    """Write schedule.md with fitness breakdown and race table."""
    fitness_config = fitness_evaluator.config
    bd = report.fitness_breakdown
    schedule = format_schedule(report.best_individual.chromosome, tracks)
    races = [e for e in schedule if not e.is_break]
    dist_divisor = fitness_config.w_dist * fitness_config.K_dist
    total_distance_km = bd.distance_penalty / dist_divisor if dist_divisor else None

    real_individual = Individual(chromosome=REAL_2026_CALENDAR)
    real_bd = fitness_evaluator.get_fitness_breakdown(real_individual)
    real_distance_km = real_bd.distance_penalty / dist_divisor if dist_divisor else None

    lines = [
        "# F1 Optimized Calendar",
        "",
        "## GA Configuration",
        "",
        f"- Population Size: {ga_config.population_size}",
        f"- Generations Run: {report.generations_run}/{ga_config.generations}",
        f"- Crossover: {ga_config.crossover_type} (p={ga_config.crossover_probability})",
        f"- Mutation: {ga_config.mutation_type} (p={ga_config.mutation_probability})",
        f"- Total Distance: {f'{total_distance_km:,.0f} km' if total_distance_km is not None else 'N/A'}"
        f" (Real 2026: {f'{real_distance_km:,.0f} km' if real_distance_km is not None else 'N/A'})",
        "",
        "## Fitness Results",
        "",
        f"**Total Fitness Score:** {report.best_fitness:,.2f}",
        "",
        "| Component | GA Score | Real 2026 | Delta | Improvement |",
        "|-----------|--------:|----------:|------:|------------:|",
        f"| Distance | {bd.distance_penalty:,.2f} | {real_bd.distance_penalty:,.2f} | {bd.distance_penalty - real_bd.distance_penalty:+,.2f} | {_improvement(bd.distance_penalty, real_bd.distance_penalty)} |",
        f"| Environment | {bd.environment_penalty:,.2f} | {real_bd.environment_penalty:,.2f} | {bd.environment_penalty - real_bd.environment_penalty:+,.2f} | {_improvement(bd.environment_penalty, real_bd.environment_penalty)} |",
        f"| Jetlag | {bd.jetlag_penalty:,.2f} | {real_bd.jetlag_penalty:,.2f} | {bd.jetlag_penalty - real_bd.jetlag_penalty:+,.2f} | {_improvement(bd.jetlag_penalty, real_bd.jetlag_penalty)} |",
        f"| Headers | {bd.headers_penalty:,.2f} | {real_bd.headers_penalty:,.2f} | {bd.headers_penalty - real_bd.headers_penalty:+,.2f} | {_improvement(bd.headers_penalty, real_bd.headers_penalty)} |",
        f"| Logistics | {bd.logistics_penalty:,.2f} | {real_bd.logistics_penalty:,.2f} | {bd.logistics_penalty - real_bd.logistics_penalty:+,.2f} | {_improvement(bd.logistics_penalty, real_bd.logistics_penalty)} |",
        f"| Hard Violations | {bd.hard_violations_penalty:,.2f} | {real_bd.hard_violations_penalty:,.2f} | {bd.hard_violations_penalty - real_bd.hard_violations_penalty:+,.2f} | {_improvement(bd.hard_violations_penalty, real_bd.hard_violations_penalty)} |",
        f"| **Total** | **{bd.total_fitness:,.2f}** | **{real_bd.total_fitness:,.2f}** | **{bd.total_fitness - real_bd.total_fitness:+,.2f}** | **{_improvement(bd.total_fitness, real_bd.total_fitness)}** |",
        "",
        "## Race Schedule",
        "",
        "| # | Track | Location | Week |",
        "|---|-------|----------|-----:|",
    ]
    for i, entry in enumerate(races, start=1):
        lines.append(f"| {i} | {entry.track_name} | {entry.location} | {entry.week} |")

    path.write_text("\n".join(lines) + "\n")


def write_summary_md(
    results: list[tuple[str, GAConfig, GAReport]],
    path: Path,
    runs: int = 1,
) -> None:
    """Write summary.md ranking all multi-config results by fitness."""
    sorted_results = sorted(results, key=lambda r: r[2].best_fitness)

    lines = [
        "# F1 GA Optimizer — Multi-Config Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Configs: {len(results)}, Runs per config: {runs}",
        "",
        "| Rank | Config | Best Fitness | Gens Run |",
        "|-----:|--------|-------------:|---------:|",
    ]
    for rank, (name, ga_config, report) in enumerate(sorted_results, start=1):
        lines.append(
            f"| {rank} | {name} | {report.best_fitness:,.2f}"
            f" | {report.generations_run}/{ga_config.generations} |"
        )
    if sorted_results:
        lines += [
            "",
            f"**Best configuration:** {sorted_results[0][0]}  ",
            f"**Best fitness score:** {sorted_results[0][2].best_fitness:,.2f}",
        ]
    path.write_text("\n".join(lines) + "\n")
