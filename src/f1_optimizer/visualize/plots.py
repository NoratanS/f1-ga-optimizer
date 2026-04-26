"""Seaborn-based visualizations for GA optimization results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from f1_optimizer.core.ga.ga import GAReport


_SOFT_COMPONENTS = [
    ("distance_penalty", "Distance", "#4C72B0"),
    ("environment_penalty", "Environment", "#DD8452"),
    ("jetlag_penalty", "Jetlag", "#55A467"),
    ("headers_penalty", "Headers", "#C44E52"),
    ("logistics_penalty", "Logistics", "#8172B3"),
]


def plot_fitness_curve(report: GAReport, output_path: Path) -> None:
    """Save a stacked component-convergence chart for the best individual.

    Stacked area shows component composition over time; the cumulative top
    of the stack equals total fitness, redrawn as a black line for emphasis.
    Log y-axis handles the wide dynamic range across components, and the
    feasibility transition (where M·hard_violations drops to 0) is marked
    with a vertical line.
    """
    sns.set_style("whitegrid")

    breakdowns = report.best_breakdown_per_gen
    generations = list(range(len(breakdowns)))

    component_series = [
        [getattr(bd, attr) for bd in breakdowns] for attr, _, _ in _SOFT_COMPONENTS
    ]
    labels = [label for _, label, _ in _SOFT_COMPONENTS]
    colors = [color for _, _, color in _SOFT_COMPONENTS]
    totals = [bd.total_fitness for bd in breakdowns]

    feasible_start = next(
        (i for i, bd in enumerate(breakdowns) if bd.hard_violations_penalty == 0),
        None,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        generations,
        component_series,
        labels=labels,
        colors=colors,
        alpha=0.75,
    )
    ax.plot(generations, totals, color="black", linewidth=1.5, label="Total Fitness")

    if feasible_start is not None:
        ax.axvline(
            feasible_start,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Feasibility (gen {feasible_start})",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (log scale)")
    ax.set_title("F1 Calendar Optimization — Component Convergence")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    sns.despine(fig=fig)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
