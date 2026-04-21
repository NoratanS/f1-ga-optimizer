"""Seaborn-based visualizations for GA optimization results."""
from __future__ import annotations

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from f1_optimizer.core.ga.ga import GAReport


def plot_fitness_curve(report: GAReport, output_path: Path) -> None:
    """Save a seaborn line chart of best and average fitness per generation."""

    generations = list(range(len(report.best_fitness_per_gen)))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=generations, y=report.best_fitness_per_gen, ax=ax, label="Best Fitness", color="steelblue")
    sns.lineplot(x=generations, y=report.avg_fitness_per_gen, ax=ax, label="Avg Fitness", color="darkorange", alpha=0.7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.set_title("F1 Calendar Optimization — Fitness Convergence")
    ax.legend()
    sns.despine(fig=fig)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
