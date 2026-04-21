"""CLI for F1 Calendar GA Optimizer."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from f1_optimizer.cli.config import (
    ConfigLoadError,
    create_example_fitness_config,
    create_example_ga_config,
    expand_ga_configs,
    load_fitness_config,
    load_ga_config,
)
from f1_optimizer.cli.runner import run_multi_config, run_single_config
from f1_optimizer.core.evaluation.distance_matrix import DistanceMatrix
from f1_optimizer.core.evaluation.fitness import FitnessEvaluator
from f1_optimizer.data_access.loader import CsvTrackLoader, CsvTrackLoaderError

app = typer.Typer(help="F1 Calendar Genetic Algorithm Optimizer", add_completion=False)
console = Console()


@app.command()
def optimize(
    data_file: Path = typer.Argument(..., help="Path to CSV file with F1 track data", exists=True),
    fitness_config: Path | None = typer.Option(
        None, "--fitness-config", "-f", help="Fitness config TOML (uses defaults if omitted)"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="GA config TOML (uses defaults if omitted)"
    ),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory"),
    multi_config: bool = typer.Option(False, "--multi-config", help="Run all parameter permutations"),
    runs: int = typer.Option(1, "--runs", help="Repeat each config N times, report best", min=1),
) -> None:
    """Optimize F1 calendar schedule using Genetic Algorithm."""
    try:
        start = time.monotonic()

        f_config = load_fitness_config(fitness_config)
        tracks = CsvTrackLoader(data_file).load()
        fitness_evaluator = FitnessEvaluator(DistanceMatrix(tracks), f_config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            if multi_config:
                configs = expand_ga_configs(config)
                task = progress.add_task(f"Config 1/{len(configs)}...", total=None)

                def on_start(name: str, idx: int, total: int) -> None:
                    progress.update(task, description=f"Config {idx + 1}/{total}: {name}")

                run_multi_config(
                    configs,
                    fitness_evaluator,
                    tracks,
                    output,
                    runs=runs,
                    on_config_start=on_start,
                )
            else:
                ga_config = load_ga_config(config)
                progress.add_task("Evolving...", total=None)
                run_single_config(ga_config, fitness_evaluator, tracks, output, runs=runs)

        elapsed = time.monotonic() - start
        console.print(f"Done in {elapsed:.1f}s → {output}/")

    except ConfigLoadError as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=1)
    except CsvTrackLoaderError as e:
        console.print(f"[red]Data error:[/red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def generate_config(
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", help="Directory to write config files"
    ),
) -> None:
    """Generate example fitness_config.toml and ga_config.toml."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in [
            ("fitness_config.toml", create_example_fitness_config()),
            ("ga_config.toml", create_example_ga_config()),
        ]:
            path = output_dir / filename
            if path.exists():
                if not typer.confirm(f"{path} already exists. Overwrite?"):
                    continue
            path.write_text(content)
            console.print(f"[green]Created:[/green] {path}")

        console.print(
            f"\nUsage: f1-optimizer optimize data.csv"
            f" --fitness-config {output_dir}/fitness_config.toml"
            f" --config {output_dir}/ga_config.toml"
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
