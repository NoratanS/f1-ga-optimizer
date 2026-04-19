"""CLI for F1 Calendar GA Optimizer using Typer and Rich."""

from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from f1_optimizer.data_access.loader import CsvTrackLoader, CsvTrackLoaderError
from f1_optimizer.core.evaluation.fitness import FitnessEvaluator, DistanceMatrix
from f1_optimizer.core.ga.ga import GeneticAlgorithm
from f1_optimizer.cli.config import load_config_file, ConfigLoadError, create_example_config
from f1_optimizer.cli.schedule import format_schedule, format_schedule_display

app = typer.Typer(help="F1 Calendar Genetic Algorithm Optimizer")
console = Console()


@app.command()
def optimize(
    data_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with F1 track data",
        exists=True,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to TOML config file (uses defaults if not provided)",
        exists=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Optimize F1 calendar schedule using Genetic Algorithm.
    
    Takes a CSV file with track data and runs the GA to find an optimized
    schedule that minimizes travel distance between races.
    """
    try:
        console.print("[bold cyan]🏎️  F1 Calendar GA Optimizer[/bold cyan]")
        console.print()
        
        # Load configuration
        console.print("[yellow]Loading configuration...[/yellow]")
        fitness_config, ga_config = load_config_file(config)
        if config:
            console.print(f"[green]✓ Loaded config from {config}[/green]")
        else:
            console.print("[green]✓ Using default configuration[/green]")
        console.print()
        
        # Load track data
        console.print(f"[yellow]Loading track data from:[/yellow] {data_file}")
        loader = CsvTrackLoader(data_file)
        tracks = loader.load()
        console.print(f"[green]✓ Loaded {len(tracks)} tracks[/green]")
        console.print()
        
        # Initialize fitness evaluator
        distance_matrix = DistanceMatrix(tracks)
        fitness_evaluator = FitnessEvaluator(distance_matrix, fitness_config)
        
        # Display GA configuration
        console.print("[yellow]GA Configuration:[/yellow]")
        config_table = Table(show_header=False, box=None)
        config_table.add_row("Population Size", str(ga_config.population_size))
        config_table.add_row("Generations", str(ga_config.generations))
        config_table.add_row("Crossover Type", ga_config.crossover_type)
        config_table.add_row("Mutation Type", ga_config.mutation_type)
        config_table.add_row("Stagnation Threshold", str(ga_config.stagnation_threshold))
        console.print(config_table)
        console.print()
        
        # Run GA with progress bar
        console.print("[yellow]Running genetic algorithm...[/yellow]")
        ga = GeneticAlgorithm(ga_config, fitness_evaluator)
        
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Evolving population...", total=ga_config.generations)
            
            # Run evolution
            report = ga.evolve()
            progress.update(task, completed=report.generations_run)
        
        console.print()
        
        # Display results
        display_results(report, tracks, verbose)
        
        console.print("[green]✓ Optimization complete![/green]")
        
    except ConfigLoadError as e:
        console.print(f"[red]✗ Config Error:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)
    except CsvTrackLoaderError as e:
        console.print(f"[red]✗ Data Loading Error:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]✗ Configuration Error:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]✗ Unexpected Error:[/red] {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def generate_config(
    output: Path = typer.Option(
        Path("config.toml"),
        "--output",
        "-o",
        help="Output path for the generated config file",
    ),
) -> None:
    """Generate an example configuration file.
    
    Creates a TOML config file with all available options and descriptions.
    You can customize this file and pass it with --config flag.
    """
    try:
        if output.exists():
            overwrite = typer.confirm(f"{output} already exists. Overwrite?")
            if not overwrite:
                console.print("[yellow]Aborted[/yellow]")
                raise typer.Exit(code=0)
        
        output.write_text(create_example_config())
        console.print(f"[green]✓ Generated example config: {output}[/green]")
        console.print(f"[cyan]Edit it and use: f1-optimizer optimize data/data.csv --config {output}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]✗ Error creating config:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)


def display_results(report, tracks, verbose: bool) -> None:
    """Display optimization results using Rich formatting."""
    # Best fitness panel
    best_fitness_panel = Panel(
        f"[bold cyan]{report.best_fitness:,.2f}[/bold cyan]",
        title="[bold]Best Fitness Score[/bold]",
        expand=False,
    )
    console.print(best_fitness_panel)
    console.print()
    
    # Fitness breakdown from report
    breakdown = report.fitness_breakdown
    console.print("[bold]Fitness Breakdown[/bold]")
    breakdown_table = Table(show_header=False, box=None)
    breakdown_table.add_row("Distance Penalty", f"{breakdown.distance_penalty:,.2f}")
    breakdown_table.add_row("Environment Penalty", f"{breakdown.environment_penalty:,.2f}")
    breakdown_table.add_row("Jetlag Penalty", f"{breakdown.jetlag_penalty:,.2f}")
    breakdown_table.add_row("Headers Penalty", f"{breakdown.headers_penalty:,.2f}")
    breakdown_table.add_row("Logistics Penalty", f"{breakdown.logistics_penalty:,.2f}")
    breakdown_table.add_row("Soft Total", f"[yellow]{breakdown.soft_penalties_total:,.2f}[/yellow]")
    breakdown_table.add_row("Hard Violations Penalty", f"{breakdown.hard_violations_penalty:,.2f}")
    breakdown_table.add_row("Total", f"[cyan]{breakdown.total_fitness:,.2f}[/cyan]")
    console.print(breakdown_table)
    console.print()
    
    # Statistics table
    stats_table = Table(title="[bold]Statistics[/bold]", show_header=False, box=None)
    stats_table.add_row("Generations Run", str(report.generations_run))
    stats_table.add_row("Final Average Fitness", f"{report.avg_fitness_per_gen[-1]:,.2f}")
    stats_table.add_row("Improvement", f"{(report.best_fitness_per_gen[0] - report.best_fitness):,.2f}")
    console.print(stats_table)
    console.print()
    
    # Best schedule
    schedule = format_schedule(report.best_individual.chromosome, tracks)
    console.print("[bold]Optimized Schedule[/bold]")
    schedule_display = format_schedule_display(schedule, compact=True)
    console.print(schedule_display)
    console.print()
    
    if verbose:
        # Detailed verbose output
        console.print("[bold]Full Schedule with Breaks[/bold]")
        full_schedule_display = format_schedule_display(schedule, compact=False)
        console.print(full_schedule_display)
        console.print()
        
        console.print("[bold]Fitness History (first 10 gens):[/bold]")
        history_table = Table(show_header=True)
        history_table.add_column("Generation", style="cyan")
        history_table.add_column("Best Fitness", style="green")
        history_table.add_column("Avg Fitness", style="yellow")
        
        for i, (best, avg) in enumerate(
            zip(
                report.best_fitness_per_gen[:10],
                report.avg_fitness_per_gen[:10],
            )
        ):
            history_table.add_row(str(i), f"{best:,.2f}", f"{avg:,.2f}")
        console.print(history_table)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
