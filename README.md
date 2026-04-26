# 🏎️ F1 Calendar GA Optimizer

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Managed by uv](https://img.shields.io/badge/managed_by-uv-purple)](https://github.com/astral-sh/uv)

A Genetic Algorithm engine that optimizes the Formula 1 race calendar — minimizing total travel distance while respecting real-world constraints around weather, logistics, and human fatigue.

## Architecture

- **`core/`** — GA engine and fitness evaluation
  - **`core/ga/`** — evolutionary loop ([ga.py](src/f1_optimizer/core/ga/ga.py)), tournament [selection](src/f1_optimizer/core/ga/selection.py), [crossover](src/f1_optimizer/core/ga/crossover.py) operators (OX1, PMX), and [mutation](src/f1_optimizer/core/ga/mutation.py) operators (swap, inversion)
  - **`core/evaluation/`** — [fitness](src/f1_optimizer/core/evaluation/fitness.py) aggregation, [hard constraints](src/f1_optimizer/core/evaluation/hard_constraints.py), [soft constraints](src/f1_optimizer/core/evaluation/soft_constraints.py), and the precomputed [distance matrix](src/f1_optimizer/core/evaluation/distance_matrix.py)
  - **`core/models.py`** — Pydantic domain models (`Track`, `Schedule`, configs)
- **`data_access/`** — CSV loading and validation with Pydantic
- **`cli/`** — Typer-based CLI with single-run, multi-config, and multi-run modes
- **`visualize/`** — Seaborn fitness convergence plots

The optimization problem (decision variables, hard/soft constraints, and the fitness function) is formally modeled for the 2026 F1 season in [constraints.md](constraints.md).

## Tech Stack

- **Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Build System**: `hatchling`
- **Testing**: `pytest`
- **Data Validation**: `pydantic`
- **CLI**: `typer` + `rich`
- **Plots**: `seaborn` + `matplotlib`

## Quickstart

```bash
git clone https://github.com/NoratanS/f1-ga-optimizer.git
cd f1-ga-optimizer
uv run pytest                              # run tests
uv run f1-optimizer --help                 # explore CLI
```

## CLI Usage

### Single Run

Run the GA once with default configuration:

```bash
f1-optimizer optimize data/data.csv
```

With custom configs:

```bash
f1-optimizer optimize data/data.csv \
  --fitness-config fitness_config.toml \
  --config ga_config.toml \
  --output results/
```

**Options:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--fitness-config` | `-f` | defaults | Fitness penalty config TOML |
| `--config` | `-c` | defaults | GA parameters config TOML |
| `--output` | `-o` | `results/` | Output directory |
| `--multi-config` | | off | Expand list values into all permutations |
| `--runs` | | `1` | Repeat each config N times, keep best result |

### Multi-Config Run

Explore parameter combinations by specifying lists in the GA config:

```toml
# ga_config.toml
population_size = [50, 100, 200]
generations = 1000
crossover_probability = [0.7, 0.8, 0.9]
mutation_probability = 0.1
crossover_type = "ox1"
mutation_type = "swap"
tournament_size = 5
stagnation_threshold = 50
elite_size = 2
```

```bash
f1-optimizer optimize data/data.csv --config ga_config.toml --multi-config
```

This generates 3×3 = 9 configurations, runs each, and writes a ranked `summary.md`.

### Multi-Run

Repeat a config N times and report the best result (useful for assessing variance):

```bash
# Single config, 5 runs
f1-optimizer optimize data/data.csv --config ga_config.toml --runs 5

# Multi-config, 3 runs per configuration
f1-optimizer optimize data/data.csv --config ga_config.toml --multi-config --runs 3
```

### Recommended Workflow

The GA has many tunable parameters and stochastic outcomes, so a two-stage approach works best:

**Stage 1 — Broad exploration.** Run a wide multi-config sweep with many parameter permutations and a small number of runs per config (e.g. `--runs 5`). Use shorter generation counts to keep the sweep cheap. Inspect `summary.md` to identify the parameter regions that consistently produce good fitness.

```bash
f1-optimizer optimize data/data.csv \
  --config ga_config.toml \
  --multi-config \
  --runs 2 \
  --output results/sweep/
```

**Stage 2 — Focused refinement.** Narrow the config to the top-performing region from stage 1, then re-run with more repetitions and longer generations to reduce variance and let the GA converge.

```bash
f1-optimizer optimize data/data.csv \
  --config ga_config_refined.toml \
  --multi-config \
  --runs 20 \
  --output results/refined/
```

This avoids burning compute on parameter combinations that are unlikely to win, while still giving the final candidates enough runs to distinguish signal from noise.

### Generate Config Templates

```bash
f1-optimizer generate-config --output-dir .
```

Creates `fitness_config.toml` and `ga_config.toml` with all available options.

## Output Structure

### Single Run

```
results/
├── config.toml       # GA parameters used
├── schedule.md       # Best schedule + fitness breakdown
└── plot.png          # Fitness convergence chart
```

### Multi-Config Run

```
results/
├── config_1_ps50_cp0.7/
│   ├── config.toml
│   ├── schedule.md
│   └── plot.png
├── config_2_ps50_cp0.8/
│   └── ...
└── summary.md        # All configs ranked by fitness
```

## Data Overview

The optimizer reads a single CSV ([data/data.csv](data/data.csv)) describing the 24 tracks of the **2026 F1 calendar** — this project was built specifically with the 2026 season in mind.

| Column | Description |
|--------|-------------|
| `ID` | Unique track identifier (1–24) |
| `City` | Host city |
| `Country` | Host country |
| `Track_Name` | Official circuit name |
| `Region` | Geographic region (e.g. `Europe`, `Asia`, `Americas`, `Middle East`, `Oceania`) — used by the region-bounce and flyaway logistics penalties |
| `Latitude` | Decimal degrees, used for Haversine distance |
| `Longitude` | Decimal degrees, used for Haversine distance |
| `UTC_Offset` | Standard timezone offset (hours), used for the jet-lag penalty |
| `Temperatures` | List of 40 weekly mean temperatures (°C), one per season week |

**Season window.** The 40-element `Temperatures` array maps to the season slots: index `0` is the **first week of March**, index `39` is the **first week of December**, covering 40 weeks total.

**Temperature source.** Weekly means were computed from the [Open-Meteo](https://open-meteo.com/) historical weather API, averaged over the **last 5 years** at each track's coordinates. They feed both the hard weather constraint (HC3) and the environmental soft penalty ($Z_{env}$).

## Configuration

### Fitness Config (`fitness_config.toml`)

Controls penalty coefficients and optimization weights. All runs share the same fitness config.

```toml
# Penalty coefficients
K_dist = 1.0          # per km of travel
K_env = 1000.0        # per °C temperature deviation
K_utc = 500.0         # per timezone hour jump
K_double = 5000.0     # double-header week
K_triple = 15000.0    # triple-header week
K_bounce = 10000.0    # region change between consecutive races
K_flyaway = 50000.0   # intercontinental back-to-back
M = 1000000.0         # hard constraint violation multiplier

# Fitness weights (must sum to 1.0)
w_dist = 0.25
w_env = 0.25
w_log = 0.25
w_hum = 0.25

# Temperature bounds (°C)
T_hard_min = 5.0
T_hard_max = 40.0
T_opt_min = 18.0
T_opt_max = 28.0

summer_break_window = [17, 26]  # 0-indexed week range
ramadan_window = [0, 3]
```

### GA Config (`ga_config.toml`)

Controls the evolutionary algorithm. Supports list values for multi-config mode.

```toml
population_size = 100
generations = 1000
crossover_probability = 0.8
mutation_probability = 0.1
crossover_type = "ox1"   # "ox1" or "pmx"
mutation_type = "swap"   # "swap" or "inversion"
tournament_size = 5
stagnation_threshold = 50
elite_size = 2
```