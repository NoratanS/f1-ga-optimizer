# 🏎️ F1 Calendar GA Optimizer

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Managed by uv](https://img.shields.io/badge/managed_by-uv-purple)](https://github.com/astral-sh/uv)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A highly optimized Genetic Algorithm (GA) engine designed to schedule and optimize the Formula 1 race calendar. The algorithm aims to minimize travel distance between Grand Prix locations while adhering to real-world logistical constraints and weather conditions.

## 🧬 Architecture

This project is built with modern Python standards, focusing on a clean, object-oriented, and test-driven design.

- **`core/`**: The heart of the Genetic Algorithm (Genotype, Crossover, Mutation, Selection).
- **`data_access/`**: Robust data parsing and validation using `Pydantic`.
- **`visualize/`**: (Coming soon) Map generation and route plotting.

## 🛠️ Tech Stack

- **Dependency Management & Environments**: [uv](https://github.com/astral-sh/uv)
- **Build System**: `hatchling` (Editable installs)
- **Testing**: `pytest` (Strict TDD approach)
- **Data Validation**: `pydantic`
- **CI/CD**: Automated Code Review powered by Anthropic's Claude 3.5 Sonnet via Qodo Merge.

## 🚀 Quickstart

This project uses `uv` for lightning-fast, reproducible environments. You don't need to manually create virtual environments.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NoratanS/f1-ga-optimizer.git](https://github.com/NoratanS/f1-ga-optimizer.git)
   cd f1-ga-optimizer
   ```

2. **Run tests (uv will automatically sync the environment):**
   ```bash
   uv run pytest
   ```

## 📈 Development Roadmap
- [x] Epic 1: Domain Models (Genotype) & Core Architecture
- [ ] Epic 2: Data Access & Pydantic Validation (CSV Loading)
- [ ] Epic 3: GA Operators (Fitness, Crossover, Mutation)
- [ ] Epic 4: Evolutionary Loop & CLI Integration
- [ ] Epic 5: Visualization