# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zppy_interfaces is a Python package providing extra functionality for E3SM climate model analysis. It processes log files, generates time series plots, and runs PCMDI diagnostics. The package is designed to be called by zppy or used standalone for climate model analysis.

## Development Environment Setup

Set up the development environment using conda:
```bash
conda clean --all --y
conda env create -f conda/dev.yml -n zppy-interfaces-dev
conda activate zppy-interfaces-dev
pip install .
pre-commit install
```

## Common Development Commands

### Installation and Setup
- `pip install .` - Install package in development mode
- `pip install -e .[testing]` - Install with testing dependencies
- `pip install -e .[qa]` - Install with quality assurance tools

### Code Quality and Testing
- `pytest` - Run all tests
- `pytest tests/unit/budget_analysis/` - Run specific module tests
- `pytest tests/unit/budget_analysis/test_atm_parser.py` - Run single test file
- `pre-commit run --all-files` - Run all pre-commit hooks
- `black .` - Format code with Black
- `isort .` - Sort imports
- `flake8` - Check code style
- `mypy zppy_interfaces/` - Type checking

### CLI Tools Testing
Test the main CLI applications:
- `zi-budget-analysis --help` - Budget analysis tool
- `zi-global-time-series --help` - Global time series plots
- `zi-pcmdi-link-observation --help` - PCMDI observation linking
- `zi-pcmdi-mean-climate --help` - PCMDI mean climate diagnostics
- `zi-pcmdi-variability-modes --help` - PCMDI variability modes
- `zi-pcmdi-enso --help` - PCMDI ENSO diagnostics
- `zi-pcmdi-synthetic-plots --help` - PCMDI synthetic plots

## Architecture Overview

### Main Components

**budget_analysis/** - E3SM water and energy budget analysis
- `__main__.py` - CLI entry point with legacy and whole-model modes
- `parser.py` - Core budget parsing logic for coupler logs
- `ingestion/` - Component-specific log parsers (atm, ocn, ice, lnd, cpl)
- `plotting.py` - HTML plot generation for legacy mode
- `viz.py` - Visualization for whole-model mode
- `checks.py` - Budget conservation checks
- `normalization.py` - Data normalization utilities

**global_time_series/** - Global time series plot generation
- `__main__.py` - CLI with viewer vs PDF output modes
- `coupled_global/` - Core time series generation logic
- `create_ocean_ts.py` - Ocean-specific time series processing
- `utils.py` - Parameter handling utilities

**pcmdi_diags/** - PCMDI diagnostics suite
- Multiple CLI tools for different diagnostic types
- `viewer.py` - HTML viewer generation
- `synthetic_plots/` - Synthetic plot utilities

**multi_utils/** - Shared utilities
- `logger.py` - Logging setup for child processes
- `viewer.py` - Common viewer functionality

### Data Flow Patterns

**Budget Analysis (whole-model mode):**
1. Ingest: Parse multiple log file types (cpl, atm, ocn, ice, lnd)
2. Normalize: Standardize data formats and units
3. Check: Run conservation checks and compute residuals
4. Visualize: Generate HTML reports with interactive plots

**Global Time Series:**
1. Ocean processing: Extract time series from MPAS-Analysis results (optional)
2. Coupled analysis: Generate regional and global plots
3. Output: HTML viewer (interactive) or PDF (static) based on make_viewer setting

### Key Configuration Files

- `pyproject.toml` - Package configuration, dependencies, CLI entry points
- `conda/dev.yml` - Development environment specification
- `.pre-commit-config.yaml` - Code quality hooks (black, isort, flake8, mypy)
- `.flake8` - Flake8 configuration (line length 119, specific ignores)

### Testing Strategy

- Unit tests in `tests/unit/` organized by module
- Example scripts in `examples/` showing realistic usage
- Integration with pre-commit hooks for quality assurance
- pytest with coverage reporting capabilities

## Important Notes

- The package handles both compressed (.gz) and uncompressed log files
- Budget analysis supports both "legacy" (coupler-only) and "whole-model" modes
- Time series generation can produce either interactive HTML viewers or static PDFs
- All CLI tools use argparse with comprehensive help documentation
- The codebase follows strict code quality standards with Black formatting and comprehensive linting