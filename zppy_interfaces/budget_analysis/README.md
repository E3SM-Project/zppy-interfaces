# E3SM Budget Analysis Module

The `budget_analysis` module provides comprehensive water and energy budget analysis for E3SM climate model simulations. It analyzes coupler log files to extract budget data, performs conservation checks, and generates interactive visualizations.

## Overview

This module processes E3SM component log files (coupler, atmosphere, ocean, ice, land) to:
- Extract water, heat, and carbon budget terms
- Perform conservation checks across components
- Calculate residuals and identify budget imbalances
- Generate interactive HTML reports with time series plots
- Support both legacy (coupler-only) and whole-model analysis modes

## Modes of Operation

### Legacy Mode (Default)
- **Purpose**: Original coupler-only cumulative budget analysis
- **Input**: Coupler log files (`cpl.log.*.gz`)
- **Output**: Interactive HTML plots and ASCII summary tables
- **Use case**: Quick budget overview using only coupler data

### Whole-Model Mode
- **Purpose**: Comprehensive multi-component budget analysis with conservation checks
- **Input**: Log files from all available components (cpl, atm, ocn, ice, lnd)
- **Output**: Detailed HTML reports with conservation diagnostics
- **Use case**: Complete budget closure analysis across all model components

## CLI Usage

### Basic Legacy Analysis
```bash
zi-budget-analysis \
  --log_path /path/to/case/archive/logs \
  --start_year 114 \
  --end_year 206
```

### Whole-Model Analysis
```bash
zi-budget-analysis \
  --log_path /path/to/case/archive/logs \
  --start_year 114 \
  --end_year 150 \
  --mode whole-model \
  --budget_types water,heat \
  --frequency monthly \
  --output_dir ./results
```

### Command Line Options

- `--log_path` (required): Directory containing log files
- `--start_year` (required): Starting year for analysis
- `--end_year` (required): Ending year for analysis
- `--budget_types`: Comma-separated list (water,heat,carbon) [default: water,heat]
- `--mode`: Analysis mode (legacy,whole-model) [default: legacy]
- `--frequency`: Temporal frequency (monthly,annual) [default: annual]
- `--output_dir`: Output directory [default: current directory]
- `--output_html`: Generate HTML plots [default: True]

## Architecture

### Core Components

**Data Ingestion** (`ingestion/`)
- `base.py` - Abstract base parser class
- `cpl_parser.py` - Coupler log parser
- `atm_parser.py` - Atmosphere log parser
- `ocn_parser.py` - Ocean log parser
- `ice_parser.py` - Sea ice log parser
- `lnd_parser.py` - Land log parser

**Data Processing**
- `parser.py` - Legacy mode parsing logic
- `schema.py` - Standardized data table schema
- `normalization.py` - Unit conversion and data standardization

**Analysis**
- `checks.py` - Budget conservation checks and residual calculations
- `plotting.py` - Legacy mode HTML plot generation
- `viz.py` - Whole-model mode report generation

### Data Schema

The module uses a standardized "tidy" DataFrame schema with columns:
- `time`: Year (annual) or year + fractional month (monthly)
- `component`: Model component (atm, lnd, ocn, ice_nh, ice_sh, etc.)
- `quantity`: Budget type (water, heat, carbon)
- `term`: Specific flux or state variable name
- `value`: Numerical value in original units
- `units`: Original unit string
- `source`: Source log file type (cpl, atm, etc.)
- `period`: Temporal frequency (annual, monthly)
- `table_type`: Data type (flux, flux_integrated, state)

## Budget Conservation Checks

### Water Budget Checks
- **Component Fluxes**: Verify coupler component flux balances
- **Interface Matching**: Compare fluxes at component boundaries
- **Component Closure**: Check internal conservation within each component
- **Residual Analysis**: Calculate and track budget imbalances over time

### Heat Budget Checks
- **Component Fluxes**: Energy flux balance verification
- **Interface Matching**: Energy flux consistency across boundaries
- **Component Closure**: Internal energy conservation (limited for atmosphere and land)
- **Note**: Atmospheric and land energy checks are limited due to incomplete flux data in ATM and LND logs

### Carbon Budget Checks
- **Component Fluxes**: Carbon flux balance verification (basic implementation)

## Output Formats

### Legacy Mode Output
- Interactive HTML plots with time series of cumulative budgets
- ASCII summary tables written to stdout
- Plots saved as both PNG and HTML files

### Whole-Model Mode Output
- Comprehensive HTML reports for each budget type
- Interactive time series plots with residual analysis
- Component-by-component breakdown
- Landing page linking all generated reports
- Conservation check results with pass/fail indicators

## File Structure

```
budget_analysis/
├── __init__.py              # Module initialization
├── __main__.py              # CLI entry point
├── parser.py                # Legacy parsing logic
├── plotting.py              # Legacy HTML generation
├── schema.py                # Data schema definitions
├── normalization.py         # Data normalization utilities
├── checks.py                # Conservation check definitions
├── viz.py                   # Whole-model visualization
└── ingestion/              # Component-specific parsers
    ├── __init__.py
    ├── base.py             # Abstract parser base class
    ├── cpl_parser.py       # Coupler log parser
    ├── atm_parser.py       # Atmosphere parser
    ├── ocn_parser.py       # Ocean parser
    ├── ice_parser.py       # Sea ice parser
    └── lnd_parser.py       # Land parser
```

## Example Workflows

### Quick Budget Check (Legacy)
```python
from zppy_interfaces.budget_analysis import main
import sys

# Simulate CLI arguments
sys.argv = [
    'zi-budget-analysis',
    '--log_path', '/path/to/logs',
    '--start_year', '100',
    '--end_year', '150',
    '--budget_types', 'water'
]

main()
```

### Comprehensive Analysis (Whole-Model)
```python
from zppy_interfaces.budget_analysis import main
import sys

sys.argv = [
    'zi-budget-analysis',
    '--log_path', '/path/to/logs',
    '--start_year', '100',
    '--end_year', '150',
    '--mode', 'whole-model',
    '--budget_types', 'water,heat',
    '--frequency', 'monthly',
    '--output_dir', './budget_results'
]

main()
```

## Integration with zppy

This module is designed to be called by zppy for automated budget analysis in E3SM post-processing workflows. The CLI interface allows seamless integration with zppy configuration files.

## Limitations

- Atmospheric energy budget analysis is limited due to incomplete energy flux information in ATM log files
- Land component energy budget analysis is not available due to missing energy flux data in LND log files
- Carbon budget checks are basic and may need expansion for comprehensive carbon cycle analysis
- Monthly frequency analysis requires sufficient log file temporal resolution
- Large log files may require significant memory for processing
