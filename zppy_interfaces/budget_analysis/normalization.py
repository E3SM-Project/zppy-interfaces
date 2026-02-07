"""Normalize the tidy event table for budget analysis."""

import pandas as pd

from .schema import COL_TABLE_TYPE, COL_UNITS, COL_VALUE

# Seconds per year (365-day calendar)
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalizations. Returns a new DataFrame with added columns.

    Adds 'normalized_value' and 'normalized_units' columns:
    - Flux rates (kg/m2s*1e6) -> mm/yr
    - Flux integrated (kg/m2*1e6) -> mm
    - State values (kg/m2*1e6) -> mm
    """
    df = df.copy()
    df["normalized_value"] = df[COL_VALUE].copy()
    df["normalized_units"] = df[COL_UNITS].copy()

    # Flux rates: kg/m2s * 1e6 -> mm/yr
    # 1 kg/m2 = 1 mm, so (val * 1e-6) kg/m2/s * seconds_per_year = mm/yr
    flux_mask = df[COL_TABLE_TYPE] == "flux"
    df.loc[flux_mask, "normalized_value"] = (
        df.loc[flux_mask, COL_VALUE] * SECONDS_PER_YEAR / 1e6
    )
    df.loc[flux_mask, "normalized_units"] = "mm/yr"

    # Integrated fluxes and states: kg/m2 * 1e6 -> mm
    integrated_mask = df[COL_TABLE_TYPE].isin(["flux_integrated", "state"])
    df.loc[integrated_mask, "normalized_value"] = (
        df.loc[integrated_mask, COL_VALUE] / 1e6
    )
    df.loc[integrated_mask, "normalized_units"] = "mm"

    return df
