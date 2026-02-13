"""Normalize the tidy event table for budget analysis."""

import pandas as pd

from .schema import COL_QUANTITY, COL_SOURCE, COL_TABLE_TYPE, COL_UNITS, COL_VALUE

# Seconds per year (365-day calendar)
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalizations. Returns a new DataFrame with added columns.

    Adds 'normalized_value' and 'normalized_units' columns:

    Water:
    - Flux rates (kg/m2s*1e6) -> mm/yr
    - Flux integrated (kg/m2*1e6) -> mm
    - State values (kg/m2*1e6) -> mm

    Heat:
    - Flux rates kept in W/m2 (cumulative scaling to J/m2 at plot time)

    Carbon:
    - Flux rates (kg-C/m2s*1e10) -> kg-C/m2*1e10/yr
    """
    df = df.copy()
    df["normalized_value"] = df[COL_VALUE].copy()
    df["normalized_units"] = df[COL_UNITS].copy()

    # --- Water ---
    water_mask = df[COL_QUANTITY] == "water"

    # Flux rates: kg/m2s * 1e6 -> mm/yr (ocean), kg/m2s -> mm/yr (ice)
    water_flux = water_mask & (df[COL_TABLE_TYPE] == "flux")

    # Ice data: kg/m2s -> mm/yr (no 1e6 factor)
    ice_flux = water_flux & (df[COL_SOURCE] == "ice")
    df.loc[ice_flux, "normalized_value"] = (
        df.loc[ice_flux, COL_VALUE] * SECONDS_PER_YEAR
    )

    # Ocean/other data: kg/m2s*1e6 -> mm/yr (with 1e6 factor)
    other_flux = water_flux & (df[COL_SOURCE] != "ice")
    df.loc[other_flux, "normalized_value"] = (
        df.loc[other_flux, COL_VALUE] * SECONDS_PER_YEAR / 1e6
    )
    df.loc[water_flux, "normalized_units"] = "mm/yr"

    # Integrated fluxes and states: kg/m2 * 1e6 -> mm
    water_integrated = water_mask & df[COL_TABLE_TYPE].isin(
        ["flux_integrated", "state"]
    )
    df.loc[water_integrated, "normalized_value"] = (
        df.loc[water_integrated, COL_VALUE] / 1e6
    )
    df.loc[water_integrated, "normalized_units"] = "mm"

    # --- Heat ---
    # Keep W/m2 as-is (no conversion needed)
    # Cumulative residual plots scale to J/m2 at plot time

    # --- Carbon ---
    # Flux rates: kg-C/m2s * 1e10 -> kg-C/m2*1e10 /yr
    carbon_flux = (df[COL_QUANTITY] == "carbon") & (df[COL_TABLE_TYPE] == "flux")
    df.loc[carbon_flux, "normalized_value"] = (
        df.loc[carbon_flux, COL_VALUE] * SECONDS_PER_YEAR
    )
    df.loc[carbon_flux, "normalized_units"] = "kg-C/m2*1e10/yr"

    return df
