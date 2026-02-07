"""Budget checks: definitions and evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import (
    COL_COMPONENT,
    COL_PERIOD,
    COL_QUANTITY,
    COL_SOURCE,
    COL_TABLE_TYPE,
    COL_TERM,
    COL_TIME,
)


@dataclass
class CheckResult:
    """Result of a single budget check across all time steps."""

    name: str
    description: str
    years: np.ndarray
    lhs: np.ndarray
    rhs: np.ndarray
    residual: np.ndarray  # lhs - rhs
    cumulative_residual: np.ndarray
    components: Optional[Dict[str, np.ndarray]] = field(default=None)


def _select(
    df: pd.DataFrame,
    period: Optional[str] = None,
    **filters,
) -> pd.DataFrame:
    """Filter df by column=value filters, optionally by period.

    If period is None, prefer 'annual' if available, else use 'monthly'.
    For monthly data, aggregate to annual by summing per year.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask = mask & (df[col] == val)
    subset = df[mask]

    if subset.empty:
        return subset

    # Determine period to use
    available = subset[COL_PERIOD].unique()
    if period is not None:
        subset = subset[subset[COL_PERIOD] == period]
    elif "annual" in available:
        subset = subset[subset[COL_PERIOD] == "annual"]
    else:
        # Monthly only — aggregate to annual per year
        # Rates (flux) get averaged; totals (state, flux_integrated) get summed
        group_keys = [
            COL_TIME,
            COL_COMPONENT,
            COL_QUANTITY,
            COL_TERM,
            COL_SOURCE,
            COL_TABLE_TYPE,
        ]
        flux_rows = subset[subset[COL_TABLE_TYPE] == "flux"]
        other_rows = subset[subset[COL_TABLE_TYPE] != "flux"]
        parts = []
        if not flux_rows.empty:
            parts.append(
                flux_rows.groupby(group_keys, as_index=False).agg(
                    {"normalized_value": "mean", "normalized_units": "first"}
                )
            )
        if not other_rows.empty:
            parts.append(
                other_rows.groupby(group_keys, as_index=False).agg(
                    {"normalized_value": "sum", "normalized_units": "first"}
                )
            )
        if parts:
            subset = pd.concat(parts, ignore_index=True)
        else:
            return subset.iloc[:0]

    return subset.sort_values(COL_TIME)


class BudgetCheck:
    """Base class for budget checks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        raise NotImplementedError


class CplComponentFluxes(BudgetCheck):
    """Per-component net water flux + global residual (*SUM*).

    Components dict includes each component's cumulative net flux
    plus a '*SUM*' entry for the global residual.
    """

    def __init__(self) -> None:
        super().__init__(
            "cpl_component_fluxes",
            "Coupler cumulative net water flux per component + residual",
        )

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        rows = _select(df, **{COL_SOURCE: "cpl", COL_TERM: "*SUM*"})
        if rows.empty:
            return None
        pivot = rows.pivot_table(
            index=COL_TIME, columns=COL_COMPONENT, values="normalized_value"
        ).sort_index()
        years = pivot.index.values

        components = {}
        for col in pivot.columns:
            components[col] = np.cumsum(pivot[col].values)

        residual = (
            pivot["*SUM*"].values
            if "*SUM*" in pivot.columns
            else pivot.sum(axis=1).values
        )
        return CheckResult(
            self.name,
            self.description,
            years,
            np.zeros_like(residual),
            residual,
            -residual,
            np.cumsum(-residual),
            components=components,
        )


class InterfaceMatch(BudgetCheck):
    """Do the coupler and component model agree on net water flux?

    Compares coupler *SUM* in the component column vs component's *SUM* flux.
    Works for any component (lnd, ocn, etc.).
    """

    def __init__(self, component: str, source: str) -> None:
        super().__init__(
            f"{component}_interface_match",
            f"{component} net water flux: coupler vs {source} model",
        )
        self.component = component  # coupler column name (e.g. "lnd", "ocn")
        self.source = source  # source tag in event table (e.g. "lnd", "ocn")

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        cpl = _select(
            df, **{COL_SOURCE: "cpl", COL_TERM: "*SUM*", COL_COMPONENT: self.component}
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        comp = _select(
            df, **{COL_SOURCE: self.source, COL_TERM: "*SUM*", COL_TABLE_TYPE: "flux"}
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        if cpl.empty or comp.empty:
            return None
        merged = cpl.join(comp, lsuffix="_cpl", rsuffix="_comp", how="inner")
        if merged.empty:
            return None
        years = merged.index.values
        c = merged["normalized_value_cpl"].values
        m = merged["normalized_value_comp"].values
        r = c - m
        return CheckResult(self.name, self.description, years, c, m, r, np.cumsum(r))


class LndClosure(BudgetCheck):
    """Does land storage change equal the integrated flux?

    Compares *NET CHANGE* TOTAL (state table) vs *SUM* (integrated flux table).
    """

    def __init__(self) -> None:
        super().__init__(
            "lnd_closure",
            "Land water closure: ΔStorage vs ∫Flux dt",
        )

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        storage = _select(
            df,
            **{
                COL_SOURCE: "lnd",
                COL_TERM: "*NET CHANGE*_TOTAL",
                COL_TABLE_TYPE: "state",
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        flux = _select(
            df,
            **{COL_SOURCE: "lnd", COL_TERM: "*SUM*", COL_TABLE_TYPE: "flux_integrated"},
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        if storage.empty or flux.empty:
            return None
        merged = storage.join(flux, lsuffix="_stor", rsuffix="_flux", how="inner")
        if merged.empty:
            return None
        years = merged.index.values
        ds = merged["normalized_value_stor"].values
        fi = merged["normalized_value_flux"].values
        r = ds - fi
        return CheckResult(self.name, self.description, years, ds, fi, r, np.cumsum(r))


class OcnClosure(BudgetCheck):
    """Does ocean mass change equal the net flux?

    Ocean logs are monthly. _select auto-aggregates to annual.
    Compares mass_change (state) vs *SUM* flux.
    """

    def __init__(self) -> None:
        super().__init__(
            "ocn_closure",
            "Ocean water closure: ΔMass vs net flux",
        )

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        mass = _select(
            df,
            **{COL_SOURCE: "ocn", COL_TERM: "mass_change", COL_TABLE_TYPE: "flux"},
        )
        if mass.empty:
            return None
        mass_ts = mass[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        flux = _select(
            df,
            **{COL_SOURCE: "ocn", COL_TERM: "*SUM*", COL_TABLE_TYPE: "flux"},
        )
        if flux.empty:
            return None
        flux_ts = flux[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        merged = mass_ts.join(flux_ts, lsuffix="_mass", rsuffix="_flux", how="inner")
        if merged.empty:
            return None
        years = merged.index.values
        ds = merged["normalized_value_mass"].values
        fi = merged["normalized_value_flux"].values
        r = ds - fi
        return CheckResult(self.name, self.description, years, ds, fi, r, np.cumsum(r))


DEFAULT_WATER_CHECKS: List[BudgetCheck] = [
    CplComponentFluxes(),
    InterfaceMatch("lnd", "lnd"),
    InterfaceMatch("ocn", "ocn"),
    LndClosure(),
    OcnClosure(),
]


def run_checks(
    df: pd.DataFrame,
    checks: Optional[List[BudgetCheck]] = None,
) -> List[CheckResult]:
    """Run budget checks against the normalized event table."""
    if checks is None:
        checks = DEFAULT_WATER_CHECKS
    results = []
    for check in checks:
        result = check.evaluate(df)
        if result is not None:
            results.append(result)
            print(f"  Check '{check.name}': {len(result.years)} years")
        else:
            print(f"  WARNING: Check '{check.name}' skipped (missing data)")
    return results
