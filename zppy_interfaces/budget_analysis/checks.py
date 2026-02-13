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
    lhs_label: str = "LHS"
    rhs_label: str = "RHS"
    components: Optional[Dict[str, np.ndarray]] = field(default=None)


def _select(
    df: pd.DataFrame,
    period: Optional[str] = None,
    **filters,
) -> pd.DataFrame:
    """Filter df by column=value filters, optionally by period.

    If period is None, prefer 'annual' if available, else use 'monthly'.
    Monthly data is kept at monthly resolution (COL_TIME encodes
    year + fractional month).
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
        # Monthly — keep at monthly resolution.
        # COL_TIME already encodes year + fractional month,
        # so each month has a unique time value.
        subset = subset[subset[COL_PERIOD] == "monthly"]

    return subset.sort_values(COL_TIME)


class BudgetCheck:
    """Base class for budget checks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        raise NotImplementedError


class CplComponentFluxes(BudgetCheck):
    """Per-component net flux + global residual (*SUM*).

    Components dict includes each component's cumulative net flux
    plus a '*SUM*' entry for the global residual.
    Supports both water and heat quantities.
    """

    def __init__(self, quantity: str = "water") -> None:
        super().__init__(
            f"cpl_{quantity}_component_fluxes",
            f"Coupler cumulative net {quantity} flux per component + residual",
        )
        self.quantity = quantity

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        rows = _select(
            df,
            **{COL_SOURCE: "cpl", COL_TERM: "*SUM*", COL_QUANTITY: self.quantity},
        )
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
    """Do the coupler and component model agree on net flux?

    Compares coupler *SUM* in the component column vs component's *SUM* flux.
    Works for any component (lnd, ocn, etc.) and quantity (water, heat).
    """

    def __init__(
        self,
        component: str,
        source: str,
        quantity: str = "water",
        comp_sum_term: str = "*SUM*",
    ) -> None:
        super().__init__(
            f"{component}_{quantity}_interface_match",
            f"{component} net {quantity} flux: coupler vs {source} model",
        )
        self.component = component
        self.source = source
        self.quantity = quantity
        self.comp_sum_term = comp_sum_term

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        cpl = _select(
            df,
            **{
                COL_SOURCE: "cpl",
                COL_TERM: "*SUM*",
                COL_COMPONENT: self.component,
                COL_QUANTITY: self.quantity,
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        comp = _select(
            df,
            **{
                COL_SOURCE: self.source,
                COL_TERM: self.comp_sum_term,
                COL_TABLE_TYPE: "flux",
                COL_QUANTITY: self.quantity,
            },
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
        return CheckResult(
            self.name,
            self.description,
            years,
            c,
            m,
            r,
            np.cumsum(r),
            lhs_label=f"cpl ({self.component})",
            rhs_label=f"{self.source} ({self.comp_sum_term})",
        )


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
        return CheckResult(
            self.name,
            self.description,
            years,
            ds,
            fi,
            r,
            np.cumsum(r),
            lhs_label="ΔStorage (*NET CHANGE*)",
            rhs_label="∫Flux dt (*SUM*)",
        )


class OcnClosure(BudgetCheck):
    """Does ocean mass/energy change equal the net flux?

    Ocean logs are monthly. _select auto-aggregates to annual.
    For water: compares mass_change vs *SUM* flux.
    For heat: compares energy_change vs *SUM* flux.
    """

    CHANGE_TERM: Dict[str, str] = {
        "water": "Mass change",
        "heat": "Energy change",
    }

    SUM_TERM: Dict[str, str] = {
        "water": "SUM VOLUME FLUXES",
        "heat": "SUM IMP+EXP HEAT FLUXES",
    }

    def __init__(self, quantity: str = "water") -> None:
        super().__init__(
            f"ocn_{quantity}_closure",
            f"Ocean {quantity} closure: Δ{'Mass' if quantity == 'water' else 'Energy'} vs net flux",
        )
        self.quantity = quantity

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        change_term = self.CHANGE_TERM[self.quantity]
        sum_term = self.SUM_TERM[self.quantity]

        mass = _select(
            df,
            **{
                COL_SOURCE: "ocn",
                COL_TERM: change_term,
                COL_TABLE_TYPE: "flux",
                COL_QUANTITY: self.quantity,
            },
        )
        if mass.empty:
            return None
        mass_ts = mass[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        flux = _select(
            df,
            **{
                COL_SOURCE: "ocn",
                COL_TERM: sum_term,
                COL_TABLE_TYPE: "flux",
                COL_QUANTITY: self.quantity,
            },
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
        change_label = "ΔMass" if self.quantity == "water" else "ΔEnergy"
        return CheckResult(
            self.name,
            self.description,
            years,
            ds,
            fi,
            r,
            np.cumsum(r),
            lhs_label=f"{change_label} ({change_term})",
            rhs_label=f"Net Flux ({sum_term})",
        )


class IceClosure(BudgetCheck):
    """Ice mass/energy closure: storage change flux vs net flux.

    For water: compares Mass change flux vs Net mass flux (both kg/m2s -> mm/yr).
    For heat: compares Energy change flux vs Net energy flux (both W/m2).
    """

    CHANGE_TERM: Dict[str, str] = {
        "water": "Mass change flux",
        "heat": "Energy change flux",
    }

    FLUX_TERM: Dict[str, str] = {
        "water": "Net mass flux",
        "heat": "Net energy flux",
    }

    def __init__(self, quantity: str = "water") -> None:
        super().__init__(
            f"ice_{quantity}_closure",
            f"Ice {quantity} closure: {'Mass' if quantity == 'water' else 'Energy'} change flux vs net flux",
        )
        self.quantity = quantity

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        change_term = self.CHANGE_TERM[self.quantity]
        flux_term = self.FLUX_TERM[self.quantity]

        # Get ice storage change flux
        storage_flux = _select(
            df,
            **{
                COL_SOURCE: "ice",
                COL_TERM: change_term,
                COL_QUANTITY: self.quantity,
                COL_TABLE_TYPE: "flux",
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        # Get ice net flux
        net_flux = _select(
            df,
            **{
                COL_SOURCE: "ice",
                COL_TERM: flux_term,
                COL_QUANTITY: self.quantity,
                COL_TABLE_TYPE: "flux",
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        if storage_flux.empty or net_flux.empty:
            return None

        merged = storage_flux.join(
            net_flux, lsuffix="_change", rsuffix="_net", how="inner"
        )
        if merged.empty:
            return None

        years = merged.index.values
        change = merged["normalized_value_change"].values
        net = merged["normalized_value_net"].values
        residual = change - net

        return CheckResult(
            self.name,
            self.description,
            years,
            change,
            net,
            residual,
            np.cumsum(residual),
            lhs_label=f"Storage Change ({change_term})",
            rhs_label=f"Net Flux ({flux_term})",
        )


class IceInterfaceMatch(BudgetCheck):
    """Do the coupler and ice model agree on net flux?

    Special case for ice: coupler splits ice into 'ice_nh' and 'ice_sh' components,
    so we sum both hemispheres to compare with ice model's total.
    """

    def __init__(self, quantity: str = "water") -> None:
        super().__init__(
            f"ice_{quantity}_interface_match",
            f"Ice net {quantity} flux: coupler (nh+sh) vs ice model",
        )
        self.quantity = quantity

    def evaluate(self, df: pd.DataFrame) -> Optional[CheckResult]:
        # Get coupler data for both ice hemispheres
        cpl_nh = _select(
            df,
            **{
                COL_SOURCE: "cpl",
                COL_TERM: "*SUM*",
                COL_COMPONENT: "ice_nh",
                COL_QUANTITY: self.quantity,
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        cpl_sh = _select(
            df,
            **{
                COL_SOURCE: "cpl",
                COL_TERM: "*SUM*",
                COL_COMPONENT: "ice_sh",
                COL_QUANTITY: self.quantity,
            },
        )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

        if cpl_nh.empty or cpl_sh.empty:
            print(f"DEBUG: Missing coupler ice hemisphere data for {self.quantity}")
            if cpl_nh.empty:
                print("  Missing ice_nh data")
            if cpl_sh.empty:
                print("  Missing ice_sh data")
            return None

        # Sum both hemispheres
        cpl_combined = cpl_nh.join(cpl_sh, lsuffix="_nh", rsuffix="_sh", how="inner")
        if cpl_combined.empty:
            print("DEBUG: No overlapping time periods between ice_nh and ice_sh")
            return None

        cpl_combined["normalized_value"] = (
            cpl_combined["normalized_value_nh"] + cpl_combined["normalized_value_sh"]
        )
        cpl_total = cpl_combined[["normalized_value"]]

        # Get ice model data - need to determine the right term
        # For water, try common mass flux terms, for heat try energy flux terms
        ice_terms = {
            "water": ["Net mass flux", "*SUM*", "Mass change flux"],
            "heat": ["Net energy flux", "*SUM*", "Energy change flux"],
        }

        ice_model = None
        used_term = None
        for term in ice_terms[self.quantity]:
            ice_candidate = _select(
                df,
                **{
                    COL_SOURCE: "ice",
                    COL_TERM: term,
                    COL_TABLE_TYPE: "flux",
                    COL_QUANTITY: self.quantity,
                },
            )[[COL_TIME, "normalized_value"]].set_index(COL_TIME)

            if not ice_candidate.empty:
                ice_model = ice_candidate
                used_term = term
                break

        if ice_model is None:
            return None

        merged = cpl_total.join(ice_model, lsuffix="_cpl", rsuffix="_ice", how="inner")
        if merged.empty:
            return None

        years = merged.index.values
        c = merged["normalized_value_cpl"].values
        m = merged["normalized_value_ice"].values
        r = c - m

        return CheckResult(
            self.name,
            self.description,
            years,
            c,
            m,
            r,
            np.cumsum(r),
            lhs_label="cpl (ice_nh + ice_sh)",
            rhs_label=f"ice ({used_term})",
        )


DEFAULT_WATER_CHECKS: List[BudgetCheck] = [
    CplComponentFluxes(quantity="water"),
    InterfaceMatch("lnd", "lnd", quantity="water"),
    InterfaceMatch("ocn", "ocn", quantity="water", comp_sum_term="SUM VOLUME FLUXES"),
    IceInterfaceMatch(quantity="water"),
    LndClosure(),
    OcnClosure(quantity="water"),
    IceClosure(quantity="water"),
]

DEFAULT_HEAT_CHECKS: List[BudgetCheck] = [
    CplComponentFluxes(quantity="heat"),
    InterfaceMatch(
        "ocn", "ocn", quantity="heat", comp_sum_term="SUM IMP+EXP HEAT FLUXES"
    ),
    IceInterfaceMatch(quantity="heat"),
    OcnClosure(quantity="heat"),
    IceClosure(quantity="heat"),
]

DEFAULT_CARBON_CHECKS: List[BudgetCheck] = [
    CplComponentFluxes(quantity="carbon"),
]


def run_checks(
    df: pd.DataFrame,
    checks: Optional[List[BudgetCheck]] = None,
) -> List[CheckResult]:
    """Run budget checks against the normalized event table."""
    if checks is None:
        checks = DEFAULT_WATER_CHECKS
    # Determine time unit label from data period
    periods = df[COL_PERIOD].unique() if not df.empty else []
    if "monthly" in periods:
        time_label = "months"
    else:
        time_label = "years"
    results = []
    for check in checks:
        result = check.evaluate(df)
        if result is not None:
            results.append(result)
            print(f"  Check '{check.name}': {len(result.years)} {time_label}")
        else:
            print(f"  WARNING: Check '{check.name}' skipped (missing data)")
    return results
