"""Translation between the canonical (EAM) variable names used by the original
global time series plots and the EAMxx variables they are computed from.

The original plots (``plots_original``) are written in terms of EAM variable
names such as ``RESTOM``, ``TREFHT``, and ``QFLX``. EAMxx writes the same
physical quantities under different names, and some of them must be derived
from more than one EAMxx field. Rather than duplicating the plotting code for
EAMxx, we keep the canonical names everywhere and translate them here, right
where the data is read.

Two kinds of translation are needed:

* aliases: the canonical variable is a single EAMxx variable under a different
  name (e.g., ``TREFHT`` is ``T_2m``);
* derivations: the canonical variable is a combination of other variables
  (e.g., ``FSNTOA`` is
  ``SW_flux_dn_at_model_top - SW_flux_up_at_model_top``).

Derivations are expressed in terms of other canonical names where possible, so
they get resolved recursively, just like the EAM derivations in
``DatasetWrapper.globalAnnualHelper``.

Which model wrote the data is detected rather than configured: if any of these
EAMxx variables is present, the data is EAMxx. See ``is_eamxx_data``.

These definitions match the EAMxx entries of ``DERIVED_VARIABLES`` in
e3sm_diags (``e3sm_diags/derivations/derivations.py``); the formulas below are
the equivalents of e3sm_diags' ``rst``, ``restom3``, ``netflux6``, and
``prect``. Note that EAMxx's "model top" is the top of the atmosphere, so for
EAMxx there is no distinction between the model-top fluxes (``FSNT``/``FLNT``)
and the TOA fluxes (``FSNTOA``/``FLUT``).
"""

from typing import Callable, Dict, Hashable, Iterable, List, Set, Tuple

from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)

# Canonical variable name -> EAMxx variable name, for one-to-one renames.
EAMXX_ALIASES: Dict[str, str] = {
    # Reference height (2 m) air temperature [K]
    "TREFHT": "T_2m",
    # Surface (radiative) temperature [K]
    "TS": "surf_radiative_T",
    # Upwelling longwave flux at the top of the model [W m-2]
    "FLUT": "LW_flux_up_at_model_top",
    # Surface sensible heat flux [W m-2]
    "SHFLX": "surf_sens_flux",
    # Surface latent heat flux [W m-2]
    # EAMxx provides this directly, so unlike EAM it does not have to be
    # derived from the water flux and the frozen precipitation.
    "LHFLX": "surface_upward_latent_heat_flux",
    # Surface water flux [kg m-2 s-1]
    "QFLX": "surf_evap",
}

# Canonical variable name -> (variables it is computed from, how to combine
# them). Source variables may themselves be canonical names, in which case they
# are resolved recursively.
EAMXX_DERIVATIONS: Dict[str, Tuple[List[str], Callable]] = {
    # Net downward radiative flux at the top of the model [W m-2].
    # FSNTOA and FLUT are canonical names, not EAMxx variables; they are
    # resolved further down this table, giving
    # SW_flux_dn_at_model_top - SW_flux_up_at_model_top - LW_flux_up_at_model_top.
    "RESTOM": (["FSNTOA", "FLUT"], lambda fsntoa, flut: fsntoa - flut),
    # The same quantity at the top of the atmosphere
    "RESTOA": (["FSNTOA", "FLUT"], lambda fsntoa, flut: fsntoa - flut),
    # Net downward shortwave flux at the top of the model [W m-2]
    "FSNTOA": (
        ["SW_flux_dn_at_model_top", "SW_flux_up_at_model_top"],
        lambda swdn, swup: swdn - swup,
    ),
    # Net downward shortwave flux at the surface [W m-2]
    "FSNS": (
        ["SW_flux_dn_at_model_bot", "SW_flux_up_at_model_bot"],
        lambda swdn, swup: swdn - swup,
    ),
    # Net upward longwave flux at the surface [W m-2]
    "FLNS": (
        ["LW_flux_up_at_model_bot", "LW_flux_dn_at_model_bot"],
        lambda lwup, lwdn: lwup - lwdn,
    ),
    # Net downward heat flux at the surface [W m-2].
    # All four source names are canonical, resolving to
    # (SW_flux_dn_at_model_bot - SW_flux_up_at_model_bot)
    # - (LW_flux_up_at_model_bot - LW_flux_dn_at_model_bot)
    # - surf_sens_flux - surface_upward_latent_heat_flux.
    "RESSURF": (
        ["FSNS", "FLNS", "SHFLX", "LHFLX"],
        lambda fsns, flns, shflx, lhflx: fsns - flns - shflx - lhflx,
    ),
    # Total surface precipitation rate [kg m-2 s-1].
    # EAMxx splits precipitation into liquid and ice rather than into
    # convective and large-scale, and reports them as volume fluxes [m s-1],
    # hence the factor of 1e3 (the density of water).
    # Note that e3sm_diags reports PRECT in mm/day; here it is left in
    # kg m-2 s-1 so that it can be differenced with QFLX directly.
    "PRECT": (
        ["precip_liq_surf_mass_flux", "precip_ice_surf_mass_flux"],
        lambda precip_liq, precip_ice: 1.0e3 * (precip_liq + precip_ice),
    ),
}


def get_eamxx_source_variables(var: str) -> List[str]:
    """List the EAMxx variables `var` is computed from.

    Canonical names are expanded recursively; names that are neither aliased
    nor derived are assumed to be EAMxx names already.
    """
    if var in EAMXX_DERIVATIONS:
        source_variables: List[str] = []
        for source_var in EAMXX_DERIVATIONS[var][0]:
            for expanded_var in get_eamxx_source_variables(source_var):
                # Keep the order, but don't list a variable more than once.
                if expanded_var not in source_variables:
                    source_variables.append(expanded_var)
        return source_variables
    return [EAMXX_ALIASES.get(var, var)]


def get_eamxx_variables() -> Set[str]:
    """Every EAMxx variable the plots can be computed from."""
    variables: Set[str] = set(EAMXX_ALIASES.values())
    for var in EAMXX_DERIVATIONS:
        variables.update(get_eamxx_source_variables(var))
    return variables


def is_eamxx_data(available_variables: Iterable[Hashable]) -> bool:
    """Whether these variables came from EAMxx rather than EAM.

    Any variable the plots can use is enough to tell the two apart, since the
    models share none of these names. That means detection still works when
    only some of the plots are requested, and data without any EAMxx variable
    is read exactly as before.
    """
    eamxx_variables: Set[str] = get_eamxx_variables() & set(available_variables)
    if eamxx_variables:
        logger.info(
            f"Reading EAMxx variables, based on finding {sorted(eamxx_variables)}"
        )
        return True
    return False
