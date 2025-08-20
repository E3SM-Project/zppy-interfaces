# Script to plot some global atmosphere and ocean time series
import csv
import importlib.resources as imp_res
from typing import Any, Dict, List, Tuple

import cftime
import numpy as np
import xarray

from zppy_interfaces.global_time_series.coupled_global_dataset_wrapper import (
    DatasetWrapper,
)
from zppy_interfaces.global_time_series.coupled_global_plotting import make_plot_pdfs
from zppy_interfaces.global_time_series.coupled_global_utils import Metric, Variable
from zppy_interfaces.global_time_series.coupled_global_viewer import (
    create_viewer,
    create_viewer_index,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)


# Useful helper functions and classes #########################################


def get_vars_original(plots_original: List[str]) -> List[Variable]:
    # NOTE: These are ALL atmosphere variables
    vars_original: List[Variable] = []
    if ("net_toa_flux_restom" in plots_original) or (
        "net_atm_energy_imbalance" in plots_original
    ):
        vars_original.append(Variable("RESTOM"))
    if "net_atm_energy_imbalance" in plots_original:
        vars_original.append(Variable("RESSURF"))
    if "global_surface_air_temperature" in plots_original:
        vars_original.append(Variable("TREFHT"))
    if "toa_radiation" in plots_original:
        vars_original.append(Variable("FSNTOA"))
        vars_original.append(Variable("FLUT"))
    if "net_atm_water_imbalance" in plots_original:
        vars_original.append(Variable("PRECC"))
        vars_original.append(Variable("PRECL"))
        vars_original.append(Variable("QFLX"))
    return vars_original


def land_csv_row_to_var(csv_row: List[str]) -> Variable:
    # “A” or “T” for global average over land area or global total, respectively
    metric: Metric
    if csv_row[1] == "A":
        metric = Metric.AVERAGE
    elif csv_row[1] == "T":
        metric = Metric.TOTAL
    else:
        raise ValueError(f"Invalid metric={csv_row[1]}")
    return Variable(
        variable_name=csv_row[0],
        metric=metric,
        scale_factor=float(csv_row[2]),
        original_units=csv_row[3],
        final_units=csv_row[4],
        group=csv_row[5],
        long_name=csv_row[6],
    )


def construct_land_variables(requested_vars: List[str]) -> List[Variable]:
    var_list: List[Variable] = []
    header = True
    csv_filename = str(
        imp_res.files("zppy_interfaces.global_time_series") / "zppy_land_fields.csv"
    )
    with open(csv_filename, newline="") as csv_file:
        logger.debug("Reading zppy_land_fields.csv")
        var_reader = csv.reader(csv_file)
        for row in var_reader:
            # logger.debug(f"row={row}")
            # Skip the header row
            if header:
                header = False
            else:
                # If set to "all" then we want all variables.
                # Design note: we can't simply run all variables if requested_vars is empty because
                # that would actually mean the user doesn't want to make *any* land plots.
                if (requested_vars == ["all"]) or (row[0] in requested_vars):
                    row_elements_strip_whitespace: List[str] = list(
                        map(lambda x: x.strip(), row)
                    )
                    var_list.append(land_csv_row_to_var(row_elements_strip_whitespace))
    return var_list


def construct_generic_variables(requested_vars: List[str]) -> List[Variable]:
    var_list: List[Variable] = []
    for var_name in requested_vars:
        var_list.append(Variable(var_name))
    return var_list


class RequestedVariables(object):
    def __init__(self, parameters: Parameters):
        self.vars_original: List[Variable] = get_vars_original(
            parameters.plots_original
        )
        self.vars_land: List[Variable] = construct_land_variables(parameters.plots_lnd)

        # Use generic constructor
        self.vars_atm: List[Variable] = construct_generic_variables(
            parameters.plots_atm
        )
        self.vars_ice: List[Variable] = construct_generic_variables(
            parameters.plots_ice
        )
        self.vars_ocn: List[Variable] = construct_generic_variables(
            parameters.plots_ocn
        )


# Setup #######################################################################
def get_data_dir(parameters: Parameters, component: str, conditional: bool) -> str:
    return (
        f"{parameters.case_dir}/post/{component}/glb/ts/monthly/{parameters.ts_num_years_str}yr/"
        if conditional
        else ""
    )


def get_exps(parameters: Parameters) -> List[Dict[str, Any]]:
    # Experiments
    use_atmos: bool = (parameters.plots_atm != []) or (parameters.plots_original != [])
    # Use set intersection: check if any of these 3 plots were requested
    set_intersection: set = set(["change_ohc", "max_moc", "change_sea_level"]) & set(
        parameters.plots_original
    )
    has_original_ocn_plots: bool = set_intersection != set()
    use_ocn: bool = (parameters.plots_ocn != []) or has_original_ocn_plots
    ocean_dir = get_data_dir(parameters, "ocn", use_ocn)
    exps: List[Dict[str, Any]] = [
        {
            "atmos": get_data_dir(parameters, "atm", use_atmos),
            "ice": get_data_dir(parameters, "ice", parameters.plots_ice != []),
            "land": get_data_dir(parameters, "lnd", parameters.plots_lnd != []),
            "ocean": ocean_dir,
            "moc": ocean_dir,
            "vol": ocean_dir,
            "name": parameters.experiment_name,
            "yoffset": 0.0,
            "yr": ([parameters.year1, parameters.year2],),
            "color": f"{parameters.color}",
        }
    ]
    return exps


def load_all_var_data(
    exp: Dict[str, Any],
    exp_key: str,
    var_list: List[Variable],
    valid_vars: List[str],
    invalid_vars: List[str],
) -> Tuple[List[Variable], Dict[str, Tuple[xarray.core.dataarray.DataArray, str]]]:
    """Load data for all variables for all regions.
    
    Args:
        exp: Experiment configuration
        exp_key: Key for data directory
        var_list: List of variables to load
        valid_vars: List to track successfully loaded variables
        invalid_vars: List to track variables that couldn't be loaded
        
    Returns:
        Tuple of (list of successfully loaded variables, dictionary of data arrays)
    """
    new_var_list: List[Variable] = []
    all_data_dict = {}
    
    if exp[exp_key] != "":
        try:
            dataset_wrapper: DatasetWrapper = DatasetWrapper(exp[exp_key])
        except Exception as e:
            logger.critical(e)
            logger.critical(
                f"DatasetWrapper object could not be created for {exp_key}={exp[exp_key]}"
            )
            raise e
            
        for var in var_list:
            var_str: str = var.variable_name
            try:
                # Get data for all regions
                data_array: xarray.core.dataarray.DataArray
                units: str
                data_array, units = dataset_wrapper.globalAnnual(var, all_regions=True)
                
                # Store the result keyed by variable name
                all_data_dict[var_str] = (data_array, units)
                
                # Track successful variables
                valid_vars.append(str(var_str))
                new_var_list.append(var)
                
                # Store year info if not already present
                if "year" not in exp["annual"]:
                    years: np.ndarray[cftime.DatetimeNoLeap] = data_array.coords["time"].values
                    exp["annual"]["year"] = [x.year for x in years]
                    
            except Exception as e:
                logger.error(e)
                logger.error(f"globalAnnual failed for {var_str}")
                invalid_vars.append(str(var_str))
                
        del dataset_wrapper
        
    return new_var_list, all_data_dict


def extract_region_data(
    all_data_dict: Dict[str, Tuple[xarray.core.dataarray.DataArray, str]],
    rgn: str,
) -> Dict[str, Tuple[xarray.core.dataarray.DataArray, str]]:
    """Extract data for a specific region from the all-regions data dictionary.
    
    Args:
        all_data_dict: Dictionary mapping variable names to (data_array, units) tuples
        rgn: Region to extract ('glb', 'n', or 's')
        
    Returns:
        Dictionary with data arrays extracted for the specified region
    """
    region_data_dict = {}
    
    # Map region string to index
    if rgn == "glb":
        n = 0
    elif rgn == "n":
        n = 1
    elif rgn == "s":
        n = 2
    else:
        raise RuntimeError(f"Invalid rgn={rgn}")
    
    # Process each variable
    for var_str, (data_array, units) in all_data_dict.items():
        # Extract region if the data has multiple regions
        if "rgn" in data_array.dims and data_array.sizes["rgn"] > 1:
            # Extract the specific region
            region_data = data_array.isel(rgn=n)
        elif rgn != "glb":
            # If no rgn dimension but trying to get n or s, that's an error
            raise RuntimeError(
                f"var={var_str} only has global data. Cannot process rgn={rgn}"
            )
        else:
            # No rgn dimension but wanting global data, or already extracted
            region_data = data_array
            
        # Store in output dictionary
        region_data_dict[var_str] = (region_data, units)
    
    return region_data_dict


def load_all_region_data(
    parameters: Parameters, requested_variables: RequestedVariables
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Load all data for all regions at once.
    
    Args:
        parameters: Configuration parameters
        requested_variables: Variables to load for each component
        
    Returns:
        Tuple of (experiment data, valid variables, invalid variables)
    """
    # Get experiment configurations
    exps: List[Dict[str, Any]] = get_exps(parameters)
    
    # Track valid and invalid variables by component
    component_valid_vars = {"atmos": [], "ice": [], "land": [], "ocean": []}
    component_invalid_vars = {"atmos": [], "ice": [], "land": [], "ocean": []}
    
    # Process each experiment
    for exp in exps:
        # Initialize annual data and all-regions data storage
        exp["annual"] = {}
        exp["all_regions_data"] = {}
        
        # Initialize component data dictionaries
        exp["all_regions_data"]["atmos"] = {}
        exp["all_regions_data"]["ice"] = {}
        exp["all_regions_data"]["land"] = {}
        exp["all_regions_data"]["ocean"] = {}
        
        # Load data for each component - original vars (atmosphere variables)
        requested_variables.vars_original, atmos_original_data = load_all_var_data(
            exp,
            "atmos",
            requested_variables.vars_original,
            component_valid_vars["atmos"],
            component_invalid_vars["atmos"],
        )
        exp["all_regions_data"]["atmos"].update(atmos_original_data)
        
        # Load data for each component - atmosphere variables
        requested_variables.vars_atm, atmos_data = load_all_var_data(
            exp,
            "atmos",
            requested_variables.vars_atm,
            component_valid_vars["atmos"],
            component_invalid_vars["atmos"],
        )
        exp["all_regions_data"]["atmos"].update(atmos_data)
        
        # Load data for each component - ice variables
        requested_variables.vars_ice, ice_data = load_all_var_data(
            exp,
            "ice",
            requested_variables.vars_ice,
            component_valid_vars["ice"],
            component_invalid_vars["ice"],
        )
        exp["all_regions_data"]["ice"].update(ice_data)
        
        # Load data for each component - land variables
        requested_variables.vars_land, land_data = load_all_var_data(
            exp,
            "land",
            requested_variables.vars_land,
            component_valid_vars["land"],
            component_invalid_vars["land"],
        )
        exp["all_regions_data"]["land"].update(land_data)
        
        # Load data for each component - ocean variables
        requested_variables.vars_ocn, ocn_data = load_all_var_data(
            exp,
            "ocean",
            requested_variables.vars_ocn,
            component_valid_vars["ocean"],
            component_invalid_vars["ocean"],
        )
        exp["all_regions_data"]["ocean"].update(ocn_data)
        
        # Special handling for ocean heat content
        if exp["ocean"] != "":
            try:
                dataset_wrapper = DatasetWrapper(exp["ocean"])
                data_array, units = dataset_wrapper.globalAnnual(Variable("ohc"), all_regions=True)
                
                # Store in all regions data
                exp["all_regions_data"]["ocean"]["ohc"] = (data_array, units)
                
                # Track as valid variable
                component_valid_vars["ocean"].append("ohc")
                
                del dataset_wrapper
            except Exception as e:
                logger.error(e)
                logger.error("Failed to load ohc data")
                component_invalid_vars["ocean"].append("ohc")
                
        # Special handling for ocean volume
        if exp["vol"] != "":
            try:
                dataset_wrapper = DatasetWrapper(exp["vol"])
                data_array, units = dataset_wrapper.globalAnnual(Variable("volume"), all_regions=True)
                
                # Store in all regions data
                exp["all_regions_data"]["ocean"]["volume"] = (data_array, units)
                
                # Track as valid variable
                component_valid_vars["ocean"].append("volume")
                
                del dataset_wrapper
            except Exception as e:
                logger.error(e)
                logger.error("Failed to load volume data")
                component_invalid_vars["ocean"].append("volume")
    
    # Log success and failures for all components
    for component, valid_vars in component_valid_vars.items():
        if valid_vars:
            logger.info(f"{component} variables were computed successfully: {valid_vars}")
    
    for component, invalid_vars in component_invalid_vars.items():
        if invalid_vars:
            logger.error(f"{component} variables could not be computed: {invalid_vars}")
            
    return exps, component_valid_vars, component_invalid_vars


def process_data(
    all_region_exps: List[Dict[str, Any]], rgn: str
) -> List[Dict[str, Any]]:
    """Process data for a specific region.
    
    Args:
        all_region_exps: Experiments with all-regions data already loaded
        rgn: Region to process ('glb', 'n', or 's')
        
    Returns:
        List of experiment dictionaries with region-specific data
    """
    # Create a deep copy to avoid modifying the original
    import copy
    exps = copy.deepcopy(all_region_exps)
    
    # Extract region-specific data for each experiment
    for exp in exps:
        # Extract atmosphere data
        if "atmos" in exp["all_regions_data"]:
            atmos_region_data = extract_region_data(exp["all_regions_data"]["atmos"], rgn)
            exp["annual"].update(atmos_region_data)
            
        # Extract ice data
        if "ice" in exp["all_regions_data"]:
            ice_region_data = extract_region_data(exp["all_regions_data"]["ice"], rgn)
            exp["annual"].update(ice_region_data)
            
        # Extract land data
        if "land" in exp["all_regions_data"]:
            land_region_data = extract_region_data(exp["all_regions_data"]["land"], rgn)
            exp["annual"].update(land_region_data)
            
        # Extract ocean data
        if "ocean" in exp["all_regions_data"]:
            ocean_region_data = extract_region_data(exp["all_regions_data"]["ocean"], rgn)
            exp["annual"].update(ocean_region_data)
            
        # Process OHC anomalies if available
        if "ohc" in exp["annual"]:
            # anomalies with respect to first year
            ohc_data, ohc_units = exp["annual"]["ohc"]
            ohc_anomaly = ohc_data - ohc_data[0]
            exp["annual"]["ohc"] = (ohc_anomaly, ohc_units)
        
        # Process volume anomalies if available
        if "volume" in exp["annual"]:
            # anomalies with respect to first year
            volume_data, volume_units = exp["annual"]["volume"]
            volume_anomaly = volume_data - volume_data[0]
            exp["annual"]["volume"] = (volume_anomaly, volume_units)
            
        # Clean up all_regions_data to save memory
        del exp["all_regions_data"]
        
    return exps


# Run coupled_global for a single region ##########################################################
def run_region(
    parameters: Parameters, 
    requested_variables: RequestedVariables, 
    rgn: str, 
    all_region_exps: List[Dict[str, Any]]
):
    """Process and plot data for a specific region.
    
    Args:
        parameters: Configuration parameters
        requested_variables: Variables to process
        rgn: Region to process ('glb', 'n', or 's')
        all_region_exps: Experiment data with all regions already loaded
    """
    # Extract data for this specific region
    exps: List[Dict[str, Any]] = process_data(all_region_exps, rgn)

    # Set up x-axis limits
    xlim: List[float] = [float(parameters.year1), float(parameters.year2)]

    # Track successful and failed plots
    valid_plots: List[str] = []
    invalid_plots: List[str] = []

    # Use list of tuples rather than a dict, to keep order
    # Note: we use `parameters.plots_original` rather than `requested_variables.vars_original`
    # because the "original" plots are expecting plot names that are not variable names.
    # The model components however are expecting plot names to be variable names.
    mapping: List[Tuple[str, List[str]]] = [
        ("original", parameters.plots_original),
        ("atm", list(map(lambda v: v.variable_name, requested_variables.vars_atm))),
        ("ice", list(map(lambda v: v.variable_name, requested_variables.vars_ice))),
        ("lnd", list(map(lambda v: v.variable_name, requested_variables.vars_land))),
        ("ocn", list(map(lambda v: v.variable_name, requested_variables.vars_ocn))),
    ]
    
    # Generate plots for each component
    for component, plot_list in mapping:
        make_plot_pdfs(
            parameters,
            rgn,
            component,
            xlim,
            exps,
            plot_list,
            valid_plots,
            invalid_plots,
        )
        
    # Log results
    if valid_plots:
        logger.info(f"These {rgn} region plots generated successfully: {valid_plots}")
    
    if invalid_plots:
        logger.error(
            f"These {rgn} region plots could not be generated successfully: {invalid_plots}"
        )


def get_vars(requested_variables: RequestedVariables, component: str) -> List[Variable]:
    """Get variable list for a specific component."""
    vars: List[Variable]
    if component == "original":
        vars = requested_variables.vars_original
    elif component == "atm":
        vars = requested_variables.vars_atm
    elif component == "ice":
        vars = requested_variables.vars_ice
    elif component == "lnd":
        vars = requested_variables.vars_land
    elif component == "ocn":
        vars = requested_variables.vars_ocn
    else:
        raise ValueError(f"Invalid component={component}")
    return vars


def coupled_global(parameters: Parameters) -> None:
    """Main entry point for the global time series plots.
    
    Changes from original version:
    - Load all data for all regions once, then process each region using that data
    - This reduces I/O operations significantly
    """
    # Initialize variables for all components
    requested_variables = RequestedVariables(parameters)
    
    # OPTIMIZATION: Load all data for all regions once
    logger.info("Loading data for all regions...")
    all_region_exps, valid_vars, invalid_vars = load_all_region_data(parameters, requested_variables)
    
    # Process each region using the already-loaded data
    for rgn in parameters.regions:
        logger.info(f"Processing region: {rgn}")
        run_region(parameters, requested_variables, rgn, all_region_exps)
    
    # Create viewer if requested
    if parameters.make_viewer:
        # In this case, we don't want the summary PDF.
        # Rather, we want to construct a viewer similar to E3SM Diags.
        title_and_url_list: List[Tuple[str, str]] = []
        
        # Create viewers for each component except original
        for component in ["atm", "ice", "lnd", "ocn"]:  # Don't create viewer for original component
            vars = get_vars(requested_variables, component)
            if vars:
                url = create_viewer(parameters, vars, component)
                logger.info(f"Viewer URL for {component}: {url}")
                title_and_url_list.append((component, url))
                
        # Special case for original plots: always use user-provided dimensions.
        vars = get_vars(requested_variables, "original")
        if vars:
            logger.info("Using user provided dimensions for original plots PDF")
            title_and_url_list.append(
                (
                    "original",
                    f"{parameters.figstr}_glb_original.pdf",
                )
            )
        
        # Create index page for all viewers
        index_url: str = create_viewer_index(parameters.results_dir, title_and_url_list)
        logger.info(f"Viewer index URL: {index_url}")
