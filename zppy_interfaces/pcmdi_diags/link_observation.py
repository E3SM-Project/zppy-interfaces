import argparse
import sys
from typing import Dict, List

from zppy_interfaces.pcmdi_diags.pcmdi_zppy_util import ObservationLinker


class LinkObservationParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.model_name: str = f"{args['model_name_ref']}.{args['tableID_ref']}"
        self.variables: List[str] = args["vars"].split(",")
        self.obs_sets: List[str] = args["obs_sets"].split(",")
        self.obs_ts: str = args["obs_ts"]
        self.obstmp_dir: str = args["obstmp_dir"]


def main():
    parameters: LinkObservationParameters = _get_args()
    # Mapping from observational variable names to CMIP-standard
    alt_obs_map: Dict[str, str] = {
        "pr": "PRECT",
        "sst": "ts",
        "sfcWind": "si10",
        "taux": "tauu",
        "tauy": "tauv",
        "rltcre": "toa_cre_lw_mon",
        "rstcre": "toa_cre_sw_mon",
        "rtmt": "toa_net_all_mon",
    }
    linker = ObservationLinker(
        model_name=parameters.model_name,
        variables=parameters.variables,
        obs_sets=parameters.obs_sets,
        ts_dir_ref_source=parameters.obs_ts,
        obstmp_dir=parameters.obstmp_dir,
        altobs_dic=alt_obs_map,
        obs_alias_file="reference_alias.json",
    )
    linker.link_obs_data()
    linker.process_derived_variables()


def _get_args() -> LinkObservationParameters:
    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="zi-pcmdi-link-observation <args>"
    )
    parser.add_argument("--model_name_ref", type=str)
    parser.add_argument("--tableID_ref", type=str)
    parser.add_argument("--vars", type=str)
    parser.add_argument("--obs_sets", type=str)
    parser.add_argument("--obs_ts", type=str)
    parser.add_argument("--obstmp_dir", type=str)

    # Ignore the first arg
    # (zi-pcmdi-link-observation)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    return LinkObservationParameters(vars(args))
