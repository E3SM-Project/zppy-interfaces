import argparse
import glob
import json
import os
import re
import shutil
import sys
from typing import Dict, List

from pcmdi_metrics.io import xcdat_open

from zppy_interfaces.multi_utils.logger import _setup_child_logger, _setup_root_logger
from zppy_interfaces.pcmdi_diags.utils import ALT_OBS_MAP

# Set up the root logger and module level logger. The module level logger is
# a child of the root logger.
_setup_root_logger()
logger = _setup_child_logger(__name__)


# Classes #####################################################################
class LinkObservationParameters(object):
    def __init__(self, args: Dict[str, str]):
        self.model_name: str = f"{args['model_name_ref']}.{args['tableID_ref']}"
        self.variables: List[str] = args["vars"].split(",")
        self.obs_sets: List[str] = args["obs_sets"].split(",")
        self.obs_ts: str = args["obs_ts"]
        self.obstmp_dir: str = args["obstmp_dir"]


class ObservationLinker:
    def __init__(
        self,
        model_name,
        variables,
        obs_sets,
        ts_dir_ref_source,
        obstmp_dir,
        obs_alias_file,
        altobs_dic,
    ):
        self.model_name = model_name
        self.variables = variables
        self.obs_sets = obs_sets
        self.ts_dir_ref_source = ts_dir_ref_source
        self.obstmp_dir = obstmp_dir
        self.obs_dic = json.load(open(obs_alias_file))
        self.altobs_dic = altobs_dic

    def _resolve_obs_file(self, varin, obsid):
        if varin not in self.obs_dic or obsid not in self.obs_dic[varin]:
            print(f"[Warning] No alias found for variable '{varin}' in obsid '{obsid}'")
            return None, None

        obsname = self.obs_dic[varin][obsid]
        obsstr = (
            obsname.replace("_", "*").replace("-", "*")
            if "ceres_ebaf" in obsname
            else obsname
        )
        pattern = os.path.join(self.ts_dir_ref_source, obsstr, f"{varin}_*.nc")
        fpaths = sorted(glob.glob(pattern))

        if fpaths and os.path.exists(fpaths[0]):
            return fpaths[0], varin

        # Try altobs mapping
        if varin in self.altobs_dic:
            alt_var = self.altobs_dic[varin]
            pattern_alt = os.path.join(
                self.ts_dir_ref_source, obsstr, f"{alt_var}_*.nc"
            )
            fpaths = sorted(glob.glob(pattern_alt))
            if fpaths and os.path.exists(fpaths[0]):
                return fpaths[0], alt_var

        print(f"[Warning] Observation file not found for {varin} ({obsid})")
        return None, None

    def link_obs_data(self):
        for i, vv in enumerate(self.variables):
            varin = re.split(r"_|-", vv)[0] if "_" in vv or "-" in vv else vv
            if len(self.obs_sets) > 1 and len(self.obs_sets) == len(self.variables):
                obsid = self.obs_sets[i]
            else:
                obsid = self.obs_sets[0]

            filepath, resolved_var = self._resolve_obs_file(varin, obsid)
            if filepath:
                template = os.path.basename(filepath)
                parts = template.replace(".nc", "").split("_")
                if len(parts) < 3:
                    print(f"[Error] Unexpected filename format: {template}")
                    continue
                yms, yme = parts[-2][:6], parts[-1][:6]
                obsname = self.obs_dic[varin][obsid].replace(".", "_")
                out = os.path.join(
                    self.obstmp_dir,
                    f"{self.model_name.replace('put_model_here', obsname)}.{varin}.{yms}-{yme}.nc",
                )

                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    if resolved_var == varin:
                        os.symlink(filepath, out)
                        print(f"[Info] Linked {resolved_var} → {out}")
                    else:
                        ds = xcdat_open(filepath)
                        ds = ds.rename({resolved_var: varin})
                        ds.to_netcdf(out)
                        print(
                            f"[Info] Renamed and saved {resolved_var} as {varin} → {out}"
                        )
                else:
                    print(f"[Info] Skipping existing file: {out}")

    def derive_var(self, vout, var_dic):
        template = None
        out = None
        ds_out = None

        for i, (var, scale) in enumerate(var_dic.items()):
            fpaths = sorted(glob.glob(os.path.join(self.obstmp_dir, f"*.{var}.*.nc")))
            if not fpaths:
                print(
                    f"[Warning] No file found for base variable '{var}' needed to derive '{vout}'"
                )
                continue

            ds = xcdat_open(fpaths[0])
            if i == 0:
                template = os.path.basename(fpaths[0])
                out = os.path.join(
                    self.obstmp_dir, template.replace(f".{var}.", f".{vout}.")
                )
                shutil.copy(fpaths[0], out)
                ds_out = ds.rename_vars({var: vout})
                ds_out[vout] = ds_out[vout] * scale
            else:
                ds_other = xcdat_open(fpaths[0])
                if ds_out:
                    ds_out[vout] = ds_out[vout] + ds_other[var] * scale
                else:
                    raise ValueError("ds_out is None")

        if template and ds_out:
            ds_out.to_netcdf(out)
            print(f"[Info] Derived variable '{vout}' written to {out}")

    def process_derived_variables(self):
        for vv in self.variables:
            if vv in ["rltcre", "rstcre"]:
                fpaths = sorted(glob.glob(os.path.join(self.obstmp_dir, f"*{vv}_*.nc")))
                if not fpaths:
                    if vv == "rstcre":
                        self.derive_var("rstcre", {"rsutcs": 1, "rsut": -1})
                    elif vv == "rltcre":
                        self.derive_var("rltcre", {"rlutcs": 1, "rlut": -1})


# Functions ###################################################################
def main():
    parameters: LinkObservationParameters = _get_args()
    linker = ObservationLinker(
        model_name=parameters.model_name,
        variables=parameters.variables,
        obs_sets=parameters.obs_sets,
        ts_dir_ref_source=parameters.obs_ts,
        obstmp_dir=parameters.obstmp_dir,
        altobs_dic=ALT_OBS_MAP,
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
    parser.add_argument("--debug", type=str)

    # Ignore the first arg
    # (zi-pcmdi-link-observation)
    args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    if args.debug and args.debug.lower() == "true":
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    return LinkObservationParameters(vars(args))
