import glob
import json
import os
import re

from zppy_interfaces.multi_utils.logger import _setup_custom_logger
from zppy_interfaces.pcmdi_diags.synthetic_plots.utils import find_latest_file_list

logger = _setup_custom_logger(__name__)


class EnsoMetricsReader:
    def __init__(self, parameter, metric, stat):
        self.parameter = parameter
        self.metric = metric
        self.stat = stat
        self.metric_dict = self.parameter["diag_vars"][stat]
        self.metrics_collections = self.metric_dict["collection"]
        self.mips = [self.parameter["cmip_name"].split(".")[0]] + self.parameter[
            "model_name"
        ]
        self.dict_json_path = {}

        self.var_pattern = re.compile(r"\.(\w+)\..*\.v(\d{8})\.json$")
        self.time_pattern = re.compile(r"\.v(\d{8})\.json$")

    def run(self):
        """Collect paths to ENSO metrics JSON files and return the mapping."""
        for mip in self.mips:
            self.dict_json_path[mip] = {}
            for metrics_collection in self.metrics_collections:
                if "cmip" in mip:
                    self.dict_json_path[mip][metrics_collection] = (
                        self._get_cmip_json_path(mip, metrics_collection)
                    )
                else:
                    self.dict_json_path[mip][metrics_collection] = (
                        self._get_test_json_path(mip, metrics_collection)
                    )

            if len(self.dict_json_path[mip]) < 1:
                raise FileNotFoundError(
                    f"No Synthetic ENSO Metrics Data for {mip}, aborting..."
                )

        return self.dict_json_path

    def _get_cmip_json_path(self, mip, metrics_collection):
        path = os.path.join(
            self.parameter["cmip_path"],
            self.parameter["cmip_name"].split(".")[0],
            self.parameter["cmip_name"].split(".")[1],
            self.parameter["cmip_name"].split(".")[2],
            metrics_collection,
            f"{mip.lower()}_{self.parameter['cmip_name'].split('.')[1]}_{metrics_collection}_*.json",
        )
        matches = glob.glob(path)
        if not matches:
            raise FileNotFoundError(
                f"CMIP metrics file not found for {mip} and {metrics_collection}"
            )
        return matches[0]

    def _get_test_json_path(self, mip, metrics_collection):
        for i, model_name in enumerate(self.parameter["model_name"]):
            model_path = self.parameter["test_path"].replace(
                "put_model_here", model_name
            )
            model_files = find_latest_file_list(
                path=f"{model_path}/{metrics_collection}",
                file_pattern="*.json",
                var_pattern=self.var_pattern,
                time_pattern=self.time_pattern,
            )
            logger.info(f"{model_path}/{metrics_collection}")
            if not model_files or not os.path.exists(model_files[0]):
                raise FileNotFoundError(
                    f"No Synthetic ENSO Metrics Data For {mip} {model_name}, Aborting."
                )

            for json_path in model_files:
                with open(json_path) as ff:
                    data_json = json.load(ff)

            old_key = list(data_json["RESULTS"]["model"].keys())[0]

            data_json["RESULTS"]["model"][mip] = data_json["RESULTS"]["model"].pop(
                old_key
            )

            with open(json_path, "w", encoding="utf8") as ff:
                json.dump(
                    data_json, ff, indent=4, separators=(",", ": "), sort_keys=True
                )

        return json_path
