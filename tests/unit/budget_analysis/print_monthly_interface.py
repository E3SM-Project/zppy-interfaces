"""Print monthly normalized values for cpl and ocn interface comparison."""

import glob
import sys

import pandas as pd

sys.path.insert(0, ".")

from zppy_interfaces.budget_analysis.ingestion.cpl_parser import CplParser  # noqa: E402
from zppy_interfaces.budget_analysis.ingestion.ocn_parser import OcnParser  # noqa: E402
from zppy_interfaces.budget_analysis.normalization import normalize  # noqa: E402

LOG_PATH = "/pscratch/sd/c/chengzhu/zstash/archive/logs"
START_YEAR = 1
END_YEAR = 50

cpl = CplParser().parse_files(
    sorted(glob.glob(f"{LOG_PATH}/cpl.log.*.gz")), START_YEAR, END_YEAR
)
ocn = OcnParser().parse_files(
    sorted(glob.glob(f"{LOG_PATH}/ocn.log.*.gz")), START_YEAR, END_YEAR
)
df = normalize(pd.concat([cpl, ocn], ignore_index=True))

cpl_m = df[
    (df.source == "cpl")
    & (df.term == "*SUM*")
    & (df.component == "ocn")
    & (df.period == "monthly")
].sort_values("time")

ocn_m = df[
    (df.source == "ocn")
    & (df.term == "SUM VOLUME FLUXES")
    & (df.table_type == "flux")
    & (df.period == "monthly")
].sort_values("time")

print("=== CPL monthly ocn *SUM* ===")
print(cpl_m[["time", "normalized_value"]].to_string(index=False))
print(f"\n=== OCN monthly SUM VOLUME FLUXES ({len(ocn_m)} rows) ===")
print(ocn_m[["time", "normalized_value"]].to_string(index=False))
