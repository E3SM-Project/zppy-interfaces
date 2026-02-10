"""Diagnose ocean closure: verify log-native term names are used correctly."""

import glob
import sys

import pandas as pd

sys.path.insert(0, ".")

from zppy_interfaces.budget_analysis.ingestion.ocn_parser import OcnParser
from zppy_interfaces.budget_analysis.normalization import normalize

LOG_PATH = "/pscratch/sd/c/chengzhu/zstash/archive/logs"
START_YEAR = 1
END_YEAR = 50

ocn = OcnParser().parse_files(
    sorted(glob.glob(f"{LOG_PATH}/ocn.log.*.gz")), START_YEAR, END_YEAR
)

print(f"Total ocean rows: {len(ocn)}")
print()

# --- Water ---
water = ocn[ocn["quantity"] == "water"]
print("=== Water term counts ===")
print(water["term"].value_counts().to_string())
print()

mc = water[water["term"] == "Mass change"]
print(f"=== 'Mass change' rows: {len(mc)} ===")
if not mc.empty:
    print(mc[["time", "value", "units", "table_type", "period"]].head(12).to_string(index=False))
else:
    print("  *** NOT FOUND ***")
print()

svf = water[water["term"] == "SUM VOLUME FLUXES"]
print(f"=== 'SUM VOLUME FLUXES' rows: {len(svf)} ===")
if not svf.empty:
    print(svf[["time", "value", "units", "table_type", "period"]].head(12).to_string(index=False))
else:
    print("  *** NOT FOUND ***")
print()

# --- Heat ---
heat = ocn[ocn["quantity"] == "heat"]
print("=== Heat term counts ===")
print(heat["term"].value_counts().to_string())
print()

ec = heat[heat["term"] == "Energy change"]
print(f"=== 'Energy change' rows: {len(ec)} ===")
if not ec.empty:
    print(ec[["time", "value", "units", "table_type", "period"]].head(12).to_string(index=False))
else:
    print("  *** NOT FOUND ***")
print()

shf = heat[heat["term"] == "SUM IMP+EXP HEAT FLUXES"]
print(f"=== 'SUM IMP+EXP HEAT FLUXES' rows: {len(shf)} ===")
if not shf.empty:
    print(shf[["time", "value", "units", "table_type", "period"]].head(12).to_string(index=False))
else:
    print("  *** NOT FOUND ***")
print()

# --- Test closure checks ---
print("=== Testing OcnClosure checks ===")
df = normalize(ocn)

from zppy_interfaces.budget_analysis.checks import OcnClosure

for q in ["water", "heat"]:
    check = OcnClosure(quantity=q)
    result = check.evaluate(df)
    if result is not None:
        print(f"  {q} closure: {len(result.years)} years, max |residual| = {abs(result.residual).max():.2e}")
    else:
        print(f"  {q} closure: SKIPPED (missing data)")
