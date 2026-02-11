"""Quick test script for AtmParser on a sample atm.log file."""

import sys

from zppy_interfaces.budget_analysis.ingestion.atm_parser import AtmParser

LOG_FILE = (
    "/pscratch/sd/e/e3smtest/e3sm_scratch/pm-cpu/"
    "SMS.ne4pg2_oQU480.F2010.pm-cpu_intel.eam-thetahy_ftype2_energy"
    ".C.JNextIntegration20260210_205258/run/"
    "atm.log.48753126.260210-224733.gz"
)


def main():
    parser = AtmParser()
    log_files = [LOG_FILE]

    # --- Raw per-step data ---
    nstep_te, flux_diag = parser.parse_raw(log_files)

    print("=== nstep_te (energy fixer) ===")
    print(f"Shape: {nstep_te.shape}")
    print(f"Columns: {list(nstep_te.columns)}")
    print(nstep_te.head(3))
    print("...")
    print(nstep_te.tail(3))
    print()

    print("=== flux_diag (water/energy diagnostics) ===")
    print(f"Shape: {flux_diag.shape}")
    print(f"Columns: {list(flux_diag.columns)}")
    print(flux_diag.head(3))
    print("...")
    print(flux_diag.tail(3))
    print()

    # --- Validation ---
    print("=== Validation ===")
    assert (
        nstep_te.shape[0] == 122
    ), f"Expected 122 nstep_te rows, got {nstep_te.shape[0]}"
    print(f"  nstep_te rows: {nstep_te.shape[0]} (expected 122)")

    assert (
        flux_diag.shape[0] == 120
    ), f"Expected 120 flux_diag rows, got {flux_diag.shape[0]}"
    print(f"  flux_diag rows: {flux_diag.shape[0]} (expected 120)")

    tw_first = flux_diag["tw"].iloc[0]
    tw_last = flux_diag["tw"].iloc[-1]
    print(f"  W(n=1)  = {tw_first:.6f} kg/m2 (expect ~25.303)")
    print(f"  W(n=120)= {tw_last:.6f} kg/m2 (expect ~24.949)")

    e_diff_max = flux_diag["e_diff"].abs().max()
    print(f"  max |E difference| = {e_diff_max:.3e} W/m2")

    # Check date tracking
    print(
        f"  Date range: year {nstep_te['year'].min()}-{nstep_te['year'].max()}, "
        f"month {nstep_te['month'].min()}-{nstep_te['month'].max()}, "
        f"day {nstep_te['day'].min()}-{nstep_te['day'].max()}"
    )
    print()

    # --- Tidy event table ---
    events = parser.parse_files(log_files, 1, 1)
    print("=== Tidy event table ===")
    print(f"Shape: {events.shape}")
    print(events.to_string())

    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main() or 0)
