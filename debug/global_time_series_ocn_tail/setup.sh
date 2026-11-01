#!/bin/bash
# Set up the zppy reproducer in WORK_DIR: link forsyth2's 2026-10-30
# comprehensive_v3 data, mark global_time_series' dependencies as done,
# and render mre_zppy.cfg. Safe to rerun; WORK_DIR/output is recreated.
# Settings and their defaults: see env.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

src=/lcrc/group/e3sm/ac.forsyth2/zppy_weekly_comprehensive_v3_output/zppy_main_branch_test_20261030_run1/v3.LR.historical_0051/post
post=${case_dir}/post

# Start clean, so the ocean tail chunk is regenerated on every run
rm -rf "${WORK_DIR}/output"

mkdir -p "${post}/atm/glb/ts/monthly" "${post}/analysis/mpas_analysis/cache/timeseries/moc" "${post}/scripts"
ln -s "${src}/atm/glb/ts/monthly/5yr" "${post}/atm/glb/ts/monthly/5yr"
ln -s "${src}/analysis/mpas_analysis/cache/timeseries/moc/mocTimeSeries_1985-1995.nc" \
      "${post}/analysis/mpas_analysis/cache/timeseries/moc/mocTimeSeries_1985-1995.nc"

# Dependencies zppy computes for years = "1985-1995", ts_num_years = 5,
# ts_years = "1985-1989", "1985-1995", climo_years = "1985-1989", "1990-1995"
for dep in \
    ts_atm_monthly_glb_1985-1989-0005 \
    ts_atm_monthly_glb_1990-1994-0005 \
    mpas_analysis_ts_1985-1989_climo_1985-1989 \
    mpas_analysis_ts_1985-1995_climo_1990-1995; do
  echo "OK" > "${post}/scripts/${dep}.status"
done

sed -e "s#@WORK_DIR@#${WORK_DIR}#g" \
    -e "s#@WWW_DIR@#${WWW_DIR}#g" \
    -e "s#@ENV_COMMANDS@#${ENV_COMMANDS}#g" \
    -e "s#@ZI_SRC@#${ZI_SRC}#g" \
    "${here}/mre_zppy.cfg.template" > "${WORK_DIR}/mre_zppy.cfg"
if [ "${ATM_TAIL}" = 1 ]; then
  cat "${here}/mre_zppy_atm_tail.cfg.part" >> "${WORK_DIR}/mre_zppy.cfg"
fi

echo "ATM_TAIL=${ATM_TAIL}; atm glb ts files: $(ls "${post}/atm/glb/ts/monthly/5yr/" | grep -c '\.nc$')"
echo "fixed subtask imports zppy-interfaces from: ${ZI_SRC} ($(git -C "${ZI_SRC}" rev-parse --abbrev-ref HEAD))"
echo "Next: zppy -c ${WORK_DIR}/mre_zppy.cfg"
echo "Then, after the jobs finish: ATM_TAIL=${ATM_TAIL} bash ${here}/check.sh"
