#!/bin/bash
# Set up the zppy reproducer in WORK_DIR (rerun to start clean). See README.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
WORK_DIR=${WORK_DIR:-/lcrc/group/e3sm/${USER}/zppy_mre_gts_ocn_tail}
src=/lcrc/group/e3sm/ac.forsyth2/zppy_weekly_comprehensive_v3_output/zppy_main_branch_test_20261030_run1/v3.LR.historical_0051/post
post=${WORK_DIR}/output/v3.LR.historical_0051/post

rm -rf "${WORK_DIR}/output"
mkdir -p "${post}/atm/glb/ts/monthly" "${post}/analysis/mpas_analysis/cache/timeseries" "${post}/scripts"
ln -s "${src}/atm/glb/ts/monthly/5yr" "${post}/atm/glb/ts/monthly/5yr"
ln -s "${src}/analysis/mpas_analysis/cache/timeseries/moc" "${post}/analysis/mpas_analysis/cache/timeseries/moc"

# Mark the ts and mpas_analysis jobs global_time_series depends on as done
for dep in ts_atm_monthly_glb_1985-1989-0005 ts_atm_monthly_glb_1990-1994-0005 \
           mpas_analysis_ts_1985-1989_climo_1985-1989 mpas_analysis_ts_1985-1995_climo_1990-1995; do
  echo OK > "${post}/scripts/${dep}.status"
done

sed -e "s#@WORK_DIR@#${WORK_DIR}#" -e "s#@USER@#${USER}#" -e "s#@ZI_SRC@#$(cd "${here}/../.." && pwd)#" \
    "${here}/mre_zppy.cfg.template" > "${WORK_DIR}/mre_zppy.cfg"
echo "zppy -c ${WORK_DIR}/mre_zppy.cfg"
