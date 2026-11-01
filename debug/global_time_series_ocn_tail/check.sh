#!/bin/bash
# Pass/fail for the zppy reproducer. zi-global-time-series exits 0 even when
# a plot fails, so read the job logs rather than the .status files.
# Exits 0 only if "buggy" reproduces the failure and "fixed" does not.
WORK_DIR=${WORK_DIR:-/lcrc/group/e3sm/${USER}/zppy_mre_gts_ocn_tail}
scripts=${WORK_DIR}/output/v3.LR.historical_0051/post/scripts
cd "${scripts}" || exit 1

echo "== status"
grep -H . global_time_series_*.status

declare -A failed
for sub in buggy fixed; do
  log=$(ls -t global_time_series_${sub}_1985-1995.o* 2>/dev/null | head -n 1)
  echo "== ${sub}: ${log:-no log yet}"
  if [ -z "${log}" ]; then
    failed[${sub}]=unknown
    continue
  fi
  grep -m1 "zppy_interfaces/__init__.py" "${log}"
  grep "x and y must have same first dimension" "${log}" | sort | uniq -c
  if grep -q "plots could not be generated successfully" "${log}"; then
    grep "plots could not be generated successfully" "${log}"
    echo "RESULT: ${sub} FAILED (bug reproduced)"
    failed[${sub}]=yes
  else
    echo "RESULT: ${sub} all plots generated"
    failed[${sub}]=no
  fi
done

echo "== expected: buggy FAILED with ['change_ohc', 'change_sea_level']; fixed all plots generated"
[ "${failed[buggy]}" = yes ] && [ "${failed[fixed]}" = no ]
