#!/bin/bash
# Pass/fail for the zppy reproducer. zi-global-time-series exits 0 even when
# a plot fails, so read the job logs rather than the .status files, then
# count the years plotted (count_years.py) with each zppy-interfaces.
# Exits 0 only if the outcome matches the expectation for ATM_TAIL:
#   ATM_TAIL=0: buggy fails; fixed passes, plotting 10 years (1985-1994)
#   ATM_TAIL=1: both pass, plotting 11 years (1985-1995)
# Settings and their defaults: see env.sh.
source "$(dirname "$0")/env.sh"
cd "${case_dir}/post/scripts" || exit 1

echo "== status (ATM_TAIL=${ATM_TAIL})"
grep -H . ./*.status | grep -v "OK$" || echo "all OK"

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

declare -A years
for sub in buggy fixed; do
  pythonpath=""
  [ "${sub}" = fixed ] && pythonpath=${ZI_SRC}
  echo "== years read by ${sub} (zppy-interfaces from ${pythonpath:-the env})"
  out=$(bash -c "${ENV_COMMANDS}; PYTHONPATH=${pythonpath} python ${here}/count_years.py ${case_dir}" 2>&1 \
        | grep -E "^(x-axis|ohc|volume)")
  echo "${out}"
  years[${sub}]=$(echo "${out}" | grep -c "11 years (1985-1995)")
done

if [ "${ATM_TAIL}" = 1 ]; then
  echo "== expected: buggy and fixed both pass, x-axis/ohc/volume all 11 years (1985-1995)"
  [ "${failed[buggy]}" = no ] && [ "${failed[fixed]}" = no ] && [ "${years[buggy]}" = 3 ] && [ "${years[fixed]}" = 3 ]
else
  echo "== expected: buggy FAILED with ['change_ohc', 'change_sea_level']; fixed passes with 10 years (1985-1994)"
  [ "${failed[buggy]}" = yes ] && [ "${failed[fixed]}" = no ] && [ "${years[fixed]}" = 0 ]
fi
