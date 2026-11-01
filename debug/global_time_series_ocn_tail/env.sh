# Settings shared by setup.sh and check.sh (defaults are for Chrysalis).
#   ATM_TAIL      1: also produce the atm 1995 chunk with [ts] (see README.md)
#   WORK_DIR      where output and the rendered cfg go
#   WWW_DIR       where plots are published
#   ENV_COMMANDS  commands that activate a zppy-interfaces env
#   ZI_SRC        zppy-interfaces checkout used by the "fixed" subtask
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ATM_TAIL=${ATM_TAIL:-0}
if [ "${ATM_TAIL}" = 1 ]; then
  suffix=_atm_tail
else
  suffix=""
fi
WORK_DIR=${WORK_DIR:-/lcrc/group/e3sm/${USER}/zppy_mre_gts_ocn_tail${suffix}}
WWW_DIR=${WWW_DIR:-/lcrc/group/e3sm/public_html/diagnostic_output/${USER}/zppy_mre_gts_ocn_tail${suffix}}
ENV_COMMANDS=${ENV_COMMANDS:-source /home/ac.forsyth2/miniforge3/etc/profile.d/conda.sh; conda activate test-zi-main-20261030_run1}
ZI_SRC=${ZI_SRC:-$(cd "${here}/../.." && pwd)}
case_dir=${WORK_DIR}/output/v3.LR.historical_0051
