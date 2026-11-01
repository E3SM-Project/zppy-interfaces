# global_time_series: ocean plots fail with a final, shorter chunk

Reproducers for the `change_ohc` / `change_sea_level` failure found in the
[2026-10-30 Chrysalis test](https://github.com/E3SM-Project/zppy/discussions/885#discussioncomment-19304037):

```
ValueError: x and y must have same first dimension, but have shapes (10,) and (11,)
RuntimeError: Change in ocean heat content could not be plotted.
```

## Cause

With `years = "1985-1995"` and `ts_num_years = 5`:

- atm: `[ts] years = "1985:1995:5"` only produces the 5yr chunks 1985-1989
  and 1990-1994, so the shared x-axis `exp["annual"]["year"]` has 10 years.
- ocn: since #66, `create_ocean_ts.ocean_month` also writes the final chunk
  `mpaso.glb.199501-199512.nc`, into the same `5yr/` directory. The ohc/volume
  reader opens `5yr/*.nc`, so ohc and volume have 11 years.

`max_moc` is unaffected, since `getmoc` reads its own time axis.

The fix trims ohc and volume to the years on the shared x-axis
(`plots_original._years_subset`). The opposite case, where atm has the final
chunk but the ocean does not, already works, because `plot` shortens the
x-axis to the data (`shorten_year`).

## 1. Synthetic reproducer (seconds, no zppy)

Writes small NetCDF files laid out like the test run, then calls
`plots_original.process_data` and the two ocean plots.

```bash
# Bug: zppy-interfaces as installed in the env
python -I mre_synthetic.py
# Fix: import from this checkout instead
PYTHONPATH=$(git rev-parse --show-toplevel) python mre_synthetic.py
```

With the bug it prints `len(ohc) = 11` and raises `RuntimeError`. With the fix
it prints `len(ohc) = 10` and `change_ohc and change_sea_level plotted OK`.

## 2. zppy reproducer (Chrysalis, short debug-queue jobs)

Runs only `[global_time_series]` on forsyth2's 2026-10-30 comprehensive_v3
data: the atm glb ts, the MOC file, and the raw MPAS-O output through 1995.
`setup.sh` links that data into `WORK_DIR`, writes `OK` status files for the
`ts` / `mpas_analysis` dependencies, and renders `mre_zppy.cfg` from
`mre_zppy.cfg.template`. The cfg has two subtasks using the same env:

- `buggy`: zppy-interfaces as installed in the env
- `fixed`: the same env, with `PYTHONPATH` set to this checkout (`ZI_SRC`)

Settings and their defaults are in `env.sh`. With `ATM_TAIL=1`, `setup.sh`
also appends `mre_zppy_atm_tail.cfg.part`: a `[ts]` subtask with
`years = "1995-1995",` that produces the atm 1995 chunk, the way the
1985-1989 and 1990-1994 chunks were produced. Each setting gets its own
`WORK_DIR`.

```bash
ATM_TAIL=0 bash setup.sh      # or ATM_TAIL=1
zppy -c /lcrc/group/e3sm/${USER}/zppy_mre_gts_ocn_tail/mre_zppy.cfg
#       (ATM_TAIL=1: .../zppy_mre_gts_ocn_tail_atm_tail/mre_zppy.cfg)
# After the jobs finish:
ATM_TAIL=0 bash check.sh      # exits 0 if the outcome is as expected
```

`check.sh` reads the job logs, since `zi-global-time-series` exits 0 even when
a plot fails, so the jobs' `.status` files say `OK` either way. It then runs
`count_years.py`, which calls `plots_original.process_data` on the output with
each zppy-interfaces, and prints the years for the x-axis, ohc and volume.

Expected:

| `ATM_TAIL` | `buggy` (no fix) | `fixed` |
|---|---|---|
| 0 | fails: x-axis 10 years, ohc/volume 11 | passes: 10 years (1985-1994); the ocean's 1995 is dropped |
| 1 | passes: 11 years (1985-1995) | passes: 11 years (1985-1995) |

So to plot all of 1985-1995, `[ts]` must also produce the final, shorter
chunk, e.g. `years = "1985:1994:5", "1995-1995",`. A range that is a multiple
of `ts_num_years` (e.g. `years = "1995-2004"` with `ts_num_years = 5`) has no
final, shorter chunk, and works with or without the fix.

Keep this checkout on the branch under test until the jobs finish.
