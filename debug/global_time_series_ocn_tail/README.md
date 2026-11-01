# global_time_series: ocean plots fail with a final, shorter chunk

From the [2026-10-30 Chrysalis test](https://github.com/E3SM-Project/zppy/discussions/885#discussioncomment-19304037):

```
ValueError: x and y must have same first dimension, but have shapes (10,) and (11,)
RuntimeError: Change in ocean heat content could not be plotted.
```

**Cause:** with `years = "1985-1995"` and `ts_num_years = 5`, `[ts] years = "1985:1995:5"`
only makes the atm chunks 1985-1989 and 1990-1994 (10 years). Since #66,
`create_ocean_ts` also writes the ocean chunk 1995, so ohc and volume have 11 years.

**Fix:** `plots_original._years_subset` trims ohc and volume to the atm years.

The fix is not needed if `[ts]` produces every year `[global_time_series]`
requests, e.g. `[ts] years = "1985:1994:5", "1995-1995",`, or a range that is a
multiple of `ts_num_years`.

## Synthetic reproducer (seconds)

```bash
python -I mre_synthetic.py                                         # bug: years: atm 10, ohc 11, then RuntimeError
PYTHONPATH=$(git rev-parse --show-toplevel) python mre_synthetic.py  # fix: years: atm 10, ohc 10, change_ohc plotted OK
```

## zppy reproducer (Chrysalis)

Runs only `[global_time_series]` on forsyth2's 2026-10-30 data, twice:
`buggy` uses zppy-interfaces from the env, `fixed` uses this checkout.

```bash
bash setup.sh   # prints the zppy command to run
# After both jobs finish (zi-global-time-series exits 0 even when plots fail):
grep -H "could not be generated" /lcrc/group/e3sm/${USER}/zppy_mre_gts_ocn_tail/output/*/post/scripts/global_time_series_*.o*
```

Expected: only `buggy` is listed, with `['change_ohc', 'change_sea_level']`.
`fixed` plots 1985-1994. Uncomment the `[ts]` block in the cfg to also make
the atm 1995 chunk: then neither is listed and both plot 1985-1995.
