"""Column schema for the tidy budget event table."""

from typing import List

import pandas as pd

# Column name constants
COL_TIME = "time"  # float: year (annual) or year + (month-0.5)/12 (monthly)
COL_COMPONENT = (
    "component"  # str: "atm", "lnd", "rof", "ocn", "ice_nh", "ice_sh", "glc", "*SUM*"
)
COL_QUANTITY = "quantity"  # str: "water" (later "heat", "carbon")
COL_TERM = "term"  # str: flux or state term name
COL_VALUE = "value"  # float64: raw value in original units
COL_UNITS = "units"  # str: original units string
COL_SOURCE = "source"  # str: "cpl", "lnd", etc.
COL_PERIOD = "period"  # str: "annual" or "monthly"
COL_TABLE_TYPE = "table_type"  # str: "flux", "flux_integrated", "state"

COLUMNS: List[str] = [
    COL_TIME,
    COL_COMPONENT,
    COL_QUANTITY,
    COL_TERM,
    COL_VALUE,
    COL_UNITS,
    COL_SOURCE,
    COL_PERIOD,
    COL_TABLE_TYPE,
]


def empty_event_table() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=COLUMNS)
