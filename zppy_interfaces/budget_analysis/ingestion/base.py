"""Base class for log file parsers."""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd

VALID_FREQUENCIES = ("monthly", "annual")


class BaseParser(ABC):
    """All parsers return a tidy event table DataFrame.

    Parameters
    ----------
    frequency : str
        Temporal granularity of the output: ``"monthly"`` (default)
        or ``"daily"``.  For parsers whose log data is already at a
        fixed period (e.g. coupler annual/monthly), the frequency
        selects which records to keep.  For the atm parser it controls
        the groupby aggregation window.
    """

    def __init__(self, frequency: str = "annual") -> None:
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {VALID_FREQUENCIES}, got {frequency!r}"
            )
        self.frequency = frequency

    @abstractmethod
    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Parse log files and return a tidy event table."""
        ...
