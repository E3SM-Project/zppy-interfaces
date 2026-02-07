"""Base class for log file parsers."""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class BaseParser(ABC):
    """All parsers return a tidy event table DataFrame."""

    @abstractmethod
    def parse_files(
        self, log_files: List[str], start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Parse log files and return a tidy event table."""
        ...
