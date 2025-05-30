import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


def find_latest_file_list(
    path: str,
    file_pattern: str,
    var_pattern=r"\.(\w+)\.\d{8}\.nc$",
    time_pattern=r"\.(\d{8})\.nc$",
) -> List[str]:
    """
    Find the latest NetCDF file for each variable in the directory based on timestamps in filenames.

    Args:
        path (str): Directory to search.
        file_pattern (str): Regex to search file lists.
        var_pattern (str): Regex to extract variable name.
        time_pattern (str): Regex to extract date.

    Returns:
        List[str]: List of file paths, one for each variable (latest by timestamp).
    """
    latest_files: Dict[str, Tuple[datetime, str]] = {}
    files = glob.glob(os.path.join(path, file_pattern))

    for f in files:
        fname = os.path.basename(f)
        var_match = re.search(var_pattern, fname)
        time_match = re.search(time_pattern, fname)

        if var_match and time_match:
            var = var_match.group(1)
            try:
                timestamp = datetime.strptime(time_match.group(1), "%Y%m%d")
            except ValueError:
                continue

            if var not in latest_files or timestamp > latest_files[var][0]:
                latest_files[var] = (timestamp, f)

    return [file for _, file in latest_files.values()]


def get_highlight_models(all_models, model_name):
    """
    Prioritize models containing 'e3sm' and then any additional specified models.

    Parameters:
        data_dict (dict): Dictionary with a 'model' key containing a list of model names.
        model_name (list): List of models to also highlight (after e3sm models).

    Returns:
        list: Ordered list of unique models to highlight.
    """
    highlight_model1 = []

    # First, collect all models that contain "e3sm" (case-insensitive)
    e3sm_models = [m for m in all_models if "e3sm" in m.lower()]

    # Then collect models in model_name that are not already in e3sm_models
    additional_models = [
        m for m in all_models if m in model_name and m not in e3sm_models
    ]

    # Combine both lists
    highlight_model1 = e3sm_models + additional_models

    return highlight_model1


def shift_row_to_bottom(df, index_to_shift):
    """
    Moves the specified row to the bottom of the DataFrame and resets the index.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        index_to_shift (int): The index of the row to move to the bottom.

    Returns:
        pd.DataFrame: A new DataFrame with the row moved to the bottom and index reset.
    """
    if index_to_shift not in df.index:
        raise IndexError(f"Index {index_to_shift} not found in DataFrame.")

    df_top = df.drop(index=index_to_shift)
    df_bottom = df.loc[[index_to_shift]]

    new_df = pd.concat([df_top, df_bottom], ignore_index=True)
    return new_df
