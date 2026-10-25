# Source: app/pipeline/date_handler.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Date Handler — parse partial dates, NEVER drop rows for null dates.
"""
import pandas as pd
import numpy as np
from typing import List


def parse_faers_date(date_str: str) -> tuple:
    """
    Parse FAERS date strings that can be:
    - YYYYMMDD (8 chars)
    - YYYYMM (6 chars)
    - YYYY (4 chars)
    - Empty/null

    Returns (date_or_none, is_partial_bool)
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None, False

    s = str(date_str).strip().replace("-", "").replace("/", "")

    # Remove any non-numeric characters
    s = "".join(c for c in s if c.isdigit())

    if len(s) == 0:
        return None, False

    try:
        if len(s) >= 8:
            # Full date: YYYYMMDD
            year = int(s[:4])
            month = max(1, min(12, int(s[4:6])))
            day = max(1, min(31, int(s[6:8])))
            return pd.Timestamp(year=year, month=month, day=day).date(), False
        elif len(s) >= 6:
            # Partial: YYYYMM → first of month
            year = int(s[:4])
            month = max(1, min(12, int(s[4:6])))
            return pd.Timestamp(year=year, month=month, day=1).date(), True
        elif len(s) >= 4:
            # Partial: YYYY → Jan 1
            year = int(s[:4])
            return pd.Timestamp(year=year, month=1, day=1).date(), True
        else:
            return None, False
    except (ValueError, OverflowError):
        return None, False


def handle_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """
    Parse date columns from FAERS format. NEVER drops rows.
    Adds a {col}_partial boolean column for each date column.
    """
    df = df.copy()

    for col in date_cols:
        if col not in df.columns:
            continue

        parsed = df[col].apply(parse_faers_date)
        df[col] = parsed.apply(lambda x: x[0])

        partial_col = f"{col}_partial"
        df[partial_col] = parsed.apply(lambda x: x[1])

    return df
