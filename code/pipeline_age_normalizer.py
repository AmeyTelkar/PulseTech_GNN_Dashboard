# Source: app/pipeline/age_normalizer.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Age Normalization — convert all age representations to decimal years
and assign ICH E11 pediatric age bands.
Extended: adds age_unknown flag for cases with no age information at all.
"""
import pandas as pd
import numpy as np

# Conversion factors: multiply raw age value by this to get years
AGE_COD_TO_YEARS = {
    "DEC": 10.0,
    "YR": 1.0,
    "MON": 1 / 12,
    "WK": 1 / 52.18,
    "DY": 1 / 365.25,
    "HR": 1 / 8766,
}

# Fallback midpoints when age_cod is missing but age_grp is available
AGE_GRP_MIDPOINT = {
    "N": 0.04,        # Neonate (~15 days)
    "NWT": 0.002,     # Preterm/Neonate
    "INF": 0.5,       # Infant (0-2)
    "I": 0.5,         # Infant alias
    "C": 6.0,         # Child (2-11)
    "T": 6.0,         # Child synonym
    "TER": 6.0,
    "CHD": 6.0,
    "A": 14.5,        # Adolescent
    "ADL": 30.0,      # Adult — will be excluded in pediatric filter
    "E": 70.0,        # Elderly
    "OT": None,
    "UNK": None,
}

# Age groups that indicate pediatric patients (for filter fallback)
PEDIATRIC_AGE_GRPS = {"N", "NWT", "INF", "I", "C", "T", "TER", "CHD", "A"}


def normalize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize age to decimal years. NEVER drops rows.
    - Uses age * AGE_COD_TO_YEARS[age_cod]
    - Falls back to AGE_GRP_MIDPOINT if age_years still null
    - Sets is_age_imputed = True for imputed values
    - Sets age_unknown = True when both age and age_grp are null
    """
    df = df.copy()
    df["age_raw"] = pd.to_numeric(df.get("age", pd.Series(dtype="float64")), errors="coerce")
    df["age_cod"] = df.get("age_cod", pd.Series(dtype=str)).fillna("").str.strip().str.upper()
    df["age_grp"] = df.get("age_grp", pd.Series(dtype=str)).fillna("").str.strip().str.upper()

    # Primary path: age * multiplier
    multiplier = df["age_cod"].map(AGE_COD_TO_YEARS)
    df["age_years"] = df["age_raw"] * multiplier
    df["is_age_imputed"] = False
    df["age_unknown"] = False

    # Sanity check: reject obviously invalid ages
    df.loc[df["age_years"] < 0, "age_years"] = np.nan
    df.loc[df["age_years"] > 150, "age_years"] = np.nan

    # Fallback: impute from age_grp midpoint where age_years is null
    mask_missing = df["age_years"].isna()
    midpoint_map = df.loc[mask_missing, "age_grp"].map(AGE_GRP_MIDPOINT)
    valid_midpoints = midpoint_map.notna()
    df.loc[mask_missing & valid_midpoints, "age_years"] = midpoint_map[valid_midpoints]
    df.loc[mask_missing & valid_midpoints, "is_age_imputed"] = True

    # Flag truly unknown age (both age and age_grp null/empty)
    both_null = df["age_years"].isna() & (df["age_grp"].isin({"", "OT", "UNK"}) | df["age_grp"].isna())
    df.loc[both_null, "age_unknown"] = True

    return df


def assign_ich_e11_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ICH E11 pediatric age bands:
    - NEONATE: 0–27 days (<28/365.25 years)
    - INFANT: 28 days to <2 years
    - CHILD: 2 to <12 years
    - ADOLESCENT: 12 to <18 years
    Returns None for adults (≥18) or truly unknown age.
    """
    df = df.copy()

    def band(row):
        y = row.get("age_years")
        grp = str(row.get("age_grp", "")).upper()

        if y is None or (isinstance(y, float) and np.isnan(y)):
            # Try to assign from age_grp directly
            if grp in ("N", "NWT"):
                return "NEONATE"
            if grp in ("INF", "I"):
                return "INFANT"
            if grp in ("C", "CHD", "TER", "T"):
                return "CHILD"
            if grp == "A":
                return "ADOLESCENT"
            return None

        y = float(y)
        if y < 0:
            return None
        if y < 28 / 365.25:
            return "NEONATE"
        if y < 2:
            return "INFANT"
        if y < 12:
            return "CHILD"
        if y < 18:
            return "ADOLESCENT"
        return None  # Adult — will be dropped by pediatric filter

    df["age_group"] = df.apply(band, axis=1)
    return df
