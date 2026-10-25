# Source: app/pipeline/pediatric_filter.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Pediatric Filter — inclusive approach, never drops rows due to missing dates.
"""
import pandas as pd

PEDIATRIC_AGE_GROUPS = {"NEONATE", "INFANT", "CHILD", "ADOLESCENT"}
ADULT_AGE_GRP_CODES = {"ADL", "E", "OT"}  # confirmed adult codes in FAERS


def filter_pediatric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep row if ANY of these is true:
    1. age_years is known AND < 18
    2. age_years is null BUT age_group is a pediatric band
    3. age_years is null AND age_group is null AND age_grp is NOT a confirmed adult code

    NEVER drop a row just because a date is null.
    """
    mask_known_pediatric = df["age_years"].notna() & (df["age_years"] < 18)

    mask_group_pediatric = df["age_years"].isna() & df["age_group"].isin(PEDIATRIC_AGE_GROUPS)

    age_grp_col = df.get("age_grp", pd.Series(dtype=str)).fillna("").str.upper()
    mask_unknown_not_adult = (
        df["age_years"].isna()
        & df["age_group"].isna()
        & ~age_grp_col.isin(ADULT_AGE_GRP_CODES)
    )

    return df[mask_known_pediatric | mask_group_pediatric | mask_unknown_not_adult].copy()
