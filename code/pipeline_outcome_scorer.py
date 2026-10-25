# Source: app/pipeline/outcome_scorer.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Outcome Severity Scorer — maps outc_cod to numeric severity scores.
"""
import pandas as pd

# Severity scoring per FDA outcome codes
SEVERITY_MAP = {
    "DE": 7,  # Death
    "LT": 6,  # Life-Threatening
    "HO": 5,  # Hospitalization
    "DS": 4,  # Disability
    "CA": 3,  # Congenital Anomaly
    "RI": 2,  # Required Intervention
    "OT": 1,  # Other Serious
}


def score_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add severity_score column based on outc_cod.
    Unknown codes default to 1 (OT-equivalent).
    """
    df = df.copy()
    df["outc_cod"] = df["outc_cod"].fillna("OT").str.strip().str.upper()
    df["severity_score"] = df["outc_cod"].map(SEVERITY_MAP).fillna(1).astype(int)
    return df
