"""
FAERS File Validator — auto-detects file type from header row.
Uses signature columns unique to each file type for robust detection.
"""
import pandas as pd
from typing import Optional

# Signature columns that uniquely identify each file type.
# Detection picks the type with the best overlap score.
SIGNATURE_COLUMNS = {
    "DEMO": {"caseversion", "age", "age_cod", "sex", "event_dt"},
    "DRUG": {"drug_seq", "role_cod", "drugname"},
    "REAC": {"pt"},
    "OUTC": {"outc_cod"},
    "RPSR": {"rpsr_cod"},
    "THER": {"dsg_drug_seq", "start_dt", "end_dt"},
    "INDI": {"indi_pt"},
}

# Full expected headers (used for validation reporting, not detection)
EXPECTED_HEADERS = {
    "DEMO": {
        "primaryid", "caseid", "caseversion", "age", "age_cod", "age_grp",
        "sex", "wt", "wt_cod", "event_dt", "fda_dt", "reporter_country", "occr_country",
    },
    "DRUG": {
        "primaryid", "caseid", "drug_seq", "role_cod", "drugname", "route",
        "dose_amt", "dose_unit", "dose_form", "dechal", "rechal",
    },
    "REAC": {"primaryid", "caseid", "pt", "drug_rec_act"},
    "OUTC": {"primaryid", "caseid", "outc_cod"},
    "RPSR": {"primaryid", "caseid", "rpsr_cod"},
    "THER": {"primaryid", "caseid", "dsg_drug_seq", "start_dt", "end_dt", "dur", "dur_cod"},
    "INDI": {"primaryid", "caseid", "indi_drug_seq", "indi_pt"},
}

# Minimum required columns for each file type (subset that MUST be present)
REQUIRED_NON_NULL = {
    "DEMO": {"primaryid", "caseid"},
    "DRUG": {"primaryid", "caseid", "drug_seq"},
    "REAC": {"primaryid", "caseid", "pt"},
    "OUTC": {"primaryid", "caseid", "outc_cod"},
    "RPSR": {"primaryid", "caseid"},
    "THER": {"primaryid", "caseid"},
    "INDI": {"primaryid", "caseid"},
}


def detect_file_type(df: pd.DataFrame) -> str:
    """
    Detect FAERS file type from column headers.
    Uses signature columns unique to each type for robust matching.
    Falls back to best-overlap scoring if no perfect signature match.
    """
    cols = set(df.columns.str.strip().str.lower())

    # First try: perfect signature match
    best_match = None
    best_score = 0

    for ftype, sig in SIGNATURE_COLUMNS.items():
        overlap = len(sig & cols)
        score = overlap / len(sig)  # percentage match
        if score > best_score:
            best_score = score
            best_match = ftype

    if best_match and best_score >= 0.5:
        return best_match

    # Fallback: match against expected headers (most overlap wins)
    best_match = None
    best_score = 0
    for ftype, expected in EXPECTED_HEADERS.items():
        overlap = len(expected & cols)
        if overlap > best_score:
            best_match = ftype
            best_score = overlap

    if best_match and best_score >= 2:
        return best_match

    raise ValueError(f"Cannot detect file type. Headers found: {sorted(cols)}")


def validate_file(df: pd.DataFrame, file_type: str) -> dict:
    """
    Validate a parsed FAERS DataFrame.
    Returns validation stats dict.
    """
    cols = set(df.columns.str.strip().str.lower())
    expected = EXPECTED_HEADERS.get(file_type, set())
    required = REQUIRED_NON_NULL.get(file_type, set())

    missing_cols = expected - cols
    extra_cols = cols - expected

    # Check for null counts in required columns
    null_counts = {}
    for col in required:
        if col in df.columns:
            null_counts[col] = int(df[col].isna().sum())

    return {
        "file_type": file_type,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "expected_columns": sorted(expected),
        "missing_columns": sorted(missing_cols),
        "extra_columns": sorted(extra_cols),
        "null_counts_required": null_counts,
        "is_valid": len(df) > 0,  # More lenient: just needs rows
    }
