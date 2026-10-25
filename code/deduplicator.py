"""
DEMO Deduplication + Drug Role Cleaning.
- Within-quarter: caseid + MAX(caseversion)
- Tie-break 1: prefer i_f_code='F' (Follow-up) over 'I' (Initial)
- Tie-break 2: max fda_dt
- Tie-break 3: max primaryid
- Cross-quarter: same logic, applied after stacking multiple quarters
- Drug cleaning: remove role_cod='DN' (Definitively Not suspect)
"""
import pandas as pd

# Valid FAERS role codes for drug analysis
VALID_ROLE_CODES = {"PS", "SS", "C", "I"}


def deduplicate_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    FDA-compliant dedup: keep max caseversion per caseid.
    Tie-break 1: prefer i_f_code='F' (Follow-up) over 'I' (Initial).
    Tie-break 2: max fda_dt.
    Tie-break 3: max primaryid.
    """
    df = df.copy()

    df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce")
    df = df.dropna(subset=["caseid"])

    df["_cv"] = pd.to_numeric(df["caseversion"], errors="coerce").fillna(0)
    _ifc_series = df["i_f_code"].fillna("") if "i_f_code" in df.columns else pd.Series("", index=df.index)
    df["_ifc"] = (_ifc_series == "F").astype(int)
    df["_fdt"] = pd.to_datetime(df["fda_dt"], format="%Y%m%d", errors="coerce")
    df["_pid"] = pd.to_numeric(df["primaryid"], errors="coerce").fillna(0)

    df = (
        df.sort_values(
            ["_cv", "_ifc", "_fdt", "_pid"],
            ascending=[False, False, False, False],
        )
        .groupby("caseid", sort=False)
        .first()
        .reset_index()
    )

    df = df.drop(columns=["_cv", "_ifc", "_fdt", "_pid"])

    # Ensure primaryid is unique
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df = df.dropna(subset=["primaryid"])
    df = df.drop_duplicates(subset=["primaryid"], keep="first")

    return df.reset_index(drop=True)


def deduplicate_across_quarters(demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Called AFTER stacking multiple quarter DataFrames.
    Same logic as within-quarter but operates on combined data.
    """
    return deduplicate_demo(demo_df)


def clean_drug_roles(drug_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove role_cod='DN' (Definitively Not suspect).
    Found in real data: 14 rows in 25Q1. Must be removed before any analysis.
    """
    drug_df = drug_df.copy()
    before = len(drug_df)
    drug_df = drug_df[
        drug_df["role_cod"].fillna("").str.upper().isin(VALID_ROLE_CODES)
    ].copy()
    removed = before - len(drug_df)
    if removed > 0:
        print(f"  [DRUG] role_cod='DN' removed: {removed} rows")
    return drug_df


def run_cross_quarter_dedup(new_quarter: str) -> dict:
    """
    After a new quarter is committed to PostgreSQL, find duplicate caseids
    that appear in BOTH the new quarter AND any previous quarter.
    Keep the row from the newer quarter (higher caseversion or newer fda_dt).
    Soft-delete the older row by marking is_superseded = TRUE.

    This preserves ALL data but flags which rows are active vs superseded.
    Signal detection queries only use is_superseded = FALSE rows.

    Returns: {'duplicates_found': N, 'superseded': N}
    """
    from sqlalchemy import text
    try:
        from app.database import get_sync_session
    except ImportError:
        from database import get_sync_session

    session = get_sync_session()
    try:
        # Find caseids present in both new quarter and any older quarter
        result = session.execute(text("""
            WITH new_cases AS (
                SELECT primaryid, caseid, caseversion
                FROM demographics
                WHERE source_quarter = :new_q AND is_superseded = FALSE
            ),
            old_cases AS (
                SELECT primaryid, caseid, caseversion, source_quarter
                FROM demographics
                WHERE source_quarter != :new_q AND is_superseded = FALSE
            ),
            duplicates AS (
                SELECT n.caseid,
                       n.primaryid AS new_pid, n.caseversion AS new_cv,
                       o.primaryid AS old_pid, o.caseversion AS old_cv,
                       o.source_quarter AS old_quarter
                FROM new_cases n
                INNER JOIN old_cases o ON n.caseid = o.caseid
            )
            SELECT * FROM duplicates
        """), {"new_q": new_quarter}).fetchall()

        duplicates_found = len(result)
        superseded_count = 0
        related_tables = ["drugs", "reactions", "outcomes", "therapy",
                          "indications", "report_sources"]

        for row in result:
            # Keep the row with higher caseversion; tie-break: prefer newer quarter
            if row.new_cv >= row.old_cv:
                pid_to_supersede = row.old_pid
            else:
                pid_to_supersede = row.new_pid

            # Supersede the demographics row
            session.execute(text(
                "UPDATE demographics SET is_superseded = TRUE WHERE primaryid = :pid"
            ), {"pid": pid_to_supersede})

            # Also supersede all related tables for that primaryid
            for table in related_tables:
                session.execute(text(
                    f"UPDATE {table} SET is_superseded = TRUE WHERE primaryid = :pid"
                ), {"pid": pid_to_supersede})

            superseded_count += 1

        session.commit()
        print(f"  [DEDUP] Cross-quarter: {duplicates_found} duplicates found, "
              f"{superseded_count} rows superseded")
        return {"duplicates_found": duplicates_found, "superseded": superseded_count}
    except Exception as e:
        session.rollback()
        print(f"  [DEDUP] Error: {e}")
        return {"duplicates_found": 0, "superseded": 0, "error": str(e)}
    finally:
        session.close()

