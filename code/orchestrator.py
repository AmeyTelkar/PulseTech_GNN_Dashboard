"""
FAERS Pediatric ADR - Cleaning Pipeline V2 (Optimized)
Handles: DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI
Based on real column analysis of FAERS 2025Q1 data.
Self-contained - NO Mendeley dependency.
Optimized: parallel parsing, pyarrow engine, int64 sets.
"""
import io
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from app.pipeline.validator import detect_file_type, validate_file
    from app.pipeline.deduplicator import deduplicate_demo, clean_drug_roles
    from app.pipeline.age_normalizer import normalize_age, assign_ich_e11_group
    from app.pipeline.pediatric_filter import filter_pediatric
    from app.pipeline.drug_normalizer import normalize_drugs, normalize_route
    from app.pipeline.date_handler import handle_dates
    from app.pipeline.outcome_scorer import score_outcomes
except ImportError:
    from validator import detect_file_type, validate_file
    from deduplicator import deduplicate_demo, clean_drug_roles
    from pipeline_age_normalizer import normalize_age, assign_ich_e11_group
    from pipeline_pediatric_filter import filter_pediatric
    from pipeline_drug_normalizer import normalize_drugs, normalize_route
    from pipeline_date_handler import handle_dates
    from pipeline_outcome_scorer import score_outcomes

# ── Column drop lists (verified from real FAERS data scan) ──────────────
DEMO_DROP_COLS = [
    "auth_num",     # 90.7% null
    "lit_ref",      # 90.8% null
    "to_mfr",       # 96.7% null
    "mfr_num",      # manufacturer internal case number
    "mfr_sndr",     # manufacturer sender name
    "e_sub",        # internal FDA flag
    "rept_dt",      # redundant with fda_dt
    "init_fda_dt",  # redundant with fda_dt
    "mfr_dt",       # manufacturer date
]

DRUG_DROP_COLS = [
    "dose_vbm",      # validation flag for dose; redundant with val_vbm
    "cum_dose_chr",  # 98.3% null
    "cum_dose_unit", # 98.3% null
    "lot_num",       # 75.8% null; batch number
    "exp_dt",        # 99.9% null
    "nda_num",       # 67.5% null
    "rechal",        # 90.8% null
    "dose_freq",     # 79.1% null
]

REAC_DROP_COLS = [
    "drug_rec_act",  # 99.9% null in real data; completely useless
]

# Extended unknown indication strings (from real FAERS data)
UNKNOWN_INDI_STRINGS = {
    "product used for unknown indication",
    "drug use for unknown indication",
    "indication not on label",
    "unknown",
}

# Dechallenge encoding
DECHAL_MAP = {"Y": 1, "N": 0}


def _drop_useless_columns(df: pd.DataFrame, ftype: str) -> pd.DataFrame:
    """Drop columns verified as useless from real data scan."""
    drop_map = {
        "DEMO": DEMO_DROP_COLS,
        "DRUG": DRUG_DROP_COLS,
        "REAC": REAC_DROP_COLS,
    }
    cols_to_drop = [c for c in drop_map.get(ftype, []) if c in df.columns]
    if cols_to_drop:
        print(f"  [{ftype}] Dropping {len(cols_to_drop)} useless columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def _normalize_columns(df: pd.DataFrame, ftype: str) -> pd.DataFrame:
    """Normalize FAERS column names for cross-quarter compatibility."""
    df = df.copy()

    if ftype == "INDI":
        if "indi_drug_seq" in df.columns and "drug_seq" not in df.columns:
            df = df.rename(columns={"indi_drug_seq": "drug_seq"})

    if ftype == "REAC":
        if "pt" in df.columns and "pt_term" not in df.columns:
            df = df.rename(columns={"pt": "pt_term"})

    if ftype == "DRUG":
        if "dose_form" not in df.columns:
            df["dose_form"] = None
        if "drugname" in df.columns and "drugname_original" not in df.columns:
            df["drugname_original"] = df["drugname"].str.strip().str.upper().fillna("UNKNOWN")
        # prod_ai fallback: if drugname is blank, use prod_ai
        if "prod_ai" in df.columns and "drugname" in df.columns:
            blank_mask = df["drugname"].fillna("").str.strip().isin(["", "UNKNOWN"])
            df.loc[blank_mask, "drugname"] = df.loc[blank_mask, "prod_ai"]

    if ftype == "DEMO":
        if "reporter_country" not in df.columns and "reportercountry" in df.columns:
            df = df.rename(columns={"reportercountry": "reporter_country"})

    return df


def parse_faers_file(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a $-delimited FAERS file from raw bytes. Uses pyarrow engine if available."""
    try:
        # PyArrow engine is 2-5x faster for large files
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="$",
            dtype=str,
            encoding="utf-8",
            on_bad_lines="skip",
            engine="pyarrow",
        )
    except Exception:
        # Fallback to C engine
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="$",
            dtype=str,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False,
        )
    df.columns = df.columns.str.strip().str.lower()
    return df


def _parse_single_file(filename: str, raw: bytes) -> tuple:
    """Parse a single file - designed for parallel execution."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    t0 = time.perf_counter()
    df = parse_faers_file(raw)
    ftype = detect_file_type(df)
    df = _normalize_columns(df, ftype)
    df = _drop_useless_columns(df, ftype)
    vr = validate_file(df, ftype)
    elapsed = time.perf_counter() - t0
    print(f"  Parsed {ftype}: {len(df):,} rows x {len(df.columns)} cols ({elapsed:.1f}s)")
    return ftype, df, vr


def _filter_val_vbm(drug_df: pd.DataFrame) -> pd.DataFrame:
    """Remove val_vbm='2' rows (not FDA-validated)."""
    if "val_vbm" not in drug_df.columns:
        return drug_df
    before = len(drug_df)
    drug_df = drug_df[drug_df["val_vbm"].fillna("1") != "2"].copy()
    removed = before - len(drug_df)
    if removed > 0:
        print(f"  [DRUG] val_vbm='2' removed: {removed:,} rows")
    return drug_df


def _encode_dechal(drug_df: pd.DataFrame) -> pd.DataFrame:
    """Encode dechallenge: Y=1, N=0, U/null=None."""
    drug_df = drug_df.copy()
    if "dechal" in drug_df.columns:
        drug_df["dechal_encoded"] = drug_df["dechal"].str.upper().map(DECHAL_MAP)
    else:
        drug_df["dechal_encoded"] = None
    return drug_df


def _clean_indications(indi_df: pd.DataFrame) -> pd.DataFrame:
    """Null out unknown indication strings. Row stays — only value becomes NULL."""
    indi_df = indi_df.copy()
    if "indi_pt" in indi_df.columns:
        mask = (
            indi_df["indi_pt"]
            .fillna("")
            .str.lower()
            .str.strip()
            .isin(UNKNOWN_INDI_STRINGS)
        )
        indi_df.loc[mask, "indi_pt"] = None
        nulled = int(mask.sum())
        if nulled > 0:
            print(f"  [INDI] Unknown indications nulled: {nulled:,} rows")
    return indi_df


def _normalize_weight(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Convert weight to kg. LBS→KG conversion applied."""
    demo_df = demo_df.copy()
    if "wt" in demo_df.columns:
        demo_df["weight_kg"] = pd.to_numeric(demo_df["wt"], errors="coerce")
        if "wt_cod" in demo_df.columns:
            lbs_mask = demo_df["wt_cod"].fillna("").str.upper() == "LBS"
            demo_df.loc[lbs_mask, "weight_kg"] = demo_df.loc[lbs_mask, "weight_kg"] * 0.453592
    else:
        demo_df["weight_kg"] = np.nan
    return demo_df


def _run_pipeline_sync(files: dict, quarter: str, skip_rxnorm: bool) -> tuple:
    """
    Synchronous core of the 11-step cleaning pipeline.
    All CPU-bound pandas work runs here. Called via asyncio.to_thread().
    Optimized with parallel parsing and int64 primaryid sets.
    """
    pipeline_start = time.perf_counter()

    # -- PARALLEL Parse & detect (all files at once) ---
    parsed = {}
    validation_results = {}

    print(f"  [PIPELINE] Parsing {len(files)} files in parallel...")
    with ThreadPoolExecutor(max_workers=min(len(files), 4)) as executor:
        futures = {
            executor.submit(_parse_single_file, fn, raw): fn
            for fn, raw in files.items()
        }
        for future in as_completed(futures):
            ftype, df, vr = future.result()
            parsed[ftype] = df
            validation_results[ftype] = vr

    parse_time = time.perf_counter() - pipeline_start
    print(f"  [PIPELINE] All files parsed in {parse_time:.1f}s")

    # ── Step 1: Validate required files ───────────────────────────
    for required in ["DEMO", "DRUG", "REAC", "OUTC"]:
        if required not in parsed:
            raise ValueError(f"Missing required file: {required}. Upload DEMO, DRUG, REAC, OUTC at minimum.")

    # ── Step 2: Dedup DEMO (caseid + max caseversion + i_f_code tie-break)
    demo_raw_count = len(parsed["DEMO"])
    demo = deduplicate_demo(parsed["DEMO"])
    demo_dedup_count = len(demo)
    dedup_removed = demo_raw_count - demo_dedup_count
    print(f"  [DEDUP] {demo_raw_count:,} -> {demo_dedup_count:,} ({dedup_removed:,} removed)")

    # ── Step 3: Normalize age → age_years (decimal) ───────────────
    demo = normalize_age(demo)

    # ── Step 4: Assign ICH E11 age bands ──────────────────────────
    demo = assign_ich_e11_group(demo)

    # ── Step 5: Weight normalization ──────────────────────────────
    demo = _normalize_weight(demo)

    # ── Step 6: Pediatric filter ──────────────────────────────────
    demo = filter_pediatric(demo)
    pediatric_count = len(demo)
    print(f"  [PEDIATRIC] {demo_dedup_count:,} -> {pediatric_count:,} ({round(pediatric_count/max(demo_dedup_count,1)*100,1)}%)")

    # ── Step 7: Null-safe date handling ───────────────────────────
    date_cols = ["event_dt", "fda_dt"]
    if "mfr_dt" in demo.columns:
        date_cols.append("mfr_dt")
    demo = handle_dates(demo, date_cols=date_cols)

    if "THER" in parsed:
        parsed["THER"] = handle_dates(parsed["THER"], date_cols=["start_dt", "end_dt"])

    # -- Step 8: Filter all files to pediatric primaryids (optimized) --
    t_filter = time.perf_counter()
    # Use numeric set for faster isin() — avoids string comparison overhead
    valid_pids = set(pd.to_numeric(demo["primaryid"], errors="coerce").dropna().astype("int64"))
    for ftype in ["DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"]:
        if ftype in parsed:
            before = len(parsed[ftype])
            pid_numeric = pd.to_numeric(parsed[ftype]["primaryid"], errors="coerce")
            parsed[ftype] = parsed[ftype][pid_numeric.isin(valid_pids)].copy()
            print(f"  [{ftype}] Filtered to pediatric: {before:,} -> {len(parsed[ftype]):,}")
    print(f"  [PIPELINE] Filtering took {time.perf_counter() - t_filter:.1f}s")

    # ── Step 9: Drug cleaning ─────────────────────────────────────
    # 9a: Remove val_vbm='2' (not FDA-validated)
    parsed["DRUG"] = _filter_val_vbm(parsed["DRUG"])
    # 9b: Remove role_cod='DN' (Definitively Not suspect)
    parsed["DRUG"] = clean_drug_roles(parsed["DRUG"])
    # 9c: Route normalization (Unknown/Other → NULL, variant mapping)
    if "route" in parsed["DRUG"].columns:
        parsed["DRUG"] = normalize_route(parsed["DRUG"])
    # 9d: Encode dechallenge (Y=1, N=0, U/null=None)
    parsed["DRUG"] = _encode_dechal(parsed["DRUG"])

    # ── Step 10: Drug name normalization (local fallback) ─────────
    if skip_rxnorm:
        parsed["DRUG"]["drugname_normalized"] = (
            parsed["DRUG"]["drugname"].fillna("UNKNOWN").str.lower().str.strip()
        )
        parsed["DRUG"]["rxcui"] = None
    # If not skipping RxNorm, the async wrapper handles it after this returns

    # ── Step 11: OUTC severity scoring ────────────────────────────
    parsed["OUTC"] = score_outcomes(parsed["OUTC"])

    # ── Step 12: INDI cleaning ────────────────────────────────────
    if "INDI" in parsed:
        parsed["INDI"] = _clean_indications(parsed["INDI"])

    # ── Step 13: Tag source quarter ───────────────────────────────
    for ftype, df in parsed.items():
        df["source_quarter"] = quarter
    demo["source_quarter"] = quarter

    # ── Build preview stats ───────────────────────────────────────
    age_group_dist = demo["age_group"].value_counts(dropna=False).to_dict() if "age_group" in demo.columns else {}
    sex_dist = demo["sex"].value_counts(dropna=False).to_dict() if "sex" in demo.columns else {}
    age_unknown_count = int(demo["age_unknown"].sum()) if "age_unknown" in demo.columns else 0
    age_imputed_count = int(demo["is_age_imputed"].sum()) if "is_age_imputed" in demo.columns else 0

    null_rates = {}
    for col in ["age_years", "event_dt", "fda_dt", "sex", "weight_kg"]:
        if col in demo.columns:
            null_rates[col] = round(demo[col].isna().mean() * 100, 1)

    drug_sample = []
    if "drugname" in parsed["DRUG"].columns and "drugname_normalized" in parsed["DRUG"].columns:
        drug_sample = (
            parsed["DRUG"][["drugname", "drugname_normalized"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )

    top_adr = {}
    adr_col = "pt_term" if "pt_term" in parsed["REAC"].columns else "pt"
    if adr_col in parsed["REAC"].columns:
        top_adr = parsed["REAC"][adr_col].value_counts().head(10).to_dict()

    top_outcomes = {}
    if "outc_cod" in parsed["OUTC"].columns:
        top_outcomes = parsed["OUTC"]["outc_cod"].value_counts().to_dict()

    role_dist = {}
    if "role_cod" in parsed["DRUG"].columns:
        role_dist = parsed["DRUG"]["role_cod"].value_counts().to_dict()

    # GNN readiness: count drug-ADR pairs with n>=3 (optimized for large data)
    gnn_pairs = 0
    gnn_drugs = 0
    gnn_adrs = 0
    try:
        ps_drugs = parsed["DRUG"][parsed["DRUG"]["role_cod"] == "PS"]
        # Sample if too large to avoid memory/time explosion
        max_rows = 200_000
        if len(ps_drugs) > max_rows:
            ps_drugs = ps_drugs.sample(n=max_rows, random_state=42)
        reac_df = parsed["REAC"]
        if len(reac_df) > max_rows:
            reac_df = reac_df.sample(n=max_rows, random_state=42)

        # Use index-based join for speed
        ps_pids = set(ps_drugs["primaryid"].astype(str))
        reac_filtered = reac_df[reac_df["primaryid"].astype(str).isin(ps_pids)]

        merged = pd.merge(
            ps_drugs[["primaryid", "drugname_normalized"]],
            reac_filtered[["primaryid", adr_col]],
            on="primaryid",
            how="inner",
        )
        pair_counts = merged.groupby(["drugname_normalized", adr_col]).size()
        gnn_pairs = int((pair_counts >= 3).sum())
        gnn_drugs = int(merged["drugname_normalized"].nunique())
        gnn_adrs = int(merged[adr_col].nunique())
        print(f"  [GNN] Preview: {gnn_pairs} pairs, {gnn_drugs} drugs, {gnn_adrs} ADRs")
    except Exception as e:
        print(f"  [GNN] Preview stats skipped: {e}")
        pass

    preview_stats = {
        "quarter": quarter,
        "demo_raw_rows": demo_raw_count,
        "demo_after_dedup": demo_dedup_count,
        "demo_dedup_removed": dedup_removed,
        "demo_after_pediatric_filter": pediatric_count,
        "pediatric_pct": round(pediatric_count / max(demo_dedup_count, 1) * 100, 1),
        "age_group_distribution": age_group_dist,
        "age_unknown_count": age_unknown_count,
        "age_imputed_count": age_imputed_count,
        "sex_distribution": sex_dist,
        "null_rates": null_rates,
        "drug_rows_total": len(parsed["DRUG"]),
        "drug_ps_rows": int(parsed["DRUG"]["role_cod"].eq("PS").sum()) if "role_cod" in parsed["DRUG"].columns else 0,
        "drug_role_distribution": role_dist,
        "drug_sample": drug_sample,
        "top_adr": top_adr,
        "top_outcomes": top_outcomes,
        "gnn_drug_adr_pairs_n3": gnn_pairs,
        "gnn_unique_drugs": gnn_drugs,
        "gnn_unique_adr_terms": gnn_adrs,
        "validation": validation_results,
        "file_types_detected": list(parsed.keys()),
    }

    elapsed = time.perf_counter() - pipeline_start
    preview_stats["pipeline_elapsed_seconds"] = round(elapsed, 1)
    print(f"  [PIPELINE] TOTAL: {elapsed:.1f}s")

    return preview_stats, demo, parsed


async def run_pipeline(files: dict, quarter: str) -> tuple:
    """
    Async wrapper for the pipeline. Offloads CPU-bound work to a thread
    so uvicorn's event loop stays responsive.
    """
    import asyncio

    skip_rxnorm = os.environ.get("SKIP_RXNORM", "0") == "1"

    # Run the entire CPU-heavy pipeline in a thread pool
    preview_stats, demo, parsed = await asyncio.to_thread(
        _run_pipeline_sync, files, quarter, skip_rxnorm
    )

    # Handle async RxNorm normalization if not skipped
    if not skip_rxnorm and "drugname_normalized" not in parsed["DRUG"].columns:
        try:
            parsed["DRUG"] = await normalize_drugs(parsed["DRUG"])
        except Exception:
            parsed["DRUG"]["drugname_normalized"] = (
                parsed["DRUG"]["drugname"].fillna("UNKNOWN").str.lower().str.strip()
            )
            parsed["DRUG"]["rxcui"] = None

    return preview_stats, demo, parsed
