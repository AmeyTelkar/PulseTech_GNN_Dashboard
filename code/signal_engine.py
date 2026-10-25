"""
Signal Detection Engine — builds contingency tables and computes all 4 metrics.
Uses role_cod = 'PS' (Primary Suspect) only.
VECTORIZED: Computes all pairs in one pass using pandas groupby.
"""
import numpy as np
import pandas as pd
try:
    from app.analytics.prr import compute_prr
    from app.analytics.ror import compute_ror
    from app.analytics.ic import compute_ic
    from app.analytics.ebgm import compute_ebgm, fit_prior
except ImportError:
    from analytics_prr import compute_prr
    from analytics_ror import compute_ror
    from analytics_ic import compute_ic
    from analytics_ebgm import compute_ebgm, fit_prior


def is_signal(row: dict) -> bool:
    """True if ANY of the four methods flags a signal."""
    prr_ok = (row.get("prr") or 0) >= 2 and (row.get("prr_chi2") or 0) >= 4
    ror_ok = (row.get("ror") or 0) >= 2 and (row.get("ror_ci_lower") or 0) > 1
    ic_ok = row.get("ic025") is not None and row["ic025"] > 0
    eb_ok = row.get("eb05") is not None and row["eb05"] > 2
    return prr_ok or ror_ok or ic_ok or eb_ok


def compute_signals_for_age_group(
    drug_df: pd.DataFrame,
    reac_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    age_group: str,
    min_n: int = 3,
) -> list[dict]:
    """
    Vectorized signal detection for all drug-ADR pairs in an age group.
    Builds contingency tables using pandas groupby instead of per-pair loops.
    """
    # Filter to age group
    pids = set(demo_df[demo_df["age_group"] == age_group]["primaryid"].astype(str))
    nxx = len(pids)

    if nxx == 0:
        return []

    # Filter drugs/reactions to this age group's patients
    ps_drugs = drug_df[
        (drug_df["role_cod"] == "PS")
        & (drug_df["primaryid"].astype(str).isin(pids))
    ][["primaryid", "drugname_normalized"]].copy()
    ps_drugs["primaryid"] = ps_drugs["primaryid"].astype(str)

    reac = reac_df[reac_df["primaryid"].astype(str).isin(pids)][["primaryid", "pt_term"]].copy()
    reac["primaryid"] = reac["primaryid"].astype(str)

    if ps_drugs.empty or reac.empty:
        return []

    # Pre-compute n1x (patients per drug) and nx1 (patients per ADR)
    n1x_map = ps_drugs.groupby("drugname_normalized")["primaryid"].nunique()
    nx1_map = reac.groupby("pt_term")["primaryid"].nunique()

    # Merge to get drug-ADR co-occurrences
    merged = pd.merge(ps_drugs, reac, on="primaryid")

    # n11: unique patients with both drug AND ADR
    pair_counts = (
        merged.groupby(["drugname_normalized", "pt_term"])["primaryid"]
        .nunique()
        .reset_index(name="n11")
    )
    pair_counts = pair_counts[pair_counts["n11"] >= min_n]

    if pair_counts.empty:
        return []

    print(f"    [{age_group}] Computing {len(pair_counts)} pairs (nxx={nxx:,})...")

    # Vectorized contingency table computation
    pair_counts["n1x"] = pair_counts["drugname_normalized"].map(n1x_map).fillna(0).astype(int)
    pair_counts["nx1"] = pair_counts["pt_term"].map(nx1_map).fillna(0).astype(int)
    pair_counts["nxx"] = nxx
    pair_counts["n10"] = pair_counts["n1x"] - pair_counts["n11"]
    pair_counts["n01"] = pair_counts["nx1"] - pair_counts["n11"]
    pair_counts["n00"] = (nxx - pair_counts["n11"] - pair_counts["n10"] - pair_counts["n01"]).clip(lower=0)

    # EBGM prior: fitted once per age band from every evaluated pair (DuMouchel 1999),
    # with the likelihood truncated at min_n because only pairs with n11 >= min_n are kept
    expected = pair_counts["n1x"] * pair_counts["nx1"] / nxx
    ebgm_prior = fit_prior(pair_counts["n11"].to_numpy(), expected.to_numpy(), min_n=min_n)
    print(f"    [{age_group}] EBGM prior (a1, b1, a2, b2, w) = {tuple(round(v, 4) for v in ebgm_prior)}")

    # Compute all 4 metrics for each row
    results = []
    for _, row in pair_counts.iterrows():
        ct = {
            "n11": int(row["n11"]), "n1x": int(row["n1x"]),
            "nx1": int(row["nx1"]), "nxx": int(row["nxx"]),
            "n10": int(row["n10"]), "n01": int(row["n01"]),
            "n00": int(row["n00"]),
        }
        result = {**ct}
        result.update(compute_prr(ct))
        result.update(compute_ror(ct))
        result.update(compute_ic(ct))
        result.update(compute_ebgm(ct, prior=ebgm_prior))
        result["is_signal"] = is_signal(result)
        result["drugname_normalized"] = row["drugname_normalized"]
        result["pt_term"] = row["pt_term"]
        result["age_group"] = age_group
        results.append(result)

    flagged = sum(1 for r in results if r["is_signal"])
    print(f"    [{age_group}] {len(results)} pairs, {flagged} flagged as signals")
    return results
