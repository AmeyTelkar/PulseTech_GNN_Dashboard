"""
Heterogeneous Graph Builder for FAERS Pediatric ADR.
Self-contained — builds graphs from uploaded FAERS data only.

Node types: drug, adr, cohort (per ICH E11 age band)
Edge types:
  - ('drug', 'reported_with', 'adr'):  co-report count as edge weight
  - ('cohort', 'received', 'drug'):    report count in cohort
  - ('cohort', 'experienced', 'adr'):  report count in cohort
  - ('drug', 'co_administered', 'drug'): drugs used together (optional)
"""
import torch
import pandas as pd
import numpy as np

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


from typing import Optional


def build_heterogeneous_graph(
    demo_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    reac_df: pd.DataFrame,
    age_group: Optional[str] = None,
    use_cohort_nodes: bool = True,
    use_co_admin_edges: bool = False,
):
    """
    Build a HeteroData graph from FAERS DataFrames.
    Used for GNN training per ICH E11 age band.

    Args:
        demo_df: Demographics with primaryid, age_group
        drug_df: Drugs with primaryid, drugname_normalized, role_cod
        reac_df: Reactions with primaryid, pt_term
        age_group: If specified, filter to this age group only
        use_cohort_nodes: Include cohort (age group) nodes
        use_co_admin_edges: Include drug-drug co-administration edges
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for GNN features")

    data = HeteroData()

    # Filter to age group if specified
    if age_group and age_group != "ALL":
        pids = set(demo_df[demo_df["age_group"] == age_group]["primaryid"].astype(str))
    else:
        pids = set(demo_df["primaryid"].astype(str))

    ps_drugs = drug_df[
        (drug_df["role_cod"] == "PS")
        & (drug_df["primaryid"].astype(str).isin(pids))
    ]
    reac = reac_df[reac_df["primaryid"].astype(str).isin(pids)]

    # Build index maps
    drug_names = ps_drugs["drugname_normalized"].dropna().unique().tolist()
    adr_terms = reac["pt_term"].dropna().unique().tolist()
    cohort_groups = demo_df["age_group"].dropna().unique().tolist()

    drug_idx = {d: i for i, d in enumerate(drug_names)}
    adr_idx = {a: i for i, a in enumerate(adr_terms)}
    cohort_idx = {c: i for i, c in enumerate(cohort_groups)}

    n_drug = len(drug_names)
    n_adr = len(adr_terms)
    n_cohort = len(cohort_groups)

    if n_drug == 0 or n_adr == 0:
        return None, drug_idx, adr_idx, cohort_idx

    # Node features (identity for now)
    data["drug"].x = torch.eye(n_drug, dtype=torch.float)
    data["adr"].x = torch.eye(n_adr, dtype=torch.float)
    if use_cohort_nodes and n_cohort > 0:
        data["cohort"].x = torch.eye(n_cohort, dtype=torch.float)

    # Drug → ADR edges (reported_with)
    merged = pd.merge(
        ps_drugs[["primaryid", "drugname_normalized"]],
        reac[["primaryid", "pt_term"]],
        on="primaryid",
    )
    drug_adr_counts = (
        merged.groupby(["drugname_normalized", "pt_term"])
        .size()
        .reset_index(name="count")
    )

    src, dst, weights = [], [], []
    for _, row in drug_adr_counts.iterrows():
        d = row["drugname_normalized"]
        a = row["pt_term"]
        if d in drug_idx and a in adr_idx:
            src.append(drug_idx[d])
            dst.append(adr_idx[a])
            weights.append(float(np.log1p(row["count"])))

    if src:
        data["drug", "reported_with", "adr"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )
        data["drug", "reported_with", "adr"].edge_weight = torch.tensor(
            weights, dtype=torch.float
        )
        # Reverse edges: adr → drug (required for bidirectional message passing)
        data["adr", "rev_reported_with", "drug"].edge_index = torch.tensor(
            [dst, src], dtype=torch.long
        )
        data["adr", "rev_reported_with", "drug"].edge_weight = torch.tensor(
            weights, dtype=torch.float
        )

    # Cohort → Drug edges
    if use_cohort_nodes and n_cohort > 0:
        merged_cd = pd.merge(
            demo_df[["primaryid", "age_group"]],
            ps_drugs[["primaryid", "drugname_normalized"]],
            on="primaryid",
        )
        cd_counts = (
            merged_cd.groupby(["age_group", "drugname_normalized"])
            .size()
            .reset_index(name="count")
        )

        src_c, dst_d, w_cd = [], [], []
        for _, row in cd_counts.iterrows():
            c, d = row["age_group"], row["drugname_normalized"]
            if c in cohort_idx and d in drug_idx:
                src_c.append(cohort_idx[c])
                dst_d.append(drug_idx[d])
                w_cd.append(float(np.log1p(row["count"])))

        if src_c:
            data["cohort", "received", "drug"].edge_index = torch.tensor(
                [src_c, dst_d], dtype=torch.long
            )
            data["cohort", "received", "drug"].edge_weight = torch.tensor(
                w_cd, dtype=torch.float
            )

        # Cohort → ADR edges
        merged_ca = pd.merge(
            demo_df[["primaryid", "age_group"]],
            reac[["primaryid", "pt_term"]],
            on="primaryid",
        )
        ca_counts = (
            merged_ca.groupby(["age_group", "pt_term"])
            .size()
            .reset_index(name="count")
        )

        src_ca, dst_a, w_ca = [], [], []
        for _, row in ca_counts.iterrows():
            c, a = row["age_group"], row["pt_term"]
            if c in cohort_idx and a in adr_idx:
                src_ca.append(cohort_idx[c])
                dst_a.append(adr_idx[a])
                w_ca.append(float(np.log1p(row["count"])))

        if src_ca:
            data["cohort", "experienced", "adr"].edge_index = torch.tensor(
                [src_ca, dst_a], dtype=torch.long
            )
            data["cohort", "experienced", "adr"].edge_weight = torch.tensor(
                w_ca, dtype=torch.float
            )

    # Drug → Drug co-administration edges (optional)
    if use_co_admin_edges:
        from collections import Counter
        co_admin = (
            ps_drugs.groupby("primaryid")["drugname_normalized"]
            .apply(list)
            .reset_index()
        )
        pair_counts = Counter()
        for _, row in co_admin.iterrows():
            drugs_list = row["drugname_normalized"]
            if len(drugs_list) < 2:
                continue
            for i, d1 in enumerate(drugs_list):
                for d2 in drugs_list[i + 1:]:
                    if d1 in drug_idx and d2 in drug_idx:
                        pair_counts[(drug_idx[d1], drug_idx[d2])] += 1

        co_src, co_dst, co_w = [], [], []
        for (s, d), cnt in pair_counts.items():
            if cnt >= 2:
                co_src.extend([s, d])
                co_dst.extend([d, s])
                co_w.extend([float(np.log1p(cnt))] * 2)

        if co_src:
            data["drug", "co_administered", "drug"].edge_index = torch.tensor(
                [co_src, co_dst], dtype=torch.long
            )
            data["drug", "co_administered", "drug"].edge_weight = torch.tensor(
                co_w, dtype=torch.float
            )

    return data, drug_idx, adr_idx, cohort_idx
