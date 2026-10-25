# Source: app/gnn/ablation.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
GNN Ablation Study — measure AUC contribution of each graph component.
Not used for any result reported in the manuscript.

Runs 4 configurations:
  A: Full graph (drug→adr + cohort→drug + cohort→adr)
  B: Remove cohort nodes
  C: Full graph + co-administration edges (drug→drug)
  D: Only drug→adr edges (baseline)
"""
try:
    from app.gnn.trainer import finetune_on_pediatric
except ImportError:
    from gnn_trainer import finetune_on_pediatric


def run_ablation_study(
    demo_df, drug_df, reac_df,
    age_group: str = "CHILD",
    epochs: int = 30,
    device: str = "cpu",
) -> dict:
    """
    Run ablation study measuring AUC contribution of each graph component.

    Returns dict mapping config name → training results (including test_auc).
    """
    configs = {
        "A_full": {"use_cohort_nodes": True, "use_co_admin_edges": False},
        "B_no_cohort": {"use_cohort_nodes": False, "use_co_admin_edges": False},
        "C_with_co_admin": {"use_cohort_nodes": True, "use_co_admin_edges": True},
        "D_baseline": {"use_cohort_nodes": False, "use_co_admin_edges": False},
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Ablation [{name}] for {age_group}")
        print(f"{'='*60}")

        result = finetune_on_pediatric(
            demo_df, drug_df, reac_df, age_group,
            epochs=epochs,
            device=device,
            use_cohort_nodes=cfg["use_cohort_nodes"],
            use_co_admin_edges=cfg["use_co_admin_edges"],
        )
        results[name] = result
        auc = result.get("test_auc", "N/A")
        print(f"Ablation [{name}]: AUC = {auc}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY — {age_group}")
    print(f"{'='*60}")
    print(f"  {'Config':<25s} {'AUC':>8s} {'Edges':>8s}")
    print(f"  {'-'*45}")
    for name, r in results.items():
        auc = r.get("test_auc", "N/A")
        edges = r.get("n_edges", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)
        print(f"  {name:<25s} {auc_str:>8s} {str(edges):>8s}")

    return results
