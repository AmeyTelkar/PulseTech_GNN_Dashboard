"""
GNN Predictor — top-K novel drug-ADR predictions per age group.
"""
import torch
import numpy as np
try:
    from app.gnn.trainer import trained_models, trained_data
except ImportError:
    from gnn_trainer import trained_models, trained_data


def predict_novel_links(
    age_group: str,
    top_k: int = 50,
) -> list[dict]:
    """
    Use trained GNN to predict novel drug-ADR links not in training data.

    Returns list of {drug, adr, confidence} sorted by confidence descending.
    """
    if age_group not in trained_models or age_group not in trained_data:
        return []

    model = trained_models[age_group]
    td = trained_data[age_group]
    data = td["data"]
    drug_idx = td["drug_idx"]
    adr_idx = td["adr_idx"]

    # Reverse index maps
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    idx_to_adr = {v: k for k, v in adr_idx.items()}

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        drug_emb = out["drug"]
        adr_emb = out["adr"]

        # Get all pairwise scores
        scores = model.decode_all(drug_emb, adr_emb)  # [n_drug, n_adr]

    # Get existing edges (to exclude known links)
    edge_type = ("drug", "reported_with", "adr")
    existing_edges = set()
    if edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        for i in range(ei.size(1)):
            existing_edges.add((int(ei[0, i]), int(ei[1, i])))

    # Find novel predictions
    scores_np = scores.numpy()
    predictions = []

    # Get top-K novel links
    flat_indices = np.argsort(scores_np.flatten())[::-1]

    for flat_idx in flat_indices:
        drug_i = int(flat_idx // scores_np.shape[1])
        adr_j = int(flat_idx % scores_np.shape[1])

        if (drug_i, adr_j) in existing_edges:
            continue

        if drug_i in idx_to_drug and adr_j in idx_to_adr:
            predictions.append({
                "drug": idx_to_drug[drug_i],
                "adr": idx_to_adr[adr_j],
                "confidence": round(float(scores_np[drug_i, adr_j]), 4),
                "age_group": age_group,
            })

        if len(predictions) >= top_k:
            break

    return predictions
