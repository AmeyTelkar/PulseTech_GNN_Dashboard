# Source: app/gnn/trainer.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
GNN Trainer — Single-stage per-age-band training.
Self-contained: trains only on FAERS data uploaded through the pipeline.
No external datasets (no Mendeley).
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional

try:
    from torch_geometric.transforms import RandomLinkSplit
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from sklearn.metrics import roc_auc_score
try:
    from app.gnn.model import PediatricADRHAN
    from app.gnn.graph_builder import build_heterogeneous_graph
except ImportError:
    from model import PediatricADRHAN
    from graph_builder import build_heterogeneous_graph

# Global stores for training state
training_status: dict = {}
trained_models: dict = {}
trained_data: dict = {}

CHECKPOINT_DIR = os.environ.get("GNN_CHECKPOINT_DIR", "./gnn/checkpoints")


def save_model_checkpoint(model, age_group: str, val_auc: float) -> str:
    """Save model checkpoint per age group."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"han_{age_group}_{val_auc:.4f}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Saved checkpoint: {path}")
    return path


def train_gnn_for_age_group(
    demo_df, drug_df, reac_df,
    model_key: str,
    hidden_channels: int = 64,
    out_channels: int = 32,
    epochs: int = 100,
    lr: float = 0.01,
    use_cohort_nodes: bool = True,
    use_co_admin_edges: bool = False,
) -> dict:
    """
    Train a HAN model for link prediction on a specific age group.
    Uses only FAERS data from the pipeline — no external datasets.

    Returns training metrics dict.
    """
    if not HAS_PYG:
        return {"status": "error", "message": "PyTorch Geometric not installed"}

    training_status[model_key] = {"status": "building_graph", "epoch": 0}
    
    actual_age_group = model_key.split('_')[0] if '_' in model_key else model_key

    # Build graph
    result = build_heterogeneous_graph(
        demo_df, drug_df, reac_df, actual_age_group,
        use_cohort_nodes=use_cohort_nodes,
        use_co_admin_edges=use_co_admin_edges,
    )
    if result[0] is None:
        training_status[model_key] = {"status": "error", "message": "Insufficient data"}
        return training_status[model_key]

    data, drug_idx, adr_idx, cohort_idx = result

    # Check if we have enough edges
    edge_type = ("drug", "reported_with", "adr")
    if edge_type not in data.edge_types or data[edge_type].edge_index.size(1) < 10:
        training_status[model_key] = {"status": "error", "message": "Too few edges for training"}
        return training_status[model_key]

    training_status[model_key] = {"status": "splitting_data", "epoch": 0}

    # Train/val/test split on drug-ADR edges
    try:
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            edge_types=[edge_type],
            rev_edge_types=[("adr", "rev_reported_with", "drug")],
        )
        train_data, val_data, test_data = transform(data)
    except Exception as e:
        training_status[model_key] = {"status": "error", "message": f"Split failed: {str(e)}"}
        return training_status[model_key]

    # Initialize model
    n_drug = len(drug_idx)
    n_adr = len(adr_idx)
    in_channels = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
    metadata = data.metadata()

    model = PediatricADRHAN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=metadata,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    training_status[model_key] = {"status": "training", "epoch": 0, "total_epochs": epochs}

    best_val_auc = 0.0
    best_state = None
    losses = []

    def safe_embeddings(out_dict, n_drug, n_adr, dim):
        """Ensure 'drug' and 'adr' keys exist with correct shape."""
        if "drug" not in out_dict:
            out_dict["drug"] = torch.zeros(n_drug, dim)
        if "adr" not in out_dict:
            out_dict["adr"] = torch.zeros(n_adr, dim)
        return out_dict

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(train_data.x_dict, train_data.edge_index_dict)
        out = safe_embeddings(out, n_drug, n_adr, out_channels)

        # Decode positive edges
        edge_label_index = train_data[edge_type].edge_label_index
        edge_label = train_data[edge_type].edge_label

        pred = model.decode(out["drug"], out["adr"], edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        loss.backward()
        optimizer.step()

        losses.append(float(loss))
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x_dict, val_data.edge_index_dict)
                val_out = safe_embeddings(val_out, n_drug, n_adr, out_channels)
                val_pred = torch.sigmoid(model.decode(
                    val_out["drug"], val_out["adr"],
                    val_data[edge_type].edge_label_index,
                ))
                val_labels = val_data[edge_type].edge_label
                try:
                    val_auc = float(roc_auc_score(
                        val_labels.numpy(), val_pred.numpy()
                    ))
                except Exception:
                    val_auc = 0.0

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                training_status[model_key] = {
                    "status": "training",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": float(loss),
                    "val_auc": float(val_auc)
                }

                if (epoch + 1) % 10 == 0:
                    print(f"  [{actual_age_group}] Epoch {epoch+1}/{epochs} | loss: {loss.item():.4f} | val_auc: {val_auc:.4f}")

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x_dict, test_data.edge_index_dict)
        test_out = safe_embeddings(test_out, n_drug, n_adr, out_channels)
        test_pred = model.decode(
            test_out["drug"], test_out["adr"],
            test_data[edge_type].edge_label_index,
        )
        test_loss = F.binary_cross_entropy_with_logits(
            test_pred, test_data[edge_type].edge_label
        )

        # AUC
        pred_probs = torch.sigmoid(test_pred).numpy()
        labels = test_data[edge_type].edge_label.numpy()
        try:
            auc = round(float(roc_auc_score(labels, pred_probs)), 4)
        except Exception:
            auc = None

    # Save checkpoint
    ckpt_path = save_model_checkpoint(model, actual_age_group, best_val_auc)

    # Store trained model and data
    trained_models[model_key] = model
    trained_data[model_key] = {
        "data": data,
        "drug_idx": drug_idx,
        "adr_idx": adr_idx,
        "cohort_idx": cohort_idx,
    }

    training_status[model_key] = {
        "status": "completed",
        "epochs": epochs,
        "final_loss": round(float(losses[-1]), 4),
        "best_val_auc": round(best_val_auc, 4),
        "test_loss": round(float(test_loss), 4),
        "test_auc": auc,
        "n_drugs": len(drug_idx),
        "n_adrs": len(adr_idx),
        "n_edges": int(data[edge_type].edge_index.size(1)),
        "checkpoint_path": ckpt_path,
    }

    return training_status[model_key]


# Backward-compatible alias for ablation
finetune_on_pediatric = train_gnn_for_age_group
