"""
evaluate_link_prediction.py -- held-out link-prediction evaluation of PediatricADRHAN.

Trains the PediatricADRHAN model with exactly the settings of gnn_trainer.py / Table 5
(2 HANConv layers, 64 hidden, 32 output, 4 heads, 100 epochs, Adam, LR 0.01,
weight decay 1e-4, validation ROC-AUC every 5 epochs with best-checkpoint restore,
80/10/10 RandomLinkSplit on drug-ADR edges) on the CHILD band of the Q1 2025 data,
for 5 random seeds. For every seed the same held-out test edges (positives plus the
equal number of negatives sampled by RandomLinkSplit) are also scored with three
classical link-prediction heuristics, adapted to the bipartite drug-ADR graph:

  * Common neighbours  CN(d, a) = |N(N(d)) & N(a)|
      number of drugs that share at least one ADR with drug d and are themselves
      reported with ADR a (a length-3 path count collapsed to distinct drugs).
  * Adamic-Adar        AA(d, a) = sum_{d' in N(N(d)) & N(a)} 1 / log(1 + deg(d'))
  * Preferential attachment  PA(d, a) = deg(d) * deg(a)

Heuristics are computed on the message-passing graph that the GNN sees at test time
(training + validation drug-ADR edges), so no test edge is visible to any method.

Output: gnn_link_prediction_results.csv (one row per seed and method) and a summary
(mean +/- SD) printed to stdout and written to gnn_link_prediction_summary.csv.

Input: a loaded PulseTech PostgreSQL database (not included in this archive), read
through DATABASE_URL_SYNC, e.g.
    set DATABASE_URL_SYNC=postgresql+psycopg2://USER:PASSWORD@localhost:5432/faers_pediatric
    python evaluate_link_prediction.py --quarter 2025Q1 --age-group CHILD --seeds 0 1 2 3 4
"""
import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_builder import build_heterogeneous_graph  # noqa: E402
from model import PediatricADRHAN  # noqa: E402

EDGE = ("drug", "reported_with", "adr")
REV = ("adr", "rev_reported_with", "drug")

# Table 5 / gnn_trainer.py defaults
HIDDEN, OUT, EPOCHS, LR, WD = 64, 32, 100, 0.01, 1e-4


def load_frames(quarter: str):
    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL_SYNC")
    if not url:
        sys.exit("Set DATABASE_URL_SYNC to the PulseTech PostgreSQL database.")
    eng = create_engine(url)
    q = "AND source_quarter = :q" if quarter != "ALL" else ""
    p = {"q": quarter} if quarter != "ALL" else {}
    with eng.connect() as c:
        demo = pd.DataFrame(c.execute(text(
            "SELECT primaryid, age_group FROM demographics "
            f"WHERE age_group IS NOT NULL AND is_superseded = FALSE {q}"), p).fetchall(),
            columns=["primaryid", "age_group"])
        drug = pd.DataFrame(c.execute(text(
            "SELECT primaryid, drugname_normalized, role_cod FROM drugs "
            f"WHERE is_superseded = FALSE {q}"), p).fetchall(),
            columns=["primaryid", "drugname_normalized", "role_cod"])
        reac = pd.DataFrame(c.execute(text(
            f"SELECT primaryid, pt_term FROM reactions WHERE is_superseded = FALSE {q}"), p).fetchall(),
            columns=["primaryid", "pt_term"])
    return demo, drug, reac


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe(out, n_drug, n_adr):
    if "drug" not in out:
        out["drug"] = torch.zeros(n_drug, OUT)
    if "adr" not in out:
        out["adr"] = torch.zeros(n_adr, OUT)
    return out


def train_and_score(data, n_drug, n_adr, seed):
    """Same loop as gnn_trainer.train_gnn_for_age_group; returns test labels/scores."""
    set_seed(seed)
    split = RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=[EDGE], rev_edge_types=[REV])
    train_d, val_d, test_d = split(data)

    set_seed(seed)
    in_ch = {nt: data[nt].x.size(1) for nt in data.node_types}
    model = PediatricADRHAN(in_ch, HIDDEN, OUT, data.metadata())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_auc, best_state, best_epoch = 0.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = safe(model(train_d.x_dict, train_d.edge_index_dict), n_drug, n_adr)
        pred = model.decode(out["drug"], out["adr"], train_d[EDGE].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, train_d[EDGE].edge_label)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vo = safe(model(val_d.x_dict, val_d.edge_index_dict), n_drug, n_adr)
                vp = torch.sigmoid(model.decode(vo["drug"], vo["adr"], val_d[EDGE].edge_label_index))
                try:
                    vauc = float(roc_auc_score(val_d[EDGE].edge_label.numpy(), vp.numpy()))
                except ValueError:
                    vauc = 0.0
            if vauc > best_auc:
                best_auc, best_epoch = vauc, epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        to = safe(model(test_d.x_dict, test_d.edge_index_dict), n_drug, n_adr)
        tp = torch.sigmoid(model.decode(to["drug"], to["adr"], test_d[EDGE].edge_label_index)).numpy()
    return test_d, tp, best_auc, best_epoch


def heuristic_scores(test_d):
    """CN, AA, PA on the test-time message-passing drug-ADR graph (train + val edges)."""
    mp = test_d[EDGE].edge_index.numpy()
    drug_nb, adr_nb = defaultdict(set), defaultdict(set)
    for d, a in zip(mp[0], mp[1]):
        drug_nb[int(d)].add(int(a))
        adr_nb[int(a)].add(int(d))

    pairs = test_d[EDGE].edge_label_index.numpy()
    cn, aa, pa = [], [], []
    two_hop_cache = {}
    for d, a in zip(pairs[0], pairs[1]):
        d, a = int(d), int(a)
        if d not in two_hop_cache:
            s = set()
            for a2 in drug_nb.get(d, ()):
                s |= adr_nb[a2]
            s.discard(d)
            two_hop_cache[d] = s
        common = two_hop_cache[d] & adr_nb.get(a, set())
        cn.append(len(common))
        aa.append(sum(1.0 / math.log(1 + len(drug_nb[x])) for x in common))
        pa.append(len(drug_nb.get(d, ())) * len(adr_nb.get(a, ())))
    return {"Common neighbours": np.array(cn, float),
            "Adamic-Adar": np.array(aa, float),
            "Preferential attachment": np.array(pa, float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", default="2025Q1")
    ap.add_argument("--age-group", default="CHILD")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    demo, drug, reac = load_frames(args.quarter)
    data, drug_idx, adr_idx, _ = build_heterogeneous_graph(demo, drug, reac, args.age_group)
    n_drug, n_adr = len(drug_idx), len(adr_idx)
    n_edges = int(data[EDGE].edge_index.size(1))
    print(f"{args.age_group} {args.quarter}: {n_drug} drugs, {n_adr} ADRs, {n_edges} drug-ADR edges")

    rows = []
    for seed in args.seeds:
        test_d, gnn_p, best_val, best_ep = train_and_score(data, n_drug, n_adr, seed)
        y = test_d[EDGE].edge_label.numpy()
        scores = {"PediatricADRHAN (HANConv)": gnn_p, **heuristic_scores(test_d)}
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        for method, s in scores.items():
            rows.append({"age_group": args.age_group, "quarter": args.quarter, "seed": seed,
                         "method": method, "test_roc_auc": roc_auc_score(y, s),
                         "test_aupr": average_precision_score(y, s),
                         "n_test_pos": n_pos, "n_test_neg": n_neg,
                         "best_val_auc": best_val if method.startswith("Pediatric") else np.nan,
                         "best_epoch": best_ep if method.startswith("Pediatric") else np.nan,
                         "n_drugs": n_drug, "n_adrs": n_adr, "n_edges": n_edges})
        print(f"seed {seed}: " + ", ".join(
            f"{r['method']} AUC={r['test_roc_auc']:.4f} AUPR={r['test_aupr']:.4f}"
            for r in rows[-4:]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "gnn_link_prediction_results.csv"), index=False, float_format="%.4f")
    summ = (df.groupby("method", sort=False)[["test_roc_auc", "test_aupr"]]
              .agg(["mean", "std"]).round(4))
    summ.columns = ["roc_auc_mean", "roc_auc_sd", "aupr_mean", "aupr_sd"]
    summ.to_csv(os.path.join(args.out, "gnn_link_prediction_summary.csv"))
    print("\nMean +/- SD over seeds", args.seeds)
    print(summ.to_string())


if __name__ == "__main__":
    main()
