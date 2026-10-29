"""
validate_against_labels.py -- external validation of the S8 Table signals against drug labels.

Question: are the drug-ADR pairs that PulseTech flags more often adverse reactions that are
already listed on the drugs' labels than the pairs it does not flag, and how many flagged pairs
are instead the drug's own indication or a product-use / medication-error / lack-of-effect term?

Reference standard: SIDER 4.1 (Kuhn et al., Nucleic Acids Res 2016), side effects and
indications extracted from drug labels, as tidied by D. Himmelstein
(github.com/dhimmel/SIDER4, files data/side-effects.tsv and data/indications.tsv).
SIDER is licensed CC BY-NC-SA 4.0 and is NOT redistributed here: download it yourself:
    git clone https://github.com/dhimmel/SIDER4

Terminology: MedDRA 28.1 ASCII files (llt.asc, pt.asc, mdhier.asc) are used to map SIDER's
label terms (MedDRA lowest level terms) and the S8 Preferred Terms to the same PT codes, and to
classify terms by System Organ Class. MedDRA is licensed by the MSSO and is NOT redistributed;
point --meddra to your own MedAscii folder. Without --meddra the script falls back to
case-insensitive name matching and skips the term classification.

Drug names: S8 names are mapped to SIDER drug names with drug_name_to_sider.csv (deposited:
brand names and name variants, curated by the authors) and otherwise by removing salt words and
matching the generic name. Combination products are excluded.

Usage (from the repository root):
    python code/validate_against_labels.py --sider ../SIDER4/data --meddra /path/to/MedAscii
Writes code/label_validation_summary.csv (aggregate results only) and prints the same numbers.
Use --pairs-out FILE to also write the per-pair flags locally (do not publish that file: it is
derived from SIDER, CC BY-NC-SA).
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
S8 = HERE.parent / "Supplementary_Information" / "S8_Table_Complete_Signal_Detection.xlsx"
MAP = HERE / "drug_name_to_sider.csv"

SALTS = (r"\b(hydrochloride|hcl|sodium|potassium|calcium|magnesium|acetate|besylate|besilate|pamoate|"
         r"phosphate|sulfate|sulphate|mesylate|maleate|citrate|tartrate|bitartrate|succinate|fumarate|"
         r"hydrobromide|bromide|dihydrate|monohydrate|anhydrous|disodium|dipropionate|propionate|furoate|"
         r"valerate|hyclate|lactate|gluconate|decanoate|enanthate|cypionate|xinafoate|trihydrate|"
         r"hemihydrate)\b")
COMBO = re.compile(r"\\|/| and |,")

# Term groups that describe how a product was used, exposure, overdose or the course of the
# treated disease rather than an adverse reaction. Defined on the PRIMARY MedDRA 28.1 path,
# first matching group wins (order below). The drug's own indication is a fifth group, from SIDER.
GROUPS = [
    ("Product use or medication error",
     {"hlgt": {"Medication errors and other product use errors and issues",
               "Off label uses and intentional product misuses/use issues"}}),
    ("Exposure, overdose or poisoning",
     {"hlgt": {"Exposures, chemical injuries and poisoning", "Overdoses and underdoses NEC"}}),
    ("Lack of effect, interaction, disease course or death",
     {"hlgt": {"Therapeutic and nontherapeutic effects (excl toxicity)", "Fatal outcomes"},
      "hlt": {"Adverse effect absent"},
      "pt": {"Disease progression", "Condition aggravated", "Malignant neoplasm progression"}}),
    ("Device, social or procedural term",
     {"soc": {"Product issues", "Social circumstances", "Surgical and medical procedures"}}),
]


def norm(name: str) -> str:
    x = re.sub(SALTS, "", name.lower())
    return re.sub(r"\s+", " ", x).strip(" ,")


def load_meddra(folder):
    folder = Path(folder)
    llt = pd.read_csv(folder / "llt.asc", sep="$", header=None, usecols=[0, 1, 2],
                      names=["llt", "name", "pt"], dtype=str, quoting=3)
    pt = pd.read_csv(folder / "pt.asc", sep="$", header=None, usecols=[0, 1],
                     names=["pt", "name"], dtype=str, quoting=3)
    hier = pd.read_csv(folder / "mdhier.asc", sep="$", header=None, usecols=[0, 4, 5, 6, 7, 11],
                       names=["pt", "pt_name", "hlt", "hlgt", "soc", "primary"], dtype=str, quoting=3)
    name_to_pt = dict(zip(llt.name.str.lower(), llt.pt))           # every PT is also an LLT
    name_to_pt.update(dict(zip(pt.name.str.lower(), pt.pt)))
    prim = hier[hier.primary == "Y"].set_index("pt")
    hlts = hier.groupby("pt").hlt.agg(set).to_dict()                # primary and secondary HLTs
    return name_to_pt, prim, hlts


def odds_ratio(a, b, c, d):
    """a, b = listed / not listed among flagged; c, d = among not flagged (Woolf CI, 0.5 if a zero cell)."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    or_ = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, math.exp(math.log(or_) - 1.96 * se), math.exp(math.log(or_) + 1.96 * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sider", required=True, help="folder with side-effects.tsv and indications.tsv")
    ap.add_argument("--meddra", help="MedDRA MedAscii folder (llt.asc, pt.asc, mdhier.asc)")
    ap.add_argument("--out", default=str(HERE / "label_validation_summary.csv"))
    ap.add_argument("--pairs-out")
    args = ap.parse_args()

    d = pd.read_excel(S8, sheet_name="Signal_Detection")
    se = pd.read_csv(Path(args.sider) / "side-effects.tsv", sep="\t")
    ind = pd.read_csv(Path(args.sider) / "indications.tsv", sep="\t")
    sider_names = {n.lower(): n for n in pd.concat([se.drugbank_name, ind.drugbank_name]).unique()}

    # ---- drug mapping
    curated = pd.read_csv(MAP)
    cmap = dict(zip(curated.s8_drug_name, curated.sider_drug_name))
    def to_sider(name):
        if name in cmap:
            return cmap[name], "curated"
        if COMBO.search(name):
            return None, "combination"
        n = norm(name)
        return (sider_names[n], "generic") if n in sider_names else (None, "not in SIDER")
    m = d.drugname_normalized.map(to_sider)
    d["sider_drug"] = [x[0] for x in m]
    d["map_method"] = [x[1] for x in m]

    # ---- term mapping
    if args.meddra:
        name_to_pt, prim, hlts = load_meddra(args.meddra)
        key = lambda s: name_to_pt.get(str(s).lower())
        mode = "MedDRA 28.1 PT codes"
    else:
        prim, hlts = None, {}
        key = lambda s: str(s).lower()
        mode = "case-insensitive names"
    se_keys = se.assign(k=se.side_effect_name.map(key)).dropna(subset=["k"])
    ind_name = ind.meddra_name.fillna(ind.concept_name)
    ind_keys = ind.assign(k=ind_name.map(key)).dropna(subset=["k"])
    se_set = set(zip(se_keys.drugbank_name, se_keys.k))
    ind_set = set(zip(ind_keys.drugbank_name, ind_keys.k))
    d["pt_key"] = d.pt_term.map(key)
    covered = d.sider_drug.isin(set(se.drugbank_name))           # drug has label side effects in SIDER
    d["covered"] = covered
    d["label_listed"] = [bool(c) and (s, k) in se_set for c, s, k in zip(covered, d.sider_drug, d.pt_key)]
    d["indication"] = [bool(c) and (s, k) in ind_set for c, s, k in zip(covered, d.sider_drug, d.pt_key)]
    # lenient match: a label term in the same MedDRA High Level Term (needs MedDRA)
    label_hlts = {}
    for drug, k in se_set:
        label_hlts.setdefault(drug, set()).update(hlts.get(k, set()))
    d["label_related"] = [bool(c) and bool(hlts.get(k, set()) & label_hlts.get(s, set()))
                          for c, s, k in zip(covered, d.sider_drug, d.pt_key)]
    d["term_group"] = ""
    if prim is not None:
        info = prim.reindex(d.pt_key)
        grp = pd.Series("", index=d.index)
        for gname, crit in GROUPS[::-1]:          # reverse so that the first group wins
            hit = pd.Series(False, index=d.index)
            for level, names in crit.items():
                col = {"hlgt": info.hlgt, "hlt": info.hlt, "soc": info.soc, "pt": info.pt_name}[level]
                hit |= col.isin(names).to_numpy()
            grp[hit] = gname
        d["term_group"] = grp
    d.loc[d.indication, "term_group"] = "Indication of the drug (SIDER)"
    d["product_term"] = d.term_group != ""

    # ---- rules
    P, R, RL, I, chi = d.prr.fillna(0), d.ror.fillna(0), d.ror_ci_lower.fillna(0), d.ic025, d.prr_chi2.fillna(0)
    rules = {
        "Any-method rule (stored Signal flag)": d.is_signal.astype(bool),
        "Evans (PRR>=2, chi2>=4, n>=3)": (P >= 2) & (chi >= 4) & (d.n11 >= 3),
        "ROR>=2 and ROR CI lower bound >1": (R >= 2) & (RL > 1),
        "IC025>0": I > 0,
        "EB05>2": d.eb05 > 2,
    }

    rows = [{"measure": "term mapping", "value": mode},
            {"measure": "pairs in S8 Table", "value": len(d)},
            {"measure": "pairs whose drug has label side effects in SIDER 4.1", "value": int(covered.sum())},
            {"measure": "distinct drugs covered", "value": int(d.loc[covered, "drugname_normalized"].nunique())},
            {"measure": "distinct drugs in S8 Table", "value": int(d.drugname_normalized.nunique())}]
    print(f"S8 Table: {len(d):,} pairs, {d.drugname_normalized.nunique()} drugs; terms matched by {mode}")
    print(f"Covered by SIDER 4.1: {covered.sum():,} pairs ({covered.mean():.1%}), "
          f"{d.loc[covered, 'drugname_normalized'].nunique()} drugs")
    print(f"Mapping: {d.map_method.value_counts().to_dict()}")

    cov = d[covered]
    clean = cov[~cov.indication & ~cov.product_term]
    print("\nLabel-listed adverse reactions among covered pairs, flagged vs not flagged")
    for label, flag in rules.items():
        for subset_name, sub in [("all covered pairs", cov), ("clinical-event pairs only (term categories and indications excluded)", clean)]:
            f = flag.loc[sub.index]
            a = int((sub.label_listed & f).sum()); b = int((~sub.label_listed & f).sum())
            c = int((sub.label_listed & ~f).sum()); dd = int((~sub.label_listed & ~f).sum())
            or_, lo, hi = odds_ratio(a, b, c, dd)
            p = fisher_exact([[a, b], [c, dd]])[1]
            rows.append({"measure": "label-listed", "rule": label, "subset": subset_name,
                         "flagged_n": a + b, "flagged_listed": a, "flagged_pct": round(100 * a / max(a + b, 1), 1),
                         "unflagged_n": c + dd, "unflagged_listed": c, "unflagged_pct": round(100 * c / max(c + dd, 1), 1),
                         "odds_ratio": round(or_, 2), "or_ci_low": round(lo, 2), "or_ci_high": round(hi, 2),
                         "fisher_p": float(f"{p:.3g}")})
            print(f"  {label:<40} {subset_name:<70} flagged {a:>4}/{a + b:<5} ({100 * a / max(a + b, 1):4.1f}%)  "
                  f"not flagged {c:>4}/{c + dd:<5} ({100 * c / max(c + dd, 1):4.1f}%)  OR {or_:.2f} ({lo:.2f}-{hi:.2f}) p={p:.2g}")

    flag = rules["Any-method rule (stored Signal flag)"]
    print("\nComposition of the flagged pairs (any-method rule)")
    comp = d.loc[flag].copy()
    comp["category"] = np.where(comp.term_group != "", comp.term_group,
                        np.where(~comp.covered, "Clinical event, drug not in SIDER 4.1",
                         np.where(comp.label_listed, "Clinical event, same Preferred Term on the label",
                          np.where(comp.label_related, "Clinical event, related term (same HLT) on the label",
                                   "Clinical event, not on the label"))))
    order = ["Indication of the drug (SIDER)"] + [g for g, _ in GROUPS] + [
        "Clinical event, same Preferred Term on the label", "Clinical event, related term (same HLT) on the label",
        "Clinical event, not on the label", "Clinical event, drug not in SIDER 4.1"]
    counts = comp.category.value_counts()
    for cat in order:
        n = int(counts.get(cat, 0))
        rows.append({"measure": "composition of flagged pairs", "rule": "Any-method rule (stored Signal flag)",
                     "subset": cat, "flagged_n": int(flag.sum()), "flagged_listed": n,
                     "flagged_pct": round(100 * n / flag.sum(), 1)})
        print(f"  {cat:<55} {n:>5}  ({100 * n / flag.sum():4.1f}%)")

    from sklearn.metrics import roc_auc_score
    clin = cov[cov.term_group == ""]
    print(f"\nRanking: do larger statistics mean label-listed? clinical-event pairs of covered drugs (n={len(clin)}, "
          f"{int(clin.label_listed.sum())} label-listed). AUC 0.5 = no relation, <0.5 = label-listed pairs rank lower")
    for col, label in [("prr", "PRR"), ("ror_ci_lower", "ROR 95% CI lower bound"), ("ic025", "IC025"),
                       ("eb05", "EB05"), ("n11", "Number of cases (n11)")]:
        for target, tname in [("label_listed", "same Preferred Term"), ("label_related", "same or related term (HLT)")]:
            if not clin[target].any():
                continue
            auc = roc_auc_score(clin[target], clin[col].fillna(0))
            rows.append({"measure": "AUC, label-listed vs not (clinical-event pairs, covered drugs)", "rule": label,
                         "subset": tname, "flagged_n": len(clin),
                         "flagged_listed": int(clin[target].sum()), "auc": round(auc, 3)})
            print(f"  {label:<26} {tname:<28} AUC {auc:.3f}  ({int(clin[target].sum())} of {len(clin)} listed)")
    top = clin.groupby("label_listed").pt_term.agg(lambda x: ", ".join(x.value_counts().head(6).index))
    print(f"  most frequent label-listed terms: {top.get(True, '')}")
    print(f"  most frequent unlisted terms:     {top.get(False, '')}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSummary written to {args.out}")
    if args.pairs_out:
        d.to_csv(args.pairs_out, index=False)
        print(f"Per-pair flags written to {args.pairs_out} (SIDER-derived: do not publish)")


if __name__ == "__main__":
    main()
