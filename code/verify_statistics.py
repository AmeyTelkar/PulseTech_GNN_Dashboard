"""
Recompute every statistic in S8 Table from its margins with the deposited
analytics modules.

S8 Table stores, for each of the 6,259 drug-ADR pairs, the counts n11, n1x,
nx1 and nxx. From these the 2x2 table follows, and this script recomputes
PRR, PRR chi-square, ROR and its CI, IC and IC025 (analytics_prr / _ror / _ic)
and EBGM and EB05 (analytics_ebgm, with the prior fitted per age band as
signal_engine.py does), then compares each value with the stored one.

Usage (from the repository root or from code/):
    python code/verify_statistics.py
Needs: pandas, openpyxl, numpy, scipy.  Exit status 0 if every check passes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analytics_ebgm import compute_ebgm, fit_prior  # noqa: E402
from analytics_ic import compute_ic  # noqa: E402
from analytics_prr import compute_prr  # noqa: E402
from analytics_ror import compute_ror  # noqa: E402

S8 = HERE.parent / "Supplementary_Information" / "S8_Table_Complete_Signal_Detection.xlsx"
GRAPH = HERE.parent / "dataset" / "CHILD_Band_79Drugs_259ADRs_Graph.xlsx"

failures = 0


def check(label, ok, detail=""):
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  {detail}"))


def agree(a, b, rtol):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    same = np.isclose(a, b, rtol=rtol, atol=1e-4) | (np.isnan(a) & np.isnan(b))
    return same.mean(), int((~same).sum())


d = pd.read_excel(S8, sheet_name="Signal_Detection")
print(f"S8 Table: {len(d):,} pairs")

cells = pd.DataFrame({"n11": d.n11, "n10": d.n1x - d.n11, "n01": d.nx1 - d.n11,
                      "n00": d.nxx - d.n1x - d.nx1 + d.n11})
check("stored n10, n01, n00 equal the cells implied by the margins",
      (cells[["n10", "n01", "n00"]].values == d[["n10", "n01", "n00"]].values).all())

rec = {k: [] for k in ["prr", "prr_chi2", "ror", "ror_ci_lower", "ic", "ic025"]}
for r in d.itertuples():
    ct = dict(n11=r.n11, n1x=r.n1x, nx1=r.nx1, nxx=r.nxx,
              n10=r.n1x - r.n11, n01=r.nx1 - r.n11, n00=r.nxx - r.n1x - r.nx1 + r.n11)
    out = {**compute_prr(ct), **compute_ror(ct), **compute_ic(ct)}
    for k in rec:
        rec[k].append(np.nan if out[k] is None else out[k])

print("\nPRR, ROR and IC recomputed from the margins")
for k in rec:
    frac, bad = agree(rec[k], d[k], 1e-4)
    check(f"{k}: {frac:.2%} identical", bad == 0, f"{bad} differ")

print("\nEBGM and EB05 recomputed (prior fitted per age band, truncated at n11 >= 3)")
eb, e05 = pd.Series(np.nan, index=d.index), pd.Series(np.nan, index=d.index)
for band, g in d.groupby("age_group"):
    E = (g.n1x * g.nx1 / g.nxx).to_numpy()
    prior = fit_prior(g.n11.to_numpy(), E, min_n=3)
    print(f"     {band:<11} prior (a1, b1, a2, b2, w) = {tuple(round(v, 4) for v in prior)}")
    for idx, r in zip(g.index, g.itertuples()):
        x = compute_ebgm(dict(n11=r.n11, n1x=r.n1x, nx1=r.nx1, nxx=r.nxx), prior=prior)
        eb[idx], e05[idx] = x["ebgm"], x["eb05"]
# The optimiser can stop at slightly different points on different machines, so
# agreement is checked to 1% (stored values are rounded to 4 decimals).
for name, got in [("ebgm", eb), ("eb05", e05)]:
    frac, bad = agree(got, d[name], 1e-2)
    check(f"{name}: {frac:.2%} within 1% of the stored value", frac >= 0.99, f"{bad} differ")

print("\nSignal flag")
P, R, RL, I = d.prr.fillna(0), d.ror.fillna(0), d.ror_ci_lower.fillna(0), d.ic025
other = (P >= 2) | ((R >= 2) & (RL > 1)) | (I > 0)
rule = (d.n11 >= 3) & (other | (d.eb05 > 2))
check(f"stored is_signal = any-method rule for all {len(d):,} pairs", bool((rule == d.is_signal).all()))
check(f"pairs flagged only by EB05 > 2: {int(((d.eb05 > 2) & ~other).sum())} (signal counts unchanged by EBGM)",
      int(((d.eb05 > 2) & ~other).sum()) == 0)

print("\nCHILD-band graph workbook")
g = pd.read_excel(GRAPH, sheet_name="All_500_Edges", header=1)
m = g.merge(d[d.age_group == "CHILD"], left_on=["Drug", "ADR"], right_on=["drugname_normalized", "pt_term"])
check("all 500 edges found in S8 Table (CHILD)", len(m) == 500)
check("edge EBGM and EB05 equal S8 Table", bool(np.isclose(m.EBGM, m.ebgm).all() and np.isclose(m.EB05, m.eb05).all()))

print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
