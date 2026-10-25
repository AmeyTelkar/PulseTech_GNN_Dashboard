"""
Reproduce the Signal flags reported in the PulseTech manuscript from the
deposited results.

PulseTech uses two versions of the signal rule, and this script shows which
published number comes from which:

  Stored-results rule (manuscript Methods; S8 Table "is_signal" column)
      n11 >= 3 AND ( PRR >= 2
                     OR (ROR >= 2 AND ROR 95% CI lower bound > 1)
                     OR IC025 > 0
                     OR EB05 > 2 )

  Code rule (is_signal() in signal_engine.py, read from that file unchanged)
      as above, but the PRR criterion also requires chi-square >= 4.

The code rule leaves unflagged exactly the pairs that the stored results flag
on the PRR point estimate alone (the 269 "lowest-priority leads" in the
manuscript), and applied to the CHILD-band graph it gives the 419 signal edges
shown by the dashboard's GNN Network View (Fig 6).

Usage (from the repository root or from code/):
    python code/verify_signal_flags.py
Needs: pandas, openpyxl.  Exit status 0 if every check passes, 1 otherwise.
"""
import ast
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S8 = ROOT / "Supplementary_Information" / "S8_Table_Complete_Signal_Detection.xlsx"
GRAPH = ROOT / "dataset" / "CHILD_Band_79Drugs_259ADRs_Graph.xlsx"
BANDS = ["NEONATE", "INFANT", "CHILD", "ADOLESCENT"]

# Values stated in the manuscript (V5).
EXPECTED = {
    "pairs evaluated, four resolved bands": 6259,
    "flagged, stored-results rule": 5266,
    "flagged and meeting Evans (PRR>=2, chi2>=4, n11>=3)": 4689,
    "flagged on the PRR point estimate alone": 269,
    "CHILD graph edges": 500,
    "CHILD graph signal edges, dashboard (Fig 6)": 419,
}


def load_code_rule():
    """Return is_signal() exactly as written in signal_engine.py.

    signal_engine.py imports backend modules that are not in this archive, so
    the file cannot be imported directly; only the is_signal function is
    compiled and executed.
    """
    src = (HERE / "signal_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_signal"]
    if len(fn) != 1:
        sys.exit("is_signal() not found in signal_engine.py")
    namespace = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "signal_engine.py", "exec"), namespace)
    return namespace["is_signal"]


def clean(value):
    """Spreadsheet blanks -> None, as the database would return them."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def stored_rule(row):
    if (row.get("n11") or 0) < 3:
        return False
    prr_ok = (row.get("prr") or 0) >= 2
    ror_ok = (row.get("ror") or 0) >= 2 and (row.get("ror_ci_lower") or 0) > 1
    ic_ok = row.get("ic025") is not None and row["ic025"] > 0
    eb_ok = row.get("eb05") is not None and row["eb05"] > 2
    return prr_ok or ror_ok or ic_ok or eb_ok


def evans(row):
    return (row.get("n11") or 0) >= 3 and (row.get("prr") or 0) >= 2 and (row.get("prr_chi2") or 0) >= 4


def main():
    code_rule = load_code_rule()
    results = {}
    failures = 0

    s8 = pd.read_excel(S8, sheet_name="Signal_Detection")
    frames = [s8[s8["age_group"] == band] for band in BANDS]
    rows = [{k: clean(v) for k, v in r.items()} for r in s8.to_dict("records")]

    published = [bool(r["is_signal"]) for r in rows]
    stored = [stored_rule(r) for r in rows]
    code = [code_rule(r) for r in rows]
    ev = [evans(r) for r in rows]

    agree = sum(p == s for p, s in zip(published, stored))
    print(f"S8 Table: stored-results rule matches the published flag for {agree:,} of {len(rows):,} pairs")
    if agree != len(rows):
        failures += 1

    print("\nPer band (pairs / flagged, stored rule / flagged, code rule):")
    for band, df in zip(BANDS, frames):
        sub = [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]
        print(f"  {band:<11}{len(sub):>6,}{sum(map(stored_rule, sub)):>8,}{sum(map(code_rule, sub)):>8,}")

    results["pairs evaluated, four resolved bands"] = len(rows)
    results["flagged, stored-results rule"] = sum(stored)
    results["flagged and meeting Evans (PRR>=2, chi2>=4, n11>=3)"] = sum(s and e for s, e in zip(stored, ev))
    prr_only = [s and not c for s, c in zip(stored, code)]
    results["flagged on the PRR point estimate alone"] = sum(prr_only)
    extra_code = sum(c and not s for s, c in zip(stored, code))

    # CHILD-band graph (500 edges)
    g = pd.read_excel(GRAPH, sheet_name="All_500_Edges", header=1)
    g = g.rename(columns={"Co-occur": "n11", "PRR": "prr", "ROR": "ror", "IC025": "ic025",
                          "EB05": "eb05", "Chi2": "prr_chi2", "ROR_CI_L": "ror_ci_lower"})
    grows = [{k: clean(v) for k, v in r.items()} for r in g.to_dict("records")]
    g_pub = sum(r["Signal"] == "SIGNAL" for r in grows)
    g_stored = sum(map(stored_rule, grows))
    g_code = sum(map(code_rule, grows))
    results["CHILD graph edges"] = len(grows)
    results["CHILD graph signal edges, dashboard (Fig 6)"] = g_code

    print("\nCheck against the manuscript:")
    for key, want in EXPECTED.items():
        got = results[key]
        ok = got == want
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {key}: {got:,} (manuscript {want:,})")

    print("\nHow the two rules relate:")
    print(f"  code rule flags {sum(code):,} pairs = {sum(stored):,} - {sum(prr_only):,} PRR-only flags;"
          f" pairs flagged by the code rule but not the stored rule: {extra_code}")
    print(f"  CHILD graph: stored flag {g_pub}, stored rule {g_stored}, code rule {g_code}")
    for r in grows:
        if stored_rule(r) and not code_rule(r):
            print(f"    edge flagged only by the stored rule: {r['Drug']} - {r['ADR']}"
                  f" (n11={r['n11']}, PRR={r['prr']}, chi2={r['prr_chi2']},"
                  f" ROR CI low={r['ror_ci_lower']}, IC025={r['ic025']})")
    if extra_code:
        failures += 1

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
