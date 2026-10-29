"""
Check the counts reported in the PulseTech manuscript against the files in
this archive (dataset/ workbooks and S8 Table).

Every expected value below is copied from the manuscript (V5); the script
recomputes it from the deposited data and prints PASS or FAIL. Numbers that
cannot be recomputed from the archive are listed at the end, so that a reader
knows which statements rest on the dashboard alone.

Usage (from the repository root or from code/):
    python code/verify_reported_counts.py
Needs: pandas, openpyxl.  Exit status 0 if every check passes, 1 otherwise.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
S8 = ROOT / "Supplementary_Information" / "S8_Table_Complete_Signal_Detection.xlsx"
BANDS = ["NEONATE", "INFANT", "CHILD", "ADOLESCENT"]

failures = 0


def check(label, got, want):
    global failures
    ok = got == want
    failures += not ok
    show = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {show(got)}" + ("" if ok else f"  (manuscript: {show(want)})"))


def pct(n, d, places=2):
    return f"{100 * n / d:.{places}f}%"


S8_ALL = pd.read_excel(S8, sheet_name="Signal_Detection")


def s8_band(band):
    return S8_ALL[S8_ALL["age_group"] == band]


# ---------------------------------------------------------------- Table 2
print("Table 2 - records per ICH E11(R1) age band")
demo = pd.read_excel(DATA / "Demographics_AgeBand_Distribution.xlsx", sheet_name="Summary", header=1)
demo["Age Group"] = demo["Age Group"].fillna("UNKNOWN")
band_n = dict(zip(demo["Age Group"], demo["Total Patients"].astype(int)))
table2 = {"NEONATE": (23704, "0.69%"), "INFANT": (29963, "0.88%"), "CHILD": (149389, "4.37%"),
          "ADOLESCENT": (403278, "11.79%"), "UNKNOWN": (2813867, "82.27%")}
total = sum(band_n.values())
check("total records", total, 3420201)
for band, (n, p) in table2.items():
    check(f"{band} records", band_n[band], n)
    check(f"{band} percentage", pct(band_n[band], total), p)
check("records in the four resolved bands", sum(band_n[b] for b in BANDS), 606334)

# ---------------------------------------------------------------- quarters
print("\nQuarters loaded and the 2025 trend paragraph")
q = pd.read_excel(DATA / "Quarterly_Reporting_Volume.xlsx", header=1)
expected_q = [f"{y}Q{i}" for y in range(2021, 2026) for i in range(1, 5)]
check("quarters loaded", len(q), 20)
check("quarters are 2021Q1-2025Q4 with none missing", list(q["quarter"]) == expected_q, True)
ledger = int(q["pediatric_rows"].sum())
check("load-ledger records summed over the 20 quarters (trend paragraph)", ledger, 3875098)
check("records superseded by later case versions (ledger sum minus deduplicated total)", ledger - total, 3875098 - 3420201)
trend = {"2025Q1": (168994, 4875, "2.9%"), "2025Q2": (166743, 5990, "3.6%"),
         "2025Q3": (189658, 6512, "3.4%"), "2025Q4": (155802, 6484, "4.2%")}
for quarter, (patients, signals, ratio) in trend.items():
    row = q[q["quarter"] == quarter].iloc[0]
    check(f"{quarter} patients", int(row["pediatric_rows"]), patients)
    check(f"{quarter} signals", int(row["signal_count"]), signals)
    check(f"{quarter} signal-to-patient ratio", pct(row["signal_count"], row["pediatric_rows"], 1), ratio)

# ---------------------------------------------------------------- CHILD graph
print("\nTable 1, Figs 6-7 - CHILD-band graph")
graph = DATA / "CHILD_Band_79Drugs_259ADRs_Graph.xlsx"
edges = pd.read_excel(graph, sheet_name="All_500_Edges", header=1)
nodes = pd.read_excel(graph, sheet_name="Drug_Node_Summary", header=1)
check("edges", len(edges), 500)
check("drug nodes", edges["Drug"].nunique(), 79)
check("ADR nodes", edges["ADR"].nunique(), 259)
check("duplicate drug-ADR edges", int(edges.duplicated(["Drug", "ADR"]).sum()), 0)
geno = nodes[nodes["Drug"] == "Genotropin"].iloc[0]
check("Genotropin connections (Fig 7)", int(geno["Connections"]), 37)
check("Genotropin flagged signals (Fig 7)", int(geno["Signals"]), 28)
print("  (signal-edge count 419: see verify_signal_flags.py)")

# ---------------------------------------------------------------- PS counts
print("\nPrimary-suspect report counts")
top = {b: pd.read_excel(DATA / "Top_Suspect_Drugs_By_AgeBand.xlsx", sheet_name=f"Top100_{b}", header=1)
       for b in BANDS}


def ps(band, drug):
    t = top[band]
    hit = t[t["Drug Name (Normalized)"].str.lower() == drug.lower()]["Report Count"]
    return int(hit.iloc[0]) if len(hit) else 0


check("Dupixent, CHILD band (Table 11 text)", ps("CHILD", "Dupixent"), 35043)
check("Genotropin, CHILD band (Fig 7)", ps("CHILD", "Genotropin"), 11303)

# Unknown-band share (Limitations), using the corpus-wide totals of Table 10
table10 = {"Dupixent": 135930, "Zantac": 99942, "Humira": 89238, "Oxycontin": 74738}
claims = {"Zantac": lambda s: s > 96, "Oxycontin": lambda s: s > 92, "Humira": lambda s: s > 92,
          "Dupixent": lambda s: round(s) == 11}
for drug, total_ps in table10.items():
    resolved = sum(ps(b, drug) for b in BANDS)
    share = 100 * (total_ps - resolved) / total_ps
    check(f"{drug}: PS reports with unresolved age ({share:.1f}%) matches the Limitations statement",
          claims[drug](share), True)

# ---------------------------------------------------------------- S8 Table
print("\nS8 Table")
s8 = S8_ALL
summary = pd.read_excel(S8, sheet_name="Summary_By_AgeBand")
check("pairs evaluated", len(s8), 6259)
check("every pair has n11 >= 3", bool((s8["n11"] >= 3).all()), True)
check("signal cache quarter is 2025Q1 for every pair (manuscript: Q1 2025)", set(s8["source_quarter"]), {"2025Q1"})
nxx = {"NEONATE": 993, "INFANT": 1890, "CHILD": 11307, "ADOLESCENT": 24756}
for band, n in nxx.items():
    check(f"{band}: patients in band, 2025Q1 (manuscript Methods)", int(s8_band(band)["nxx"].iloc[0]), n)
check("pairs meeting EB05 > 2 (manuscript Limitations)", int((s8["eb05"] > 2).sum()), 3516)
for _, row in summary.iterrows():
    band = row["age_group"]
    if band in BANDS:
        sub = s8_band(band)
        check(f"{band}: summary sheet pairs = rows", int(row["drug_ADR_pairs (n11>=3)"]), len(sub))
        check(f"{band}: summary sheet signals = flagged rows", int(row["flagged (is_signal)"]), int(sub["is_signal"].sum()))

# Table 9 - ten highest-PRR flagged signals
print("\nTable 9 - ten highest-PRR flagged signals")
table9 = [  # drug, ADR, band, n11, PRR, ROR (None = undefined), IC025, EBGM
    ("Oxbryta", "Sickle cell anaemia with crisis", "ADOLESCENT", 3, 12375, 24749, 0.26, 210.03),
    ("valacyclovir hydrochloride", "Derealisation", "ADOLESCENT", 3, 12375, 24749, 0.26, 210.03),
    ("gentamicin", "Deafness unilateral", "CHILD", 3, 11304, None, 0.26, 129.20),
    ("adenosine", "Sinus arrest", "CHILD", 3, 11304, None, 0.26, 129.20),
    ("levofloxacin", "Acute generalised exanthematous pustulosis", "ADOLESCENT", 3, 10607, 18561, 0.26, 206.88),
    ("Cystagon", "Amino acid level increased", "CHILD", 5, 9418, 56500, 1.42, 216.11),
    ("Pluvicto", "Prostatic specific antigen increased", "ADOLESCENT", 3, 9282, 37125, 0.26, 213.27),
    ("Xofluza", "Colitis ischaemic", "ADOLESCENT", 3, 9280, 14848, 0.26, 203.81),
    ("rosuvastatin", "Blood creatine phosphokinase MB increased", "ADOLESCENT", 5, 8836, 13745, 1.42, 322.33),
    ("rosuvastatin", "Chronic hepatitis C", "ADOLESCENT", 5, 8836, 13745, 1.42, 322.33),
]
ranked = s8[s8["is_signal"]].sort_values("prr", ascending=False)
cutoff = round(ranked["prr"].iloc[9], 4)
flagged = ranked[ranked["prr"].round(4) >= cutoff]  # top 10 plus any pair tied with the 10th


def key(r):
    ror = None if pd.isna(r["ror"]) else round(r["ror"])
    return (r["drugname_normalized"], r["pt_term"], r["age_group"], int(r["n11"]),
            round(r["prr"]), ror, round(r["ic025"], 2), round(r["ebgm"], 2))


got9 = {key(r) for _, r in flagged.iterrows()}
check("every Table 9 row (drug, ADR, band, n11, PRR, ROR, IC025, EBGM) is among the ten highest-PRR flags",
      set(table9) <= got9, True)
ties = [k for k in got9 - set(table9)]
check("pairs tied with the 10th row and not shown (named in the Table 9 caption)",
      [(k[0], k[1]) for k in ties], [("rosuvastatin", "Renal hypertension")])
shown = flagged[[key(r) in set(table9) for _, r in flagged.iterrows()]]
n11s = sorted(int(x) for x in shown["n11"])
check("Table 9 pairs resting on three cases / on five cases", (n11s.count(3), n11s.count(5)), (7, 3))
check("Table 9 pairs with undefined ROR", int(shown["ror"].isna().sum()), 2)

# Table 11 - top co-occurrences for Dupixent and Genotropin, CHILD band
print("\nTable 11 - top co-occurrences, Dupixent and Genotropin, CHILD band")
table11 = [("Dupixent", "Dermatitis atopic", 667), ("Genotropin", "Drug dose omission by device", 618),
           ("Dupixent", "Pruritus", 553), ("Dupixent", "Eczema", 469), ("Genotropin", "Device leakage", 416),
           ("Dupixent", "Rash", 409), ("Dupixent", "Dry skin", 319), ("Genotropin", "Device mechanical issue", 298),
           ("Dupixent", "Injection site pain", 295), ("Dupixent", "Off label use", 266)]
child = s8_band("CHILD")
two = child[child["drugname_normalized"].isin(["Dupixent", "Genotropin"])].sort_values("n11", ascending=False).head(10)
got11 = [(r["drugname_normalized"], r["pt_term"], int(r["n11"])) for _, r in two.iterrows()]
check("ten rows and their order identical to Table 11", got11 == table11, True)
deg = nodes.sort_values("Connections", ascending=False)["Drug"].head(2).tolist()
check("Dupixent and Genotropin are the two highest-degree drugs in the CHILD graph", sorted(deg), ["Dupixent", "Genotropin"])

# Worked example - shared ADRs
print("\nWorked example (Table 13 text)")
a = set(child[child["drugname_normalized"] == "Dupixent"]["pt_term"])
b = set(child[child["drugname_normalized"] == "idursulfase"]["pt_term"])
check("ADR preferred terms shared by Dupixent and idursulfase, CHILD band, n11 >= 3", len(a & b), 65)

# ---------------------------------------------------------------- not checkable
print("""
Not recomputable from this archive (dashboard or database values):
  - 1,902 suspect drugs, 4,416 unique adverse-event terms, 82,650 signals across all five bands
  - Table 3 outcome counts; Table 4 per-step row counts and timings
  - Table 10 corpus-wide primary-suspect totals (used above as stated)
  - Genotropin secondary-suspect count 3,828 (Fig 7 tooltip)
  - MedDRA 28.1 check (1,381 of 1,382 terms): needs the licensed MedDRA files""")

print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
