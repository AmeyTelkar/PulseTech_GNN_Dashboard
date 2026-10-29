# PulseTech: Pediatric FAERS Pharmacovigilance Dashboard with Heterogeneous GNN Signal Exploration

**Full Title:** PulseTech: Design and implementation of a human-in-the-loop web dashboard for pediatric FAERS pharmacovigilance with heterogeneous graph neural network signal exploration

**Authors:** Telkar A, Telkar A, Javalgikar A, Ruikar D, Madanwale N.

**Journal:** PLOS ONE (submitted)

<!-- After the first GitHub release is archived by Zenodo, replace every ZENODO_CONCEPT_DOI in this file (badge line and Citation section) with the concept DOI, e.g. 10.5281/zenodo.1234567 -->
[![DOI](https://zenodo.org/badge/DOI/ZENODO_CONCEPT_DOI.svg)](https://doi.org/ZENODO_CONCEPT_DOI)

---

## Overview

This repository contains the code, derived datasets, supplementary information and figures for PulseTech, a web-based pharmacovigilance dashboard for pediatric reports in the FDA Adverse Event Reporting System (FAERS), 2021Q1–2025Q4. PulseTech combines a 13-step ingestion pipeline with a mandatory human quality gate, in-dashboard four-metric disproportionality analysis (PRR, ROR, IC/BCPNN, EBGM) across ICH E11(R1) pediatric age bands, and an interactive heterogeneous graph attention network (HANConv) view for exploring drug–adverse drug reaction (ADR) relationships.

Disproportionality signals are hypothesis-generating reporting associations, not causal findings.

---

## Repository Structure

```
.
├── code/                          # Backend source of PulseTech (see code/README.md)
│   ├── orchestrator.py            # 13-step ingestion pipeline
│   ├── validator.py, deduplicator.py, pipeline_*.py   # pipeline steps
│   ├── analytics_prr.py, _ror.py, _ic.py, _ebgm.py    # disproportionality statistics
│   ├── signal_engine.py           # 2x2 tables and four-metric signal flagging
│   ├── graph_builder.py           # Heterogeneous drug–ADR–cohort graph construction
│   ├── model.py                   # PediatricADRHAN (HANConv) model definition
│   ├── gnn_trainer.py, gnn_ablation.py  # Training and ablation
│   ├── evaluate_link_prediction.py # Held-out link-prediction evaluation (Table 12)
│   ├── gnn_link_prediction_*.csv  # Its per-seed results and summary
│   ├── validate_against_labels.py # Comparison of the signals with drug labels (S9 Table)
│   ├── drug_name_to_sider.csv, label_validation_summary.csv  # Its drug-name list and aggregate output
│   ├── predictor.py               # Candidate drug–ADR link ranking
│   ├── config.py, database.py, .env.example  # Settings (secrets only via .env)
│   ├── verify_statistics.py       # Recomputes every PRR, ROR, IC and EBGM in S8 Table from its counts
│   ├── verify_signal_flags.py     # Re-checks the S8 Table flags and the 419 graph count
│   ├── test_ebgm.py               # Tests the EBGM estimator against numerical integration
│   ├── verify_reported_counts.py  # Re-checks manuscript counts against dataset/ and S8 Table
│   ├── figure_sources/            # LaTeX/TikZ sources for Fig 3 and S1 Fig
│   ├── requirements.txt
│   └── README.md                  # Code documentation and reproducibility status
├── dataset/                       # Database exports used to verify the manuscript
│   └── README.md                  # What each workbook supports
├── Submission_Figures/            # Figures as submitted (TIFF / EPS) + PNG previews
├── Supplementary_Information/     # S1–S10 Tables (Excel) and S1 Fig
├── CITATION.cff
├── .zenodo.json                   # Zenodo metadata (authors, licence, keywords)
├── .gitattributes                 # keeps files byte-identical (no line-ending conversion)
├── LICENSE                        # MIT (code)
├── DATA_LICENSE.md                # CC BY 4.0 (data, tables, figures, documents)
└── README.md
```

---

## Key Results

| Quantity | Value |
|---|---|
| FAERS quarters loaded | 20 (2021Q1–2025Q4) |
| Patient records | 3,420,201 |
| Records with a resolved pediatric age band | 606,334 (Neonate 23,704 · Infant 29,963 · Child 149,389 · Adolescent 403,278) |
| Records with unresolved age (kept as an explicit Unknown band) | 2,813,867 (82.27%) |
| Drug–ADR pairs evaluated, four resolved bands, **Q1 2025 signal cache** | 6,259 |
| Flagged signals, four resolved bands, Q1 2025 | 5,266 |
| … of which also meet the Evans criteria | 4,689 (89.0%) |
| … pairs meeting EB05 > 2 (all also flagged by another criterion) | 3,516 |
| Flagged signals reported by the dashboard, all five bands | 82,650 |
| CHILD-band graph (Q1 2025) | 79 drug nodes, 259 ADR nodes, 500 edges |
| … signal edges shown in the dashboard (Fig 6) | 419 (83.8%) |
| Flagged pairs that are product-use, exposure, lack-of-effect, device or indication terms (drug labels, SIDER 4.1) | 1,461 of 5,266 (27.7%) |
| AUC of PRR / ROR lower bound / IC025 / EB05 for label-listed events (clinical-event pairs of covered drugs) | 0.32 / 0.35 / 0.38 / 0.31 (no statistic singles out label-listed reactions) |
| Held-out link prediction, CHILD band Q1 2025 (803 drugs, 2,537 ADRs, 11,274 edges; 5 seeds) | HANConv ROC-AUC 0.795 ± 0.011, AUPR 0.820 ± 0.013; common neighbours 0.796 / 0.816; Adamic–Adar 0.803 / 0.834; preferential attachment 0.810 / 0.861 |
| MedDRA check | 1,381 of 1,382 adverse-event terms are current MedDRA 28.1 Preferred Terms |

**Scope of the signal results:** S8 Table, the CHILD-band graph, the link-prediction evaluation and manuscript Tables 9, 11, 12 and 13 come from the dashboard's signal cache for the first quarter of 2025 (band denominators: 993 neonates, 1,890 infants, 11,307 children, 24,756 adolescents). They are not pooled across the 20 quarters.

**Important:** most records in the Unknown band are for predominantly adult products, so all-band figures describe the loaded data, not a pediatric population. Only results within the four resolved bands are pediatric findings.

---

## Signal Detection Criteria

A drug–ADR pair is flagged as a signal if **any** of the following is met. This is the rule that reproduces every flag in S8 Table.

| Metric | Criterion |
|---|---|
| PRR | PRR ≥ 2 (χ² is computed and reported but not required) |
| ROR | ROR ≥ 2 and 95% CI lower bound > 1 |
| IC | IC025 > 0 |
| EBGM | EB05 > 2 |

This any-method rule is more permissive than the conventional Evans criteria (PRR ≥ 2, χ² ≥ 4, n ≥ 3).

**Two forms of the rule.** `is_signal()` in `code/signal_engine.py` also requires χ² ≥ 4 for the PRR criterion. That stricter form flags the same pairs except the 269 that the stored results flag on the PRR point estimate alone (4,997 instead of 5,266), and applied to the 500 CHILD-band edges it gives the 419 signal edges shown by the dashboard's GNN Network View (Fig 6). The stored rule flags 420; the extra edge is Eligard – Weight increased (n11 = 8, PRR 2.01, χ² 3.05). `code/verify_signal_flags.py` demonstrates both.

---

## Reproducibility

| Component | Status |
|---|---|
| Signal-detection results (S8 Table) | Complete output with the counts n11, n1x, nx1, nxx; every PRR, ROR, IC and EBGM is recomputed from them by `code/verify_statistics.py` |
| Signal flags, signal counts and the 419 graph count | Re-checked by `code/verify_signal_flags.py` (runs from this archive) |
| Counts reported in the manuscript | Re-checked by `code/verify_reported_counts.py` against `dataset/` and S8 Table; the script lists the few numbers that rest on the dashboard alone |
| Comparison with drug labels (S9 Table) | Script, drug-name list and aggregate output included; rerunning needs your own SIDER 4.1 and MedDRA 28.1 copies |
| Link-prediction evaluation (Table 12) | Script and per-seed output included (`code/evaluate_link_prediction.py`, `code/gnn_link_prediction_*.csv`); rerunning it needs the loaded database |
| Pipeline, statistics, graph construction, model, training, link ranking | Source included (`code/`); running it needs PostgreSQL and the FAERS quarterly files |
| Fig 3 and S1 Fig | Regenerable from `code/figure_sources/` (pdflatex) |
| Figs 1, 2, 4–9 | Dashboard screenshots |
| Dashboard front end and web API (graph view, top-500 edge selection) | Not included |

### Known issues in this version

<!-- Update or remove each item when it is resolved. -->

1. **EBGM prior.** Only pairs with n11 ≥ 3 are stored, so the MGPS prior is fitted to a truncated sample; in all four bands the fit converges to an almost single-component prior. EBGM for pairs with very small expected counts is best read as a ranking. (The EBGM values were regenerated with the corrected estimator; `code/verify_statistics.py` reproduces them.)
2. **Single quarter.** The signal results are the Q1 2025 signal cache, not a pooled 20-quarter analysis.
3. **Front end.** The dashboard's web front end and web API are not included.
4. **Graph model.** On held-out CHILD-band edges the HANConv model performs comparably to common neighbours and below Adamic–Adar and preferential attachment; its rankings of unreported pairs are exploratory.
5. **Flags are not adverse reactions by default.** About a quarter of the flagged pairs are product-use, exposure, lack-of-effect, device or indication terms, and the statistics do not rank label-listed reactions higher (S9 Table). Review flags by term category.

IC is computed as log2((n11 + 0.5)/(E + 0.5)) (`code/analytics_ic.py`), as stated in the manuscript's Eq 2.

Training parameters in manuscript Table 5 and S4 Table (100 epochs, learning rate 0.01, weight decay 1e-4, 80/10/10 split) match `code/gnn_trainer.py`.

## Running the checks

Python 3.9+ with `pandas`, `openpyxl`, `numpy` and `scipy`:

```
pip install pandas openpyxl numpy scipy
python code/verify_statistics.py      # about 1 minute
python code/verify_signal_flags.py
python code/verify_reported_counts.py
python code/test_ebgm.py
```

Each script prints PASS or FAIL for every number it checks and exits with status 0 only if all pass.

---

## Data Availability

- **FAERS data:** publicly available from the [U.S. FDA](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
- **Signal-detection output:** complete results in `Supplementary_Information/S8_Table_Complete_Signal_Detection.xlsx`
- **Verification exports:** `dataset/`
- **MedDRA:** licensed from the MSSO; no MedDRA dictionary files are redistributed

---

## Citation

> Telkar A, Telkar A, Javalgikar A, Ruikar D, Madanwale N. PulseTech: Design and implementation of a human-in-the-loop web dashboard for pediatric FAERS pharmacovigilance with heterogeneous graph neural network signal exploration. *PLOS ONE* (submitted).

**Zenodo DOI:** [ZENODO_CONCEPT_DOI](https://doi.org/ZENODO_CONCEPT_DOI)

---

## License

- **Code** (`code/`): MIT License, see `LICENSE`.
- **Data, tables, figures and documents** (`dataset/`, `Supplementary_Information/`, `Submission_Figures/`, README files): Creative Commons Attribution 4.0 International (CC BY 4.0), see `DATA_LICENSE.md`.

FAERS data are a public FDA resource; MedDRA is licensed from the MSSO and no MedDRA dictionary files are included.
