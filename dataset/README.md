# dataset/ — database exports used to verify the manuscript

Each workbook is a values-only export from the PulseTech PostgreSQL analytical store (FAERS 2021Q1–2025Q4). The CHILD-band graph, like S8 Table, comes from the signal cache for the first quarter of 2025. They contain no formulas. `../code/verify_reported_counts.py` re-checks the manuscript's counts against these files and S8 Table, and lists the few numbers that cannot be recomputed from the archive.

| File | Contents | Supports |
|---|---|---|
| `Demographics_AgeBand_Distribution.xlsx` | Records per ICH E11(R1) age band | Table 2: 3,420,201 records = 606,334 in the four resolved bands + 2,813,867 Unknown |
| `Quarterly_Reporting_Volume.xlsx` | Records per quarter | 20 quarters, 2021Q1–2025Q4 |
| `Drug_RoleCode_Distribution.xlsx` | Primary suspect / secondary suspect / concomitant / interacting counts per band | Role-code statements |
| `Top_Suspect_Drugs_By_AgeBand.xlsx` | Top 100 primary-suspect drugs per band | Dupixent CHILD count (35,043); Unknown-band composition in Limitations |
| `RxNorm_Normalization_Summary.xlsx` | Top 500 normalised drug names | RxNorm normalisation (Steps 10–12) |
| `CHILD_Band_79Drugs_259ADRs_Graph.xlsx` | CHILD-band graph (Q1 2025): node summary and all 500 edges with PRR, ROR, IC, EBGM, χ², interval bounds and Signal flag; EBGM/EB05 equal S8 Table | Fig 6, Fig 7, Table 11, Table 12 example |
| `MEDDRA_LICENSE_NOTE.txt` | MedDRA licensing note | — |

The complete signal-detection results (6,259 drug–ADR pairs, four resolved bands, Q1 2025) are in `../Supplementary_Information/S8_Table_Complete_Signal_Detection.xlsx`, with the counts from which every statistic can be recomputed.

## Notes

- The CHILD-band edge list flags 420 signal edges under the stored-results rule; the dashboard view in Fig 6 shows 419 because it applies the stricter rule in `../code/signal_engine.py` (PRR criterion also requires χ² ≥ 4). The one edge in between is Eligard – Weight increased (PRR 2.01, χ² 3.05). See `../code/verify_signal_flags.py`.
- EBGM and EB05 were regenerated with the corrected MGPS module; `../code/verify_statistics.py` checks them against S8 Table.
- Adverse-event terms are MedDRA Preferred Terms exactly as reported in FAERS. No MedDRA dictionary files are included.
