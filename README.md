# Clover Field Study — Analysis Repository

**Paper:** AI companion engagement and HIV testing and PrEP initiation among young people in South Africa: a real-world cohort study

---

## Overview

This repository contains the analysis scripts and generated outputs (figures, supplementary tables, summary JSON) for the Clover Field Study manuscript. The study evaluated engagement with the Aimee AI health companion and its association with HIV testing and PrEP uptake among platform users (analytic cohort: T&C-accepted users who sent at least one message during the study window).

All analyses are in Python 3 and are locally runnable with no cloud dependencies. The de-identified source CSVs are **not** committed (see [Data](#data)).

**Study window:** 17 March – 30 November 2025 (SAST, UTC+2).

---

## Analytic cohort

| Step | N |
|---|---|
| Sent a message during study window | 9,958 |
| Accepted terms and conditions | **9,310** (analytic cohort) |
| Meaningful engagers (≥2 active days) | 5,452 (58·6%) |
| Received Phithos HIV risk score | 1,260 (13·5%) |

All outcomes are anchored at the analytic cohort (n=9,310). Same-day events — outcomes occurring on the same calendar day as the patient's first Aimee message — are excluded throughout.

---

## Repository structure

```
├── Analysis scripts (Python)
│   ├── baseline_characteristics.py — Supplementary Table 1 (cohort baseline)
│   ├── engagement_intensity.py     — Span vs intensity; push-notification analysis (Section 2)
│   ├── threshold_analysis.py       — Dose-response across active-day thresholds
│   ├── section3_analysis.py        — HIV testing & PrEP cascade; time-to-event (Section 3)
│   ├── section4_analysis.py        — HCW engagement & triage analysis (Section 4)
│   ├── section5_analysis.py        — Longitudinal risk change (Section 5 / Supplementary)
│   ├── sensitivity_analysis.py     — Complete-case sensitivity vs main analysis
│   └── plot_funnel_original.py     — Engagement funnel figure
│
├── Generated figures (PNG)
│   ├── figure_4_cascade.png             — Care linkage cascade
│   ├── figure_5a_km_hiv.png             — KM curve — HIV testing
│   ├── figure_5b_km_prep.png            — KM curve — PrEP uptake
│   ├── figure_6a_nurse_contact.png      — HCW contact groups
│   ├── figure_6b_uptake_by_flag.png     — Uptake by Aimee flag type
│   ├── figure_7a_prep_by_risk_change.png— PrEP by risk transition
│   ├── figure_7b_risk_reduction_by_nurse.png — Risk reduction by nurse contact
│   ├── figure_7c_transition_sankey.png  — Risk transition Sankey
│   └── supp_figure_1_risk.png           — Supplementary risk figure
│
├── Generated tables (CSV)
│   ├── baseline_characteristics.csv     — written by baseline_characteristics.py
│   ├── supplementary_table_2.csv        — engagement–outcome AORs (main)
│   ├── supplementary_table_2b.csv       — complete-case sensitivity AORs
│   ├── supplementary_table_3.csv
│   ├── supplementary_table_4.csv
│   ├── supplementary_table_4_overall.csv
│   └── supplementary_table_5.csv
│
├── Summary outputs (JSON)
│   ├── section3_summary.json
│   ├── section4_summary.json
│   ├── section4_sensitivity.json
│   └── section5_summary.json
│
└── Aimee Engagement Analysis .ipynb     — exploratory notebook
```

> Source data CSVs (`ficus_*` and `clover_*` tables) are required to regenerate any output but are not committed — see [Data](#data).

---

## Key outcome definitions

### Primary HIV testing outcome
Union of four sources, restricted to events on a strictly later day than first Aimee message:

| Source | Type | Verification |
|---|---|---|
| S1: CBO clinic HIV test result | Post-Aimee only | Strong — clinic-verified |
| S2: Nurse-reviewed self-test upload | All (incl. same-day Wondfo) | Moderate — nurse-reviewed |
| S3: HIV status disclosed to Aimee | Post-disclosure timestamp | Weak — may pre-date engagement |
| S4: HIV test date disclosed to Aimee | Post-disclosure timestamp | Weak — may pre-date engagement |

Primary union: **3,484 (37·4%)**  
Conservative secondary (S1+S2 only): **1,714 (18·4%)**

> **Note on S3/S4:** The disclosure timestamp records when the patient *told Aimee*, not when the test occurred. The underlying test event may pre-date platform engagement. For Phithos risk subgroup analyses, CBO-verified outcomes are used instead of the union to avoid ceiling effects.

> **Wondfo acknowledgement:** HIV self-tests for S2 were donated by Wondfo Biotech Co., Ltd as part of community outreach recruitment events.

### Primary PrEP uptake outcome
Union of two sources, same-day excluded:

| Source | Type | Verification |
|---|---|---|
| PrEP S1: CBO dispensing record | Post-Aimee only | Strong |
| PrEP S2: Self-disclosed PrEP use | Post-disclosure | Weak |

Primary union: **1,124 (12·1%)**  
Conservative secondary (CBO-verified only): **810 (8·7%)**

---

## Engagement variables

All derived from user-role messages in `pre-assessment` (LLM-driven AI) conversations only.

> **Database naming note:** In this study's Metabase/database, `pre-assessment` = LLM-driven AI conversations; `static` = non-LLM menu/onboarding; `ifu` = self-test instruction flows. This is non-intuitive and critical for correct analysis.

| Variable | Definition |
|---|---|
| Distinct active days | Unique SAST calendar days with ≥1 user message |
| Engagement span | Days between first and last user message (0 for single-session) |
| Total messages | Count of user-role messages in study window |
| AI conversation sessions | Count of distinct pre-assessment sessions |

**Meaningful engagement threshold:** ≥2 distinct active days (n=5,452, 58·6%)  
Selected as the first and largest inflection point in the dose-response curve.

---

## Statistical methods

| Analysis | Method | Script |
|---|---|---|
| Engagement–outcome associations | Logistic regression, adjusted for registration month | `engagement_intensity.py` |
| Span vs intensity mutual adjustment | Log-transformed continuous variables (log1p), restricted to ≥2-day users | `engagement_intensity.py` |
| Threshold dose-response | AORs at ≥2 through ≥14 active-day thresholds | `threshold_analysis.py` |
| Time-to-event | Kaplan–Meier + Cox regression (verified outcomes only) | `section3_analysis.py` |
| Complete-case sensitivity | Re-fit on age/sex-disclosed subset; compared against main analysis | `sensitivity_analysis.py` |
| HCW group comparisons | Logistic regression, three-group classification | `section4_analysis.py` |
| Phithos ordinal trend | Ordinal logistic, risk coded 0/1/2 | `section3_analysis.py` |
| Longitudinal risk | Logistic regression, ≥7-day separation, adjusted for baseline risk + month | `section5_analysis.py` |
| Push notification analysis | Backward merge-asof, 48-hour window | `engagement_intensity.py` |

All analyses in Python 3 using pandas, numpy, statsmodels, lifelines, scipy, matplotlib.

---

## Key findings summary

| Finding | Result |
|---|---|
| Meaningful engagers vs single-session (HIV testing) | AOR 3·85 (3·49–4·23) |
| Meaningful engagers vs single-session (PrEP uptake) | AOR 3·28 (2·78–3·86) |
| Span AOR (HIV testing, mutually adjusted) | 1·33 (1·27–1·39) |
| Intensity AOR (HIV testing, mutually adjusted) | 10·69 (9·13–12·52) |
| Return messages preceded by push notification within 48h | 84·6% |
| Two-way nurse contact AOR (HIV testing) | 9·55 (8·51–10·71) |
| Two-way nurse contact AOR (PrEP uptake) | 3·97 (3·42–4·60) |
| PrEP p-trend across Phithos risk categories | 0·979 (non-significant) |

---

## Reproducing the analysis

### Requirements
```
Python 3.9+
pandas, numpy, statsmodels, lifelines, scipy, matplotlib
```

### Installation
```bash
pip install pandas numpy statsmodels lifelines scipy matplotlib
```

### Data
De-identified CSV exports from the South Africa Prod PostgreSQL database (`ficus_`- and `clover_`-prefixed tables) are required. These are **not** publicly available due to data-use agreements. Qualified researchers may request access from the corresponding author with appropriate institutional approvals.

Each script reads its inputs from `AUDERE_DATA_DIR` and writes outputs to `AUDERE_OUT_DIR`:

| Variable | Default | Purpose |
|---|---|---|
| `AUDERE_DATA_DIR` | `./data` (a `data/` folder beside the scripts) | location of the source CSVs |
| `AUDERE_OUT_DIR`  | the script's own directory | where figures/tables/JSON are written |

So either drop the CSVs into a `data/` folder next to the scripts, or point the env vars at wherever they live:

```bash
export AUDERE_DATA_DIR=/path/to/csvs
export AUDERE_OUT_DIR=/path/to/output     # optional; defaults to repo dir
```

### Running analyses
```bash
# Supplementary Table 1 — baseline characteristics
python3 baseline_characteristics.py

# Section 2 — engagement intensity + push-notification analysis
python3 engagement_intensity.py

# Active-day threshold dose-response
python3 threshold_analysis.py

# Section 3 — HIV testing & PrEP cascade + figures
python3 section3_analysis.py

# Section 4 — HCW engagement
python3 section4_analysis.py

# Section 5 — longitudinal risk (supplementary)
python3 section5_analysis.py

# Complete-case sensitivity (reads supplementary_table_2.csv from section3)
python3 sensitivity_analysis.py
```

> `sensitivity_analysis.py` compares against `supplementary_table_2.csv`, so run `section3_analysis.py` first.

---

## Important caveats

1. **Within-platform comparisons only.** No non-platform control group exists. All findings are associations between engagement level and outcomes within the platform cohort.

2. **Push notifications drive re-engagement.** 84·6% of return messages followed a platform push notification within 48 hours. Engagement span partly reflects the platform's own nudging infrastructure, not solely intrinsic health-seeking motivation.

3. **S3/S4 contamination.** Self-disclosed sources may incorporate test events that pre-dated Aimee engagement. Disclosure timestamps record when patients told Aimee, not when the test occurred.

4. **Phithos is not a validated clinical instrument.** The low/medium/high classification is generated by a proprietary platform algorithm. It functions as a proxy for disclosure depth as much as objective HIV risk.

5. **Generalisability.** Findings reflect Aimee operating within a well-resourced PEPFAR DREAMS outreach infrastructure with co-located nursing capacity. Results may not generalise to settings without this support.

---

## Citation

[Citation to be added upon acceptance]

## Data availability

Datasets are not publicly available due to data-use agreements but may be made available to qualified researchers upon reasonable request, with appropriate institutional approvals and a non-disclosure agreement.

## Ethics

_To be added._

## Acknowledgements

_To be added._

## License

_To be added._
