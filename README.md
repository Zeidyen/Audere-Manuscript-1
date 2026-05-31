# Clover Field Study — Analysis Repository

**Paper:** AI companion engagement and HIV testing and PrEP initiation among young people in South Africa: a real-world cohort study  I am 

---

## Overview

This repository contains all analysis scripts, figure generators, and document builders for the Clover Field Study manuscript. The study evaluated engagement with the Aimee AI health companion and its association with HIV testing and PrEP uptake among 9,310 platform users (analytic cohort: T&C-accepted users who sent at least one message during the study window).

All analyses are in Python 3. Document builders use Node.js (docx library). Scripts are locally runnable with no cloud dependencies.

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
├── Data sources (CSV, not committed — see Data section below)
│   ├── ficus_messages_clover_fieldstudy_updated.csv
│   ├── ficus_patient_state_clover_fieldstudy_updated.csv
│   ├── ficus_conversations_clover_fieldstudy_updated.csv
│   ├── ficus_hcw_conversations_clover_fieldstudy_updated.csv
│   ├── ficus_patient_tasks_clover_fieldstudy_updated.csv
│   ├── ficus_patient_profiles_clover_fieldstudy_updated.csv
│   ├── ficus_risk_assessments_clover_fieldstudy_updated.csv
│   ├── ficus_self_tests_clover_fieldstudy_updated.csv
│   └── ficus_patient_aggregation_clover_fieldstudy_updated.csv
│
├── Analysis scripts (Python)
│   ├── section3_analysis.py        — HIV testing and PrEP cascade (Section 3)
│   ├── section4_analysis.py        — HCW engagement and triage analysis (Section 4)
│   ├── section5_analysis.py        — Longitudinal risk change (Section 5 / Supplementary)
│   ├── engagement_intensity.py     — Span vs intensity, threshold analysis (Section 2)
│   ├── sensitivity_analysis.py     — IPW propensity score and E-value
│   ├── threshold_analysis.py       — Dose-response across active-day thresholds
│   ├── baseline_characteristics.py — Supplementary Table 1
│   └── plot_funnel_original.py     — Engagement funnel figure
│
├── JSON outputs (intermediate results)
│   ├── section3_summary.json
│   ├── section4_summary.json
│   ├── section5_summary.json
│   ├── span_intensity_results.json
│   ├── propensity_results.json
│   ├── evalue_results.json
│   └── baseline_corrections.json
│
├── Figures (PNG)
│   ├── engagement_funnel_plot.png  — Figure 1: Engagement funnel
│   ├── figure_2_forest_plot.png    — Figure 2: Threshold dose-response
│   ├── figure_3a_span_groups.png   — Figure 3A: Span categories
│   ├── figure_3b_three_convs.png   — Figure 3B: Span vs intensity (3-session)
│   ├── figure_3c_dose_response.png — Figure 3C: Active-day dose-response
│   ├── figure_4_cascade.png        — Figure 4: Care linkage cascade
│   ├── figure_5a_km_hiv.png        — Figure 5A: KM curve — HIV testing
│   ├── figure_5b_km_prep.png       — Figure 5B: KM curve — PrEP uptake
│   ├── figure_6a_nurse_contact.png — Figure 6A: HCW contact groups
│   ├── figure_6b_uptake_by_flag.png— Figure 6B: Uptake by Aimee flag type
│   ├── figure_7a_prep_by_risk.png  — Figure 7A: PrEP by risk transition
│   ├── figure_7b_risk_reduction.png— Figure 7B: Risk reduction by nurse contact
│   └── figure_7c_transition_sankey.png — Figure 7C: Risk transition Sankey
│
└── Document builders (Node.js)
    ├── build_results_docx.js       — Section 1 (cohort and engagement)
    ├── build_engagement_intensity_docx.js — Section 2
    ├── build_section3_v2.js        — Section 3
    ├── build_section4_docx.js      — Section 4
    ├── build_section5_docx.js      — Section 5
    ├── build_methods_v2.js         — Methods
    ├── build_discussion_v3.js      — Discussion
    ├── build_limitations_v2.js     — Limitations
    └── build_baseline_docx.js      — Baseline characteristics table
```

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
| Propensity-score sensitivity | IPW with post-baseline proxies; stabilised weights trimmed 1st–99th pct | `sensitivity_analysis.py` |
| E-value | VanderWeele & Ding (2017) formula; OR→RR conversion applied | `sensitivity_analysis.py` |
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
| IPW-adjusted PrEP AOR | 2·61 (2·22–3·37) |
| E-value PrEP CI lower bound | 3·6 |

---

## Reproducing the analysis

### Requirements
```
Python 3.9+
pandas, numpy, statsmodels, lifelines, scipy, matplotlib
Node.js 18+ (for document builders only)
```

### Installation
```bash
pip install pandas numpy statsmodels lifelines scipy matplotlib
cd /path/to/repo && npm install
```

### Data
De-identified CSV exports from the South Africa Prod PostgreSQL database (ficus_ and clover_ prefixed tables) are required. These are not publicly available due to data-use agreements. Qualified researchers may request access from the corresponding author with appropriate institutional approvals.

Place all CSV files in the same directory as the analysis scripts, or update the `UP` path variable at the top of each script.

### Running analyses
```bash
# Section 3 — HIV testing and PrEP cascade + figures
python3 section3_analysis.py

# Section 4 — HCW engagement
python3 section4_analysis.py

# Section 5 — Longitudinal risk (supplementary)
python3 section5_analysis.py

# Section 2 — Engagement intensity
python3 engagement_intensity.py

# Sensitivity analysis (IPW + E-value)
python3 sensitivity_analysis.py

# Engagement funnel figure
python3 plot_funnel_original.py
```

---

## Important caveats

1. **Within-platform comparisons only.** No non-platform control group exists. All findings are associations between engagement level and outcomes within the platform cohort.

2. **Push notifications drive re-engagement.** 84·6% of return messages followed a platform push notification within 48 hours. Engagement span partly reflects the platform's own nudging infrastructure, not solely intrinsic health-seeking motivation.

3. **IPW post-baseline limitation.** All three propensity-score proxies were measured during the same window as the exposure. IPW estimates should be interpreted as bounding analyses rather than unbiased causal estimates.

4. **S3/S4 contamination.** Self-disclosed sources may incorporate test events that pre-dated Aimee engagement. Disclosure timestamps record when patients told Aimee, not when the test occurred.

5. **Phithos is not a validated clinical instrument.** The low/medium/high classification is generated by a proprietary platform algorithm. It functions as a proxy for disclosure depth as much as objective HIV risk.

6. **Generalisability.** Findings reflect Aimee operating within a well-resourced PEPFAR DREAMS outreach infrastructure with co-located nursing capacity. Results may not generalise to settings without this support.

---

## Citation

[Citation to be added upon acceptance]

## Data availability

Please note that datasets are not publicly available due to data-use agreements but may be made available to qualified researchers upon reasonable request and with appropriate institutional approvals together with a non-disclosure. 

## Ethics

Tbd

## Acknowledgements

Tbd