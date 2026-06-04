"""
SUPPLEMENTARY TABLE 2b — Complete-case sensitivity analysis
Clover Field Study, Aimee — 17 March to 30 November 2025 (SAST)

Restricts the cohort to patients with explicit age AND explicit sex disclosure,
then re-runs the same individually + mutually adjusted models, additionally
adjusting for age (continuous) and biological sex (categorical).

Purpose: demonstrate that the engagement-outcome associations from
Supplementary Table 2 hold within the disclosing subset and after
demographic adjustment.
"""

import os, json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AUDERE_DATA_DIR", os.path.join(_HERE, "data"))
OUT_DIR  = os.environ.get("AUDERE_OUT_DIR", _HERE)

SAST = "Africa/Johannesburg"
START = pd.Timestamp("2025-03-17 00:00:00", tz=SAST)
END   = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)

def p(n): return os.path.join(DATA_DIR, n)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)

# ─── Load core data ─────────────────────────────────────────────────────────
print("Loading...")
msgs = pd.read_csv(p("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner"], low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
msgs["date"] = msgs["sent_ts"].dt.date

llm = pd.read_csv(p("ficus_llm_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
llm = llm[llm["is_test_data"] == False]
llm["first_ts"] = to_sast(llm["first_recorded_message_timestamp"])
llm_in = llm[(llm["first_ts"] >= START) & (llm["first_ts"] <= END)]

pat = pd.read_csv(p("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
ctc = pd.read_csv(p("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])

st = pd.read_csv(p("ficus_self_tests_clover_fieldstudy_updated.csv"), low_memory=False)
st = st[st["is_test_data"] == False]
st["ts"] = to_sast(st["created"])

prof = pd.read_csv(p("ficus_patient_profiles_clover_fieldstudy_updated.csv"),
                   usecols=["patient_id","is_test_data","extracted_data",
                            "extracted_data → biological_sex → value"], low_memory=False)
prof = prof[prof["is_test_data"] == False]

risk = pd.read_csv(p("ficus_risk_assessments_clover_fieldstudy_updated.csv"), low_memory=False)
risk = risk[risk["is_test_data"] == False]
risk["ts"] = to_sast(risk["assessment_timestamp"])

# ─── Engagement variables ──────────────────────────────────────────────────
user_msgs = msgs[msgs["role"] == "user"]
all_pids = set(user_msgs["patient_id"].dropna().unique())
active_days = user_msgs.groupby("patient_id")["date"].nunique().rename("active_days")
t0 = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
t1 = user_msgs.groupby("patient_id")["sent_ts"].max().rename("last_aimee_ts")
span_days = ((t1 - t0).dt.total_seconds() / 86400.0).rename("span_days")
total_msgs = user_msgs.groupby("patient_id").size().rename("total_msgs")
ai_convs = llm_in.groupby("patient_id").size().rename("ai_convs")
reg_month = t0.dt.month_name().rename("reg_month")

df = pd.DataFrame({"patient_id": sorted(all_pids)})
for s in [active_days, span_days, total_msgs, ai_convs, t0, reg_month]:
    df = df.merge(s.reset_index(), on="patient_id", how="left")
df["ai_convs"] = df["ai_convs"].fillna(0).astype(int)

# ─── Outcomes (aligned to Section 3 primary definitions) ───────────────────
import json as _json
df["first_aimee_date"] = df["first_aimee_ts"].dt.date

# S1: CBO-verified clinic record (timing-corrected)
ctc_v = ctc[ctc["test_result"].isin(["Negative","Positive","Discordant"])].copy()
ctc_v = ctc_v[(ctc_v["ts"] >= START) & (ctc_v["ts"] <= END)]
ctc_v["date"] = ctc_v["ts"].dt.date
ctc_v = ctc_v.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
s1_pids = set(ctc_v[ctc_v["date"] > ctc_v["first_aimee_date"]]["patient_id"].unique())

# S2: Nurse-reviewed self-test (timing-corrected)
st_n = st[st["image_reviewer_interpretation"].notna()].copy()
st_n = st_n[(st_n["ts"] >= START) & (st_n["ts"] <= END)]
st_n["date"] = st_n["ts"].dt.date
st_n = st_n.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
s2_pids = set(st_n[st_n["date"] > st_n["first_aimee_date"]]["patient_id"].unique())
verified_pids = s1_pids | s2_pids

# S3 + S4: self-disclosed from patient_profiles
prof = pd.read_csv(p("ficus_patient_profiles_clover_fieldstudy_updated.csv"), low_memory=False)
prof = prof[prof["is_test_data"] == False]
def _get_v(row, *keys):
    try:
        d = _json.loads(row["extracted_data"]) if isinstance(row["extracted_data"], str) else row["extracted_data"]
        for k in keys: d = d[k]
        return d
    except Exception: return None
prof["hiv_status_val"]    = prof.apply(lambda r: _get_v(r, "hiv_status", "value"), axis=1)
prof["last_hiv_test_val"] = prof.apply(lambda r: _get_v(r, "last_hiv_test", "value"), axis=1)
prof["takes_prep_val"]    = prof.apply(lambda r: _get_v(r, "takes_prep", "value"), axis=1)
prof["care_linkage_hiv_testing_val"] = prof.apply(lambda r: _get_v(r, "care_linkage_hiv_testing", "value"), axis=1)

s3_pids = set(prof[prof["hiv_status_val"].isin(["negative","positive"])]["patient_id"].unique()) & all_pids
s4_pids = set(prof[prof["last_hiv_test_val"].isin(
    ["0_3_months","3_6_months","6_12_months","more_than_12_months"])]["patient_id"].unique()) & all_pids
# S5: care-linkage HIV-testing date self-disclosed to platform
s5_pids = set(prof[prof["care_linkage_hiv_testing_val"].notna() & ~prof["care_linkage_hiv_testing_val"].isin(
    ["unspecified","clientUnknown",""])]["patient_id"].unique()) & all_pids

# Primary HIV = union of all 5 sources
primary_pids = verified_pids | s3_pids | s4_pids | s5_pids

# Primary PrEP = CBO-verified + self-disclosed PrEP use
prep_cbo = ctc[ctc["medication_type"] == "PrEP"].copy()
prep_cbo = prep_cbo[(prep_cbo["ts"] >= START) & (prep_cbo["ts"] <= END)]
prep_cbo["date"] = prep_cbo["ts"].dt.date
prep_cbo = prep_cbo.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
p1_pids = set(prep_cbo[prep_cbo["date"] > prep_cbo["first_aimee_date"]]["patient_id"].unique())
p2_pids = set(prof[prof["takes_prep_val"] == True]["patient_id"].unique()) & all_pids
prep_pids = p1_pids | p2_pids

df["hiv"]  = df["patient_id"].isin(primary_pids).astype(int)
df["prep"] = df["patient_id"].isin(prep_pids).astype(int)

# ─── Demographic disclosure ────────────────────────────────────────────────
# Age — latest non-assumed disclosure from risk_assessments
ra = risk.copy()
ra["age_str"] = ra["llm_extracted_data_age"].astype(str)
ra_real = ra[~ra["age_str"].str.contains("assumed", case=False, na=False)].copy()
ra_real["age_num"] = pd.to_numeric(ra_real["age_str"], errors="coerce")
ra_real = ra_real[(ra_real["age_num"] >= 13) & (ra_real["age_num"] <= 80)]
latest_age = ra_real.sort_values("ts").groupby("patient_id").tail(1)[["patient_id","age_num"]]
df = df.merge(latest_age, on="patient_id", how="left")

# Sex — explicit only from patient_profiles
sex_df = prof[["patient_id","extracted_data → biological_sex → value"]].rename(
    columns={"extracted_data → biological_sex → value":"sex"})
sex_df = sex_df[sex_df["sex"].isin(["female","male","other"])]
df = df.merge(sex_df, on="patient_id", how="left")

# ─── Complete-case cohort ──────────────────────────────────────────────────
complete = df[df["age_num"].notna() & df["sex"].notna() & df["reg_month"].notna()].copy()
print(f"\nFull cohort:                       {len(df):,}")
print(f"Disclosed age (non-assumed):       {df['age_num'].notna().sum():,}")
print(f"Disclosed sex (explicit):          {df['sex'].notna().sum():,}")
print(f"Complete-case (both + reg_month):  {len(complete):,}")
print(f"  HIV testing events: {complete['hiv'].sum():,}")
print(f"  PrEP uptake events: {complete['prep'].sum():,}")

# Sex distribution in complete-case
print(f"\n  Sex distribution: {complete['sex'].value_counts().to_dict()}")
print(f"  Age summary: median {complete['age_num'].median():.0f}, "
      f"IQR {complete['age_num'].quantile(0.25):.0f}\u2013{complete['age_num'].quantile(0.75):.0f}")

# Drop "other" if very few (collapses to binary)
if (complete["sex"] == "other").sum() < 20:
    print(f"  Note: 'other' sex (n={(complete['sex']=='other').sum()}) collapsed into 'male' category for modelling stability")
    complete["sex_binary"] = (complete["sex"] == "female").astype(int)
    sex_term = "sex_binary"  # 1 = female (reference will be 0 = male/other)
else:
    sex_term = "C(sex, Treatment(reference='female'))"

# ─── Helpers ────────────────────────────────────────────────────────────────
ENG_VARS = ["active_days","span_days","total_msgs","ai_convs"]
ENG_LABELS = {"active_days":"Distinct active days (per day)",
              "span_days":"Engagement span (per day)",
              "total_msgs":"Total messages sent (per msg)",
              "ai_convs":"Number of AI conversations"}
OUTCOMES = [("hiv","HIV testing"), ("prep","PrEP uptake")]

def fmt_or(or_, lo, hi):
    return f"{or_:.3f} ({lo:.3f}\u2013{hi:.3f})"
def fmt_p(p):
    if p < 0.001: return "<0\u00b7001"
    return f"{p:.3f}".replace(".", "\u00b7")

def fit(formula, data):
    return smf.logit(formula, data=data).fit(disp=False, method="bfgs", maxiter=200)

# ─── Build Supp Table 2b ───────────────────────────────────────────────────
print("\n" + "=" * 100)
print("SUPPLEMENTARY TABLE 2b — Complete-case sensitivity (n = "
      f"{len(complete):,}, adjusted for reg_month + age + sex)")
print("=" * 100)

rows = []

# Individual models
for var in ENG_VARS:
    row = {"section":"Individually adjusted","variable":ENG_LABELS[var]}
    for ocol, oname in OUTCOMES:
        f = f"{ocol} ~ {var} + C(reg_month) + age_num + {sex_term}"
        m = fit(f, complete)
        or_ = np.exp(m.params[var]); ci = np.exp(m.conf_int().loc[var]); pp = m.pvalues[var]
        row[f"{oname} AOR"] = fmt_or(or_, ci[0], ci[1])
        row[f"{oname} p"]   = fmt_p(pp)
    rows.append(row)

# Mutually adjusted — REDUCED MODEL: only active_days + span_days (low VIF)
# Rationale: pre-analysis collinearity diagnostics (Supp Table 2) showed total_msgs
# and ai_convs were highly correlated with active_days (Pearson r > 0.84), with VIF
# values approaching/exceeding 5. Restricting the mutually adjusted model to the two
# low-VIF predictors (active_days + span_days, VIF 1.44) yields stable, interpretable
# estimates.
REDUCED_ENG = ["active_days","span_days"]
mut = {}
for ocol, oname in OUTCOMES:
    f = f"{ocol} ~ {' + '.join(REDUCED_ENG)} + C(reg_month) + age_num + {sex_term}"
    mut[ocol] = fit(f, complete)
for var in REDUCED_ENG:
    row = {"section":"Mutually adjusted","variable":ENG_LABELS[var]}
    for ocol, oname in OUTCOMES:
        or_ = np.exp(mut[ocol].params[var])
        ci  = np.exp(mut[ocol].conf_int().loc[var])
        pp  = mut[ocol].pvalues[var]
        row[f"{oname} AOR"] = fmt_or(or_, ci[0], ci[1])
        row[f"{oname} p"]   = fmt_p(pp)
    rows.append(row)

tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(OUT_DIR, "supplementary_table_2b.csv"), index=False)

# Print
print("\nIndividually adjusted (each variable + reg_month + age + sex)")
print("-" * 100)
print(f"{'Variable':<34}{'HIV testing AOR':>30}{'p':>10}{'PrEP uptake AOR':>30}{'p':>10}")
for r in [r for r in rows if r["section"]=="Individually adjusted"]:
    print(f"{r['variable']:<34}{r['HIV testing AOR']:>30}{r['HIV testing p']:>10}"
          f"{r['PrEP uptake AOR']:>30}{r['PrEP uptake p']:>10}")

print("\nMutually adjusted (all 4 variables + reg_month + age + sex)")
print("-" * 100)
print(f"{'Variable':<34}{'HIV testing AOR':>30}{'p':>10}{'PrEP uptake AOR':>30}{'p':>10}")
for r in [r for r in rows if r["section"]=="Mutually adjusted"]:
    print(f"{r['variable']:<34}{r['HIV testing AOR']:>30}{r['HIV testing p']:>10}"
          f"{r['PrEP uptake AOR']:>30}{r['PrEP uptake p']:>10}")

# ─── Side-by-side comparison with main analysis ────────────────────────────
print("\n" + "=" * 100)
print("Comparison: Main analysis (n=9,958) vs Sensitivity (n="+f"{len(complete):,})")
print("=" * 100)

main_tbl = pd.read_csv(os.path.join(OUT_DIR, "supplementary_table_2.csv"))
# supplementary_table_2.csv labels its sections "Individual"/"Mutually adjusted"
# and suffixes its AOR columns with "_aor" — map the sensitivity rows onto those.
SEC_MAP = {"Individually adjusted": "Individual", "Mutually adjusted": "Mutually adjusted"}
print(f"\n{'Section':<10}{'Variable':<32}{'Main HIV AOR':>22}{'Sens HIV AOR':>22}{'Main PrEP AOR':>22}{'Sens PrEP AOR':>22}")
print("-" * 130)
for r in rows:
    sec = r["section"][:8]
    var = r["variable"]
    match = main_tbl[(main_tbl["section"] == SEC_MAP.get(r["section"], r["section"])) & (main_tbl["variable"] == var)]
    if match.empty:
        continue
    main_row = match.iloc[0]
    print(f"{sec:<10}{var:<32}{main_row['HIV testing (primary)_aor']:>22}"
          f"{r['HIV testing AOR']:>22}{main_row['PrEP uptake_aor']:>22}{r['PrEP uptake AOR']:>22}")

# Also report the demographic effects themselves (informational)
print("\n" + "=" * 100)
print("Demographic effects in mutually adjusted models (informational)")
print("=" * 100)
for ocol, oname in OUTCOMES:
    print(f"\n{oname}:")
    m = mut[ocol]
    for term in ["age_num"] + ([sex_term] if sex_term == "sex_binary" else []):
        if term in m.params.index:
            or_ = np.exp(m.params[term]); ci = np.exp(m.conf_int().loc[term]); pp = m.pvalues[term]
            label = "Age (per year)" if term == "age_num" else "Female sex (vs male/other)"
            print(f"  {label:<32}{fmt_or(or_, ci[0], ci[1]):>22}  p={fmt_p(pp)}")

print("\nSaved: supplementary_table_2b.csv")
