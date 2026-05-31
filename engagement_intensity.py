"""
ENGAGEMENT INTENSITY — Section 2 of Reach/Engagement results
Clover Field Study, Aimee — 17 March to 30 November 2025 (SAST)

Builds:
  - Supplementary Table 2: individual & mutually adjusted logistic regressions
  - Engagement variables: distinct active days, engagement span, total messages,
                          number of AI conversations
  - Outcomes:
      Primary   = HIV testing (any source: CTC + self-test, post-engagement)
      Secondary = CBO-verified testing (CTC test_result only)
      PrEP uptake (medication_type == 'PrEP' from CTC, post-engagement)
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/mnt/user-data/uploads"
OUT_DIR  = "/home/claude/funnel"

SAST = "Africa/Johannesburg"
START = pd.Timestamp("2025-03-17 00:00:00", tz=SAST)
END   = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)

def path(n): return os.path.join(DATA_DIR, n)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)

# ─── Load ────────────────────────────────────────────────────────────────────
print("Loading...")
msgs = pd.read_csv(path("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner","conversation_id"],
                   low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
msgs["date"] = msgs["sent_ts"].dt.date

llm = pd.read_csv(path("ficus_llm_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
llm = llm[llm["is_test_data"] == False]
llm["first_ts"] = to_sast(llm["first_recorded_message_timestamp"])
llm_in = llm[(llm["first_ts"] >= START) & (llm["first_ts"] <= END)]

pat = pd.read_csv(path("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
ctc = pd.read_csv(path("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])

st = pd.read_csv(path("ficus_self_tests_clover_fieldstudy_updated.csv"), low_memory=False)
st = st[st["is_test_data"] == False]
st["ts"] = to_sast(st["created"])

# ─── Cohort: all platform users (sent ≥1 user message in SAST window) ───────
user_msgs = msgs[msgs["role"] == "user"]
all_pids = set(user_msgs["patient_id"].dropna().unique())

# ─── Build patient-level engagement variables ───────────────────────────────
print("Building engagement variables...")

# Active days
active_days = user_msgs.groupby("patient_id")["date"].nunique().rename("active_days")

# Engagement span (days between first and last user message)
t0 = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
t1 = user_msgs.groupby("patient_id")["sent_ts"].max().rename("last_aimee_ts")
span_days = ((t1 - t0).dt.total_seconds() / 86400.0).rename("span_days")

# Total messages (all user messages, any conv type)
total_msgs = user_msgs.groupby("patient_id").size().rename("total_msgs")

# Number of AI conversations (LLM)
ai_convs = llm_in.groupby("patient_id").size().rename("ai_convs")

# Registration month
reg_month = t0.dt.month_name().rename("reg_month")

df = pd.DataFrame({"patient_id": sorted(all_pids)})
for s in [active_days, span_days, total_msgs, ai_convs, t0, reg_month]:
    df = df.merge(s.reset_index(), on="patient_id", how="left")
df["ai_convs"] = df["ai_convs"].fillna(0).astype(int)

# ─── Build outcomes ─────────────────────────────────────────────────────────
# Primary HIV testing — any source, post first Aimee message, in SAST window
ctc_test = ctc[ctc["test_result"].notna()][["patient_id","ts","test_result"]]
ctc_test = ctc_test.merge(df[["patient_id","first_aimee_ts"]], on="patient_id", how="inner")
ctc_post = ctc_test[(ctc_test["ts"] > ctc_test["first_aimee_ts"])
                    & (ctc_test["ts"] <= END)]
st_post = st.merge(df[["patient_id","first_aimee_ts"]], on="patient_id", how="inner")
st_post = st_post[(st_post["ts"] > st_post["first_aimee_ts"]) & (st_post["ts"] <= END)]

primary_pids   = set(ctc_post["patient_id"].unique()) | set(st_post["patient_id"].unique())
secondary_pids = set(ctc_post[ctc_post["test_result"].isin(
    ["Negative","Positive","Discordant"])]["patient_id"].unique())

# PrEP uptake
prep_records = ctc[ctc["medication_type"] == "PrEP"][["patient_id","ts"]]
prep_records = prep_records.merge(df[["patient_id","first_aimee_ts"]], on="patient_id", how="inner")
prep_post = prep_records[(prep_records["ts"] > prep_records["first_aimee_ts"])
                         & (prep_records["ts"] <= END)]
prep_pids = set(prep_post["patient_id"].unique())

df["hiv_primary"]   = df["patient_id"].isin(primary_pids).astype(int)
df["hiv_secondary"] = df["patient_id"].isin(secondary_pids).astype(int)
df["prep"]          = df["patient_id"].isin(prep_pids).astype(int)

print(f"\nCohort: {len(df):,} patients")
print(f"  HIV testing (primary, all sources):    {df['hiv_primary'].sum():,} patients")
print(f"  HIV testing (secondary, CBO-verified): {df['hiv_secondary'].sum():,} patients")
print(f"  PrEP uptake:                            {df['prep'].sum():,} patients")

# ─── Helpers ────────────────────────────────────────────────────────────────
def fit_logit(outcome, formula, data):
    return smf.logit(f"{outcome} ~ {formula}", data=data).fit(disp=False)

def aor_ci_p(model, var):
    or_ = np.exp(model.params[var])
    ci = np.exp(model.conf_int().loc[var])
    p = model.pvalues[var]
    return or_, ci[0], ci[1], p

def format_or(or_, lo, hi):
    return f"{or_:.3f} ({lo:.3f}\u2013{hi:.3f})"

# ─── Supplementary Table 2: Individual + Mutually adjusted ──────────────────
data = df[df["reg_month"].notna()].copy()

ENG_VARS = ["active_days","span_days","total_msgs","ai_convs"]
ENG_LABELS = {
    "active_days":  "Distinct active days (per day)",
    "span_days":    "Engagement span (per day)",
    "total_msgs":   "Total messages sent (per msg)",
    "ai_convs":     "Number of AI conversations",
}

OUTCOMES = [
    ("hiv_primary",   "HIV testing (primary)"),
    ("hiv_secondary", "HIV testing (CBO-verified)"),
    ("prep",          "PrEP uptake"),
]

print("\n" + "=" * 110)
print("SUPPLEMENTARY TABLE 2 — Engagement metrics ~ outcomes")
print("=" * 110)

table_rows = []

# Individual models
for var in ENG_VARS:
    row = {"section":"Individual","variable":ENG_LABELS[var]}
    for ocol, oname in OUTCOMES:
        m = fit_logit(ocol, f"{var} + C(reg_month)", data)
        or_, lo, hi, p = aor_ci_p(m, var)
        row[f"{oname}_aor"] = format_or(or_, lo, hi)
        row[f"{oname}_p"]   = f"{p:.3g}"
    table_rows.append(row)

# Mutually adjusted
mutual_models = {}
for ocol, oname in OUTCOMES:
    mutual_models[ocol] = fit_logit(ocol, " + ".join(ENG_VARS) + " + C(reg_month)", data)

for var in ENG_VARS:
    row = {"section":"Mutually adjusted","variable":ENG_LABELS[var]}
    for ocol, oname in OUTCOMES:
        or_, lo, hi, p = aor_ci_p(mutual_models[ocol], var)
        row[f"{oname}_aor"] = format_or(or_, lo, hi)
        row[f"{oname}_p"]   = f"{p:.3g}"
    table_rows.append(row)

tbl = pd.DataFrame(table_rows)

print("\nIndividually adjusted models (each variable + registration month)")
print("-" * 110)
print(f"{'Variable':<32}{'Primary HIV test (AOR)':>26}{'p':>8}"
      f"{'PrEP uptake (AOR)':>22}{'p':>8}")
for r in [r for r in table_rows if r["section"] == "Individual"]:
    print(f"{r['variable']:<32}{r['HIV testing (primary)_aor']:>26}{r['HIV testing (primary)_p']:>8}"
          f"{r['PrEP uptake_aor']:>22}{r['PrEP uptake_p']:>8}")

print("\nMutually adjusted (all 4 variables + registration month)")
print("-" * 110)
print(f"{'Variable':<32}{'Primary HIV test (AOR)':>26}{'p':>8}"
      f"{'PrEP uptake (AOR)':>22}{'p':>8}")
for r in [r for r in table_rows if r["section"] == "Mutually adjusted"]:
    print(f"{r['variable']:<32}{r['HIV testing (primary)_aor']:>26}{r['HIV testing (primary)_p']:>8}"
          f"{r['PrEP uptake_aor']:>22}{r['PrEP uptake_p']:>8}")

# CBO-verified columns separately
print("\nFor reference — CBO-verified HIV testing (secondary outcome)")
print("-" * 80)
print(f"{'Variable':<32}{'Individually':>22}{'p':>8}{'Mutually adj':>22}{'p':>8}")
for var in ENG_VARS:
    ind_row = [r for r in table_rows if r["section"]=="Individual" and r["variable"]==ENG_LABELS[var]][0]
    mut_row = [r for r in table_rows if r["section"]=="Mutually adjusted" and r["variable"]==ENG_LABELS[var]][0]
    print(f"{ENG_LABELS[var]:<32}{ind_row['HIV testing (CBO-verified)_aor']:>22}"
          f"{ind_row['HIV testing (CBO-verified)_p']:>8}"
          f"{mut_row['HIV testing (CBO-verified)_aor']:>22}"
          f"{mut_row['HIV testing (CBO-verified)_p']:>8}")

tbl.to_csv(os.path.join(OUT_DIR, "supplementary_table_2.csv"), index=False)
print("\nSaved: supplementary_table_2.csv")
