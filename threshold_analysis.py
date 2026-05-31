"""
THRESHOLD ANALYSIS — Active-day threshold for meaningful engagement
Clover Field Study, Aimee — 17 March to 30 November 2025 (SAST)

Validates the "≥2 active days = meaningful engagement" definition by
testing the dose-response of HIV testing uptake against the active-day
threshold, with logistic regression adjusted for registration month.

Outcomes tested:
  1. HIV testing (broad)        — any self-test recorded in SAST window
  2. HIV-verified testing (strict) — image-reviewer-verified result
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading data (UTC → SAST)...")

msgs = pd.read_csv(path("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner"],
                   low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
msgs["date"] = msgs["sent_ts"].dt.date

user_msgs = msgs[msgs["role"] == "user"]

# Cohort: all platform users (sent ≥1 user message in SAST window)
all_pids = set(user_msgs["patient_id"].dropna().unique())
print(f"  All platform users: {len(all_pids):,}")

# Active days per patient
active_days = user_msgs.groupby("patient_id")["date"].nunique().rename("active_days")

# Registration month (from first user message)
first_msg = user_msgs.groupby("patient_id")["sent_ts"].min()
reg_month = first_msg.dt.month_name().rename("reg_month")

# Self-tests within SAST window (self-administered tests)
st = pd.read_csv(path("ficus_self_tests_clover_fieldstudy_updated.csv"), low_memory=False)
st = st[st["is_test_data"] == False]
st["ts"] = to_sast(st["created"])

# Connection-to-care testing within SAST window — facility-recorded HIV tests
pat = pd.read_csv(path("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
ctc = pd.read_csv(path("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])

# Patient first Aimee message (engagement start)
first_msg = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
first_msg_df = first_msg.reset_index()

# POST-ENGAGEMENT outcomes: testing AFTER patient's first Aimee message,
# within the SAST study window.
# Broad: any HIV test record (ctc test_result OR self-test attempted)
ctc_test = ctc[ctc["test_result"].notna()][["patient_id","ts","test_result"]]
ctc_test = ctc_test.merge(first_msg_df, on="patient_id", how="inner")
ctc_post = ctc_test[(ctc_test["ts"] > ctc_test["first_aimee_ts"])
                    & (ctc_test["ts"] <= END)]

st_post = st.merge(first_msg_df, on="patient_id", how="inner")
st_post = st_post[(st_post["ts"] > st_post["first_aimee_ts"]) & (st_post["ts"] <= END)]

test_broad_pids = set(ctc_post["patient_id"].unique()) | set(st_post["patient_id"].unique())

# Strict: verified result (ctc Negative/Positive/Discordant) OR image-reviewer-verified self-test
ctc_verified_post = ctc_post[ctc_post["test_result"].isin(["Negative","Positive","Discordant"])]
st_verified_post = st_post[
    st_post["image_reviewer_interpretation"].notna()
    & (~st_post["image_reviewer_interpretation"].astype(str).str.contains(
        "uninterpretable|invalid", case=False, na=False))
]
test_strict_pids = (set(ctc_verified_post["patient_id"].unique())
                    | set(st_verified_post["patient_id"].unique()))

print(f"  Care-recorded test POST first Aimee: {len(set(ctc_post['patient_id'].unique())):,} patients")
print(f"  Self-test POST first Aimee:          {len(set(st_post['patient_id'].unique())):,} patients")
print(f"  Union (broad, post-engagement):      {len(test_broad_pids):,} patients")
print(f"  HIV-verified (strict, post-engage):  {len(test_strict_pids):,} patients")

# ─── Build patient-level analysis frame ─────────────────────────────────────
df = pd.DataFrame({"patient_id": sorted(all_pids)})
df = df.merge(active_days.reset_index(), on="patient_id", how="left")
df = df.merge(reg_month.reset_index(), on="patient_id", how="left")
df["active_days"] = df["active_days"].fillna(0).astype(int)
df["test_broad"]  = df["patient_id"].isin(test_broad_pids).astype(int)
df["test_strict"] = df["patient_id"].isin(test_strict_pids).astype(int)

print(f"\nAnalysis frame: {len(df):,} patients")

# ─── Dose-response: uptake by active-day threshold ──────────────────────────
def uptake_table(outcome):
    rows = []
    for thr_label, mask in [
        ("1 active day (reference)", df["active_days"] == 1),
        ("≥2 active days",            df["active_days"] >= 2),
        ("≥3 active days",            df["active_days"] >= 3),
        ("≥4 active days",            df["active_days"] >= 4),
        ("≥14 active days",           df["active_days"] >= 14),
    ]:
        sub = df[mask]
        n_total = len(sub)
        n_event = sub[outcome].sum()
        pct = n_event / n_total * 100 if n_total else 0
        rows.append({"threshold": thr_label, "n": n_total,
                     "events": n_event, "pct": round(pct, 1)})
    return pd.DataFrame(rows)

print("\n" + "=" * 80)
print("OUTCOME 1: Broad HIV testing (any self-test in SAST window)")
print("=" * 80)
u = uptake_table("test_broad")
print(u.to_string(index=False))

print("\n" + "=" * 80)
print("OUTCOME 2: HIV-verified testing (image-reviewer verified result)")
print("=" * 80)
u = uptake_table("test_strict")
print(u.to_string(index=False))

# ─── Logistic regression: ≥2 days vs 1 day, adjusted for reg_month ──────────
def run_logit(outcome, label):
    print(f"\n{label}")
    print("-" * 80)
    sub = df.copy()
    sub["group"] = (sub["active_days"] >= 2).astype(int)
    # Drop patients with missing reg_month
    sub = sub[sub["reg_month"].notna()]

    # Categorical formula
    model = smf.logit(f"{outcome} ~ group + C(reg_month)", data=sub).fit(disp=False)
    or_ = np.exp(model.params["group"])
    ci_lo, ci_hi = np.exp(model.conf_int().loc["group"])
    p = model.pvalues["group"]
    n_ref = ((sub["active_days"] == 1) & sub[outcome].astype(bool)).sum()
    n_ref_total = (sub["active_days"] == 1).sum()
    n_eng = ((sub["active_days"] >= 2) & sub[outcome].astype(bool)).sum()
    n_eng_total = (sub["active_days"] >= 2).sum()
    print(f"  1-day users:    {n_ref:>4}/{n_ref_total:<6}  ({n_ref/n_ref_total*100:.1f}%)")
    print(f"  ≥2-day users:   {n_eng:>4}/{n_eng_total:<6}  ({n_eng/n_eng_total*100:.1f}%)")
    print(f"  AOR (≥2 vs 1):  {or_:.2f}  (95% CI {ci_lo:.2f}–{ci_hi:.2f}; p<{p:.4f})")
    return {"outcome": label, "or": or_, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p,
            "ref_pct": n_ref/n_ref_total*100, "eng_pct": n_eng/n_eng_total*100}

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION — ≥2 vs 1 active day, adjusted for registration month")
print("=" * 80)

run_logit("test_broad",  "Outcome 1: any self-test (broad)")
run_logit("test_strict", "Outcome 2: HIV-verified test (strict)")

# ─── Continuous dose-response: each additional active day ──────────────────
print("\n" + "=" * 80)
print("CONTINUOUS DOSE-RESPONSE — each additional active day")
print("=" * 80)

for outcome, label in [("test_broad","Broad"), ("test_strict","Strict")]:
    sub = df[df["reg_month"].notna()].copy()
    model = smf.logit(f"{outcome} ~ active_days + C(reg_month)", data=sub).fit(disp=False)
    or_ = np.exp(model.params["active_days"])
    ci_lo, ci_hi = np.exp(model.conf_int().loc["active_days"])
    p = model.pvalues["active_days"]
    print(f"  {label:<10}  AOR per active day: {or_:.3f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f}; p<{p:.4f})")
