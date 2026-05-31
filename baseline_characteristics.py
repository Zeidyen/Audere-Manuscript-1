"""
BASELINE CHARACTERISTICS — Clover Field Study
Study period: 17 March – 30 November 2025 (SAST, UTC+2)

Three cohorts:
  1. All platform users  — sent ≥1 user message in SAST study window
  2. Meaningful engagers — messaged Aimee on ≥2 distinct SAST days
                           (any non-HCW conv: pre-assessment + static + ifu)
  3. Scored subgroup     — received a low/medium/high Phithos risk score
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/mnt/user-data/uploads"
OUT_DIR  = "/home/claude/funnel"

SAST = "Africa/Johannesburg"
START = pd.Timestamp("2025-03-17 00:00:00", tz=SAST)
END   = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)

def path(name): return os.path.join(DATA_DIR, name)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)

print("Loading (UTC → SAST)...")

pat = pd.read_csv(path("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]

prof = pd.read_csv(path("ficus_patient_profiles_clover_fieldstudy_updated.csv"), low_memory=False)
prof = prof[prof["is_test_data"] == False]

agg = pd.read_csv(path("ficus_patient_aggregation_clover_fieldstudy_updated.csv"), low_memory=False)
agg = agg[agg["is_test_data"] == False]

conv = pd.read_csv(path("ficus_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
conv = conv[conv["is_test_data"] == False]
conv["first_ts"] = to_sast(conv["first_recorded_message_timestamp"])

llm = pd.read_csv(path("ficus_llm_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
llm = llm[llm["is_test_data"] == False]
llm["first_ts"] = to_sast(llm["first_recorded_message_timestamp"])
llm_in = llm[(llm["first_ts"] >= START) & (llm["first_ts"] <= END)]

msgs = pd.read_csv(path("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data",
                            "api_owner","conversation_id"],
                   low_memory=False)
msgs = msgs[msgs["is_test_data"] == False]
msgs = msgs.rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
ct_map = dict(zip(conv["conversation_id"], conv["conversation_type"]))
msgs["conv_type"] = msgs["conversation_id"].map(ct_map)
msgs["date"] = msgs["sent_ts"].dt.date
msgs["hour"] = msgs["sent_ts"].dt.hour

risk = pd.read_csv(path("ficus_risk_assessments_clover_fieldstudy_updated.csv"), low_memory=False)
risk = risk[risk["is_test_data"] == False]
risk["ts"] = to_sast(risk["assessment_timestamp"])

# ─── Cohorts ────────────────────────────────────────────────────────────────
user_msgs = msgs[msgs["role"] == "user"]

# All platform users who sent ≥1 message (funnel denominator = 9,958)
sent_pids = set(user_msgs["patient_id"].dropna().unique())

# Analytic cohort = T&C accepted ∩ sent message (n=9,310)
pstate = pd.read_csv(path("ficus_patient_state_clover_fieldstudy_updated.csv"), low_memory=False)
pstate = pstate[pstate["is_test_data"] == False]
tc_pids = set(pstate[pstate["first_terms_accepted_timestamp"].notna()]["patient_id"].dropna().unique())
all_pids = sent_pids & tc_pids  # analytic cohort — 9,310

# Meaningful engagers: ≥2 active days, any user message, within analytic cohort (broad, n=5,452)
days_per_pid = user_msgs.groupby("patient_id")["date"].nunique()
engagers_pids = set(days_per_pid[days_per_pid >= 2].index) & all_pids

scored_records = risk[(risk["ts"]>=START)&(risk["ts"]<=END)&(risk["risk_score_classification"].isin(["low","medium","high"]))]
scored_pids = set(scored_records["patient_id"].dropna().unique()) & all_pids

print(f"\nCohort sizes (analytic cohort = T&C accepted, n=9,310):")
print(f"  Analytic cohort:     {len(all_pids):,}")
print(f"  Meaningful engagers: {len(engagers_pids):,}  ({len(engagers_pids)/len(all_pids)*100:.1f}%)")
print(f"  Scored subgroup:     {len(scored_pids):,}  ({len(scored_pids)/len(all_pids)*100:.1f}%)")

# ─── Master patient frame ───────────────────────────────────────────────────
master = pd.DataFrame({"patient_id": sorted(all_pids | scored_pids)})

# Age — explicit (non-assumed) from risk_assessments, latest per patient
ra = risk.copy()
ra["age_str"] = ra["llm_extracted_data_age"].astype(str)
ra_real = ra[~ra["age_str"].str.contains("assumed", case=False, na=False)]
ra_real["age_num"] = pd.to_numeric(ra_real["age_str"], errors="coerce")
ra_real = ra_real[(ra_real["age_num"] >= 13) & (ra_real["age_num"] <= 80)]
latest_age = (ra_real.sort_values("ts").groupby("patient_id").tail(1)[["patient_id","age_num"]])
master = master.merge(latest_age, on="patient_id", how="left")

# Sex from patient_profiles (explicit only)
prof_sex = prof[["patient_id","extracted_data → biological_sex → value"]].rename(
    columns={"extracted_data → biological_sex → value":"sex_raw"})
prof_sex = prof_sex[prof_sex["sex_raw"].isin(["female","male","other"])]
master = master.merge(prof_sex, on="patient_id", how="left")

# First-message month (SAST)
first_msg = user_msgs.groupby("patient_id")["sent_ts"].min().reset_index()
first_msg["reg_month"] = first_msg["sent_ts"].dt.month_name()
master = master.merge(first_msg[["patient_id","reg_month"]], on="patient_id", how="left")

# Active days, user msg count, AI conv count, span, after-hours %
active_days = user_msgs.groupby("patient_id")["date"].nunique().rename("active_days")
master = master.merge(active_days.reset_index(), on="patient_id", how="left")

um_count = user_msgs[user_msgs["conv_type"].isin(["pre-assessment","static","ifu"])].groupby("patient_id").size().rename("user_msgs")
master = master.merge(um_count.reset_index(), on="patient_id", how="left")

ai_count = llm_in.groupby("patient_id").size().rename("ai_convs")
master = master.merge(ai_count.reset_index(), on="patient_id", how="left")

t0 = user_msgs.groupby("patient_id")["sent_ts"].min()
t1 = user_msgs.groupby("patient_id")["sent_ts"].max()
span = ((t1 - t0).dt.total_seconds() / 86400.0).rename("span_days")
master = master.merge(span.reset_index(), on="patient_id", how="left")

user_msgs["after_hours"] = (user_msgs["hour"] < 8) | (user_msgs["hour"] >= 17)
ah = (user_msgs.groupby("patient_id")["after_hours"].mean().rename("after_hours_pct") * 100)
master = master.merge(ah.reset_index(), on="patient_id", how="left")

# Risk score — latest scored per patient
scored_only = risk[risk["risk_score"].notna()].sort_values("ts").groupby("patient_id").tail(1)
master = master.merge(scored_only[["patient_id","risk_score","risk_score_classification"]],
                      on="patient_id", how="left")

# Behavioural — latest per patient from risk
def latest_field(col, new_name):
    sub = risk[risk[col].notna()].sort_values("ts").groupby("patient_id").tail(1)
    return sub[["patient_id", col]].rename(columns={col: new_name})

master = master.merge(latest_field("llm_extracted_data_condom_usage_frequency","condom"), on="patient_id", how="left")
master = master.merge(latest_field("llm_extracted_data_last_hiv_test","last_hiv_test"), on="patient_id", how="left")
master = master.merge(latest_field("llm_extracted_data_num_sexual_partners","n_partners"), on="patient_id", how="left")
master["n_partners_num"] = pd.to_numeric(master["n_partners"], errors="coerce")

# PrEP and HIV-positive flags from patient_profiles.extracted_data (JSON rollup)
print("Parsing patient_profiles JSON for PrEP / HIV status flags...")
import json as _json
prof_full = pd.read_csv(path("ficus_patient_profiles_clover_fieldstudy_updated.csv"),
                        usecols=["patient_id","is_test_data","extracted_data"],
                        low_memory=False)
prof_full = prof_full[(prof_full["is_test_data"] == False) & prof_full["extracted_data"].notna()]

def get_prep(s):
    try:    return _json.loads(s).get("takes_prep", {}).get("value") is True
    except: return False
def get_hiv_pos(s):
    try:
        d = _json.loads(s)
        st = d.get("hiv_status", {}).get("value")
        return str(st).lower() == "positive" if st is not None else False
    except: return False

prof_full["on_prep"] = prof_full["extracted_data"].apply(get_prep)
prof_full["hiv_pos"] = prof_full["extracted_data"].apply(get_hiv_pos)
prep_pids    = set(prof_full.loc[prof_full["on_prep"], "patient_id"].unique())
hiv_pos_pids = set(prof_full.loc[prof_full["hiv_pos"], "patient_id"].unique())

master["on_prep"] = master["patient_id"].isin(prep_pids)
master["hiv_pos"] = master["patient_id"].isin(hiv_pos_pids)

master["c_all"] = master["patient_id"].isin(all_pids)
master["c_eng"] = master["patient_id"].isin(engagers_pids)
master["c_scr"] = master["patient_id"].isin(scored_pids)

# Map categorical fields
def map_condom(v):
    if pd.isna(v): return None
    v = str(v).lower().strip()
    if "never" in v: return "Never"
    if "sometimes" in v or "occasion" in v: return "Sometimes"
    if "always" in v: return "Always"
    if "not sexually active" in v or "no sex" in v or "not_sex" in v: return "Not sexually active"
    return None

def map_hiv_test(v):
    if pd.isna(v): return None
    s = str(v).lower().strip()
    if s == "0_3_months":         return "Within 3 months"
    if s == "3_6_months":         return "3 to 6 months"
    if s == "6_12_months":        return "6 to 12 months"
    if s == "more_than_12_months":return "More than 12 months"
    if s == "never":              return "Never tested"
    return None  # 'clientUnknown' and anything else

master["condom_band"] = master["condom"].apply(map_condom)
master["hiv_test_band"] = master["last_hiv_test"].apply(map_hiv_test)

# ─── Build table ────────────────────────────────────────────────────────────
COHORT_LABELS = {
    "All":      f"Analytic cohort\n(n = {len(all_pids):,})",
    "Engagers": f"Meaningful engagers†\n(n = {len(engagers_pids):,})",
    "Scored":   f"Scored subgroup‡\n(n = {len(scored_pids):,})",
}
ORDER = ["All","Engagers","Scored"]

def get_cohort(name):
    if name == "All":      return master[master["c_all"]]
    if name == "Engagers": return master[master["c_eng"]]
    if name == "Scored":   return master[master["c_scr"]]

def med_iqr(series, dec=0):
    s = series.dropna()
    if len(s) == 0: return "—", 0
    q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
    return f"{med:.{dec}f} ({q1:.{dec}f}–{q3:.{dec}f})", len(s)

def n_pct(n, d):
    if d == 0: return "—"
    return f"{n:,} ({n/d*100:.1f}%)"

rows = []
def add_section(t): rows.append({"kind":"section","label":t})
def add_row(label, vals):
    rows.append({"kind":"row","label":label,
                 "All":vals.get("All",""), "Engagers":vals.get("Engagers",""), "Scored":vals.get("Scored","")})

# Demographics
add_section("Demographics")
age_vals = {}
for c in ORDER:
    coh = get_cohort(c)
    med, n_d = med_iqr(coh["age_num"])
    age_vals[c] = f"{med}\n[n={n_d:,}, {n_d/len(coh)*100:.1f}%]" if len(coh) else "—"
add_row("Age — median (IQR), years", age_vals)

sex_vals = {}
for c in ORDER:
    coh = get_cohort(c)
    disc = coh[coh["sex_raw"].isin(["female","male","other"])]
    fem = (disc["sex_raw"] == "female").sum()
    sex_vals[c] = f"{fem:,} ({fem/len(disc)*100:.1f}%)\nof {len(disc):,}" if len(disc) else "—"
add_row("Female sex — n (%)", sex_vals)

# Registration month
add_section("Registration month")
for m in ["March","April","May","June","July","August","September","October","November"]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        v[c] = n_pct((coh["reg_month"] == m).sum(), len(coh))
    add_row(m, v)

# Engagement profile
add_section("Engagement profile")
v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["active_days"]); v[c] = med
add_row("Distinct active days — median (IQR)", v)

for band, lo, hi in [("1 active day",1,1), ("2-3 active days",2,3), ("4+ active days",4,None)]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        ad = coh["active_days"].fillna(0)
        n = ((ad >= lo) & (ad <= hi)).sum() if hi else (ad >= lo).sum()
        v[c] = n_pct(n, len(coh))
    add_row(band, v)

v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["user_msgs"]); v[c] = med
add_row("User messages — median (IQR)", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["ai_convs"]); v[c] = med
add_row("AI conversations — median (IQR)", v)

v = {}
for c in ORDER:
    coh = get_cohort(c)
    n = (coh["ai_convs"].fillna(0) >= 2).sum()
    v[c] = n_pct(n, len(coh))
add_row("2+ AI conversation sessions", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["span_days"]); v[c] = med
add_row("Engagement span — median days (IQR)", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["after_hours_pct"], dec=1); v[c] = med
add_row("After-hours messages — median % (IQR)", v)

# Behavioural
add_section("Behavioural profile")
denoms_c = {}
v = {}
for c in ORDER:
    coh = get_cohort(c)
    nd = coh["condom_band"].notna().sum()
    denoms_c[c] = nd
    v[c] = n_pct(nd, len(coh))
add_row("Condom use frequency — n (%) with data", v)
for cat in ["Never","Sometimes","Always","Not sexually active"]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        n = (coh["condom_band"] == cat).sum()
        v[c] = n_pct(n, denoms_c[c]) if denoms_c[c] else "—"
    add_row(f"  {cat}", v)

denoms_h = {}
v = {}
for c in ORDER:
    coh = get_cohort(c)
    nd = coh["hiv_test_band"].notna().sum()
    denoms_h[c] = nd
    v[c] = n_pct(nd, len(coh))
add_row("Last HIV test recency — n (%) with data", v)
for cat in ["Within 3 months","3 to 6 months","6 to 12 months","More than 12 months","Never tested"]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        n = (coh["hiv_test_band"] == cat).sum()
        v[c] = n_pct(n, denoms_h[c]) if denoms_h[c] else "—"
    add_row(f"  {cat}", v)

v = {}
for c in ORDER:
    coh = get_cohort(c)
    med, n_d = med_iqr(coh["n_partners_num"])
    v[c] = f"{med}\n[n={n_d:,}, {n_d/len(coh)*100:.1f}%]" if len(coh) else "—"
add_row("Number of sexual partners — median (IQR)", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); n = coh["on_prep"].sum(); v[c] = n_pct(n, len(coh))
add_row("Self-reported on PrEP", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); n = coh["hiv_pos"].sum(); v[c] = n_pct(n, len(coh))
add_row("HIV positive — self-reported", v)

# Risk stratification
add_section("HIV risk stratification")
v = {}
for c in ORDER:
    coh = get_cohort(c); n = coh["c_scr"].sum(); v[c] = n_pct(n, len(coh))
add_row("Received risk score", v)

v = {}
for c in ORDER:
    coh = get_cohort(c); med, _ = med_iqr(coh["risk_score"], dec=3); v[c] = med
add_row("Risk score — median (IQR)", v)

for cls in ["low","medium","high"]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        n = (coh["risk_score_classification"] == cls).sum()
        d = coh["c_scr"].sum()
        v[c] = n_pct(n, d) if d else "—"
    add_row(f"  {cls.capitalize()}", v)

v = {}
for c in ORDER:
    coh = get_cohort(c)
    n = coh["risk_score_classification"].isin(["medium","high"]).sum()
    v[c] = f"{n:,} ({n/len(coh)*100:.1f}% of cohort)"
add_row("Medium or high risk", v)

risk_count = scored_records.groupby("patient_id").size()
two_plus = set(risk_count[risk_count >= 2].index)
v = {}
for c in ORDER:
    coh = get_cohort(c)
    n_scr = coh["c_scr"].sum()
    n_2p = coh["patient_id"].isin(two_plus).sum()
    v[c] = f"{n_2p:,} ({n_2p/n_scr*100:.1f}% of scored)" if n_scr else "—"
add_row("With 2+ risk assessments", v)

# ─── Print ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print(f"{'Variable':<46}{COHORT_LABELS['All']:>30}{COHORT_LABELS['Engagers']:>30}{COHORT_LABELS['Scored']:>30}")
print("=" * 140)
for r in rows:
    if r["kind"] == "section":
        print(f"\n{r['label']}")
        continue
    a = r["All"].replace("\n","  ")
    e = r["Engagers"].replace("\n","  ")
    s = r["Scored"].replace("\n","  ")
    print(f"  {r['label']:<44}{a:>30}{e:>30}{s:>30}")
# Checking this
# Save
out = []
for r in rows:
    if r["kind"] == "section":
        out.append({"variable": r["label"], "all":"","engagers":"","scored":""})
    else:
        out.append({"variable": r["label"],
                    "all":      r["All"].replace("\n"," | "),
                    "engagers": r["Engagers"].replace("\n"," | "),
                    "scored":   r["Scored"].replace("\n"," | ")})
pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "baseline_characteristics.csv"), index=False)
print(f"\nSaved: baseline_characteristics.csv")
