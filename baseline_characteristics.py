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
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AUDERE_DATA_DIR", os.path.join(_HERE, "data"))
OUT_DIR  = os.environ.get("AUDERE_OUT_DIR", _HERE)

SAST = "Africa/Johannesburg"
START = pd.Timestamp("2025-03-17 00:00:00", tz=SAST)
END   = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)
 
def path(name): return os.path.join(DATA_DIR, name)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)
 
# ─── Load ────────────────────────────────────────────────────────────────────
print("Loading...")
pat = pd.read_csv(path("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
 
prof = pd.read_csv(path("ficus_patient_profiles_clover_fieldstudy_updated.csv"),
                   usecols=["patient_id","is_test_data","extracted_data",
                            "extracted_data → biological_sex → value"],
                   low_memory=False)
prof = prof[prof["is_test_data"] == False]
prof_json = prof[prof["extracted_data"].notna()].copy()
 
agg = pd.read_csv(path("ficus_patient_aggregation_clover_fieldstudy_updated.csv"), low_memory=False)
agg = agg[agg["is_test_data"] == False]
 
conv = pd.read_csv(path("ficus_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
conv = conv[conv["is_test_data"] == False]
 
llm = pd.read_csv(path("ficus_llm_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
llm = llm[llm["is_test_data"] == False]
llm["first_ts"] = to_sast(llm["first_recorded_message_timestamp"])
llm_in = llm[(llm["first_ts"] >= START) & (llm["first_ts"] <= END)]
 
msgs = pd.read_csv(path("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner","conversation_id"],
                   low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
msgs["date"] = msgs["sent_ts"].dt.date
msgs["hour"] = msgs["sent_ts"].dt.hour
 
risk = pd.read_csv(path("ficus_risk_assessments_clover_fieldstudy_updated.csv"), low_memory=False)
risk = risk[risk["is_test_data"] == False]
risk["ts"] = to_sast(risk["assessment_timestamp"])
 
pstate = pd.read_csv(path("ficus_patient_state_clover_fieldstudy_updated.csv"), low_memory=False)
pstate = pstate[pstate["is_test_data"] == False]
pstate["tc_ts"] = to_sast(pstate["first_terms_accepted_timestamp"])
 
# ─── Cohorts ────────────────────────────────────────────────────────────────
user_all = msgs[msgs["role"] == "user"]
all_pids = set(user_all["patient_id"].dropna().unique())
 
# Meaningful engagers: ≥2 distinct days with any user message
days_per_pid = user_all.groupby("patient_id")["date"].nunique()
engagers_pids = set(days_per_pid[days_per_pid >= 2].index)
 
# Scored subgroup
scored_records = risk[risk["risk_score_classification"].isin(["low","medium","high"])]
scored_pids = set(scored_records["patient_id"].dropna().unique())
 
print(f"  All platform users:  {len(all_pids):,}")
print(f"  Meaningful engagers: {len(engagers_pids):,}")
print(f"  Scored subgroup:     {len(scored_pids):,}")
 
def grab_value(json_str, key):
    try: return json.loads(json_str).get(key, {}).get("value")
    except: return None
 
for field in ["takes_prep","hiv_status","condom_usage_frequency","last_hiv_test","num_sexual_partners"]:
    prof_json[field] = prof_json["extracted_data"].apply(lambda s, k=field: grab_value(s, k))
 
ra = risk.copy()
ra["age_str"] = ra["llm_extracted_data_age"].astype(str)
ra_real = ra[~ra["age_str"].str.contains("assumed", case=False, na=False)]
ra_real["age_num"] = pd.to_numeric(ra_real["age_str"], errors="coerce")
ra_real = ra_real[(ra_real["age_num"] >= 13) & (ra_real["age_num"] <= 80)]
latest_age = ra_real.sort_values("ts").groupby("patient_id").tail(1)[["patient_id","age_num"]]
 
master = pd.DataFrame({"patient_id": sorted(all_pids | scored_pids)})
master = master.merge(latest_age, on="patient_id", how="left")
 
sex_df = prof[["patient_id","extracted_data → biological_sex → value"]].rename(
    columns={"extracted_data → biological_sex → value":"sex_raw"})
sex_df = sex_df[sex_df["sex_raw"].isin(["female","male","other"])]
master = master.merge(sex_df, on="patient_id", how="left")
 
tc_in = pstate[(pstate["tc_ts"] >= START) & (pstate["tc_ts"] <= END)].copy()
tc_in["reg_month"] = tc_in["tc_ts"].dt.month_name()
master = master.merge(tc_in[["patient_id","reg_month"]], on="patient_id", how="left")
 
active_days = user_all.groupby("patient_id")["date"].nunique().rename("active_days")
master = master.merge(active_days.reset_index(), on="patient_id", how="left")
 
total_msgs = user_all.groupby("patient_id").size().rename("total_msgs")
master = master.merge(total_msgs.reset_index(), on="patient_id", how="left")
 
ai_count = llm_in.groupby("patient_id").size().rename("ai_convs")
master = master.merge(ai_count.reset_index(), on="patient_id", how="left")
master["ai_convs"] = master["ai_convs"].fillna(0).astype(int)
 
# Span
t0 = user_all.groupby("patient_id")["sent_ts"].min()
t1 = user_all.groupby("patient_id")["sent_ts"].max()
span = ((t1 - t0).dt.total_seconds() / 86400.0).rename("span_days")
master = master.merge(span.reset_index(), on="patient_id", how="left")
 
# After-hours %
user_all_copy = user_all.copy()
user_all_copy["after_hours"] = (user_all_copy["hour"] < 8) | (user_all_copy["hour"] >= 17)
ah_pct = (user_all_copy.groupby("patient_id")["after_hours"].mean() * 100).rename("after_hours_pct")
master = master.merge(ah_pct.reset_index(), on="patient_id", how="left")
 
# Risk score (latest scored per patient — SAST-windowed for classification breakdown)
risk_window = risk[(risk["ts"] >= START) & (risk["ts"] <= END)]
scored_only_window = risk_window[risk_window["risk_score"].notna()].sort_values("ts").groupby("patient_id").tail(1)
master = master.merge(scored_only_window[["patient_id","risk_score","risk_score_classification"]],
                      on="patient_id", how="left")
 
# PrEP — patients who DISCLOSED takes_prep (True OR False)  → target 859
prep_disclosed_pids = set(prof_json.loc[prof_json["takes_prep"].notna(), "patient_id"].unique())
master["on_prep"] = master["patient_id"].isin(prep_disclosed_pids)
 
# HIV positive (hiv_status == 'positive')
hiv_pos_pids = set(prof_json.loc[prof_json["hiv_status"].astype(str).str.lower() == "positive",
                                  "patient_id"].unique())
master["hiv_pos"] = master["patient_id"].isin(hiv_pos_pids)
 
# Behavioural — from patient_profiles JSON
master = master.merge(prof_json[["patient_id","condom_usage_frequency","last_hiv_test","num_sexual_partners"]],
                      on="patient_id", how="left")
master["n_partners_num"] = pd.to_numeric(master["num_sexual_partners"], errors="coerce")
 
# Cohort flags
master["c_all"] = master["patient_id"].isin(all_pids)
master["c_eng"] = master["patient_id"].isin(engagers_pids)
master["c_scr"] = master["patient_id"].isin(scored_pids)
 
# Condom and HIV-test bands
def map_condom(v):
    if pd.isna(v): return None
    s = str(v).lower().strip()
    if s == "never": return "Never"
    if s == "sometimes": return "Sometimes"
    if s == "always": return "Always"
    if s == "not sexually active" or s == "not_sexually_active": return "Not sexually active"
    return None
 
def map_hiv(v):
    if pd.isna(v): return None
    s = str(v).lower().strip()
    return {"0_3_months":"Within 3 months",
            "3_6_months":"3 to 6 months",
            "6_12_months":"6 to 12 months",
            "more_than_12_months":"More than 12 months",
            "never":"Never tested"}.get(s)
 
master["condom_band"]   = master["condom_usage_frequency"].apply(map_condom)
master["hiv_test_band"] = master["last_hiv_test"].apply(map_hiv)
 
# ─── Build table ───────────────────────────────────────────────────────────
ORDER = ["All","Engagers","Scored"]
COHORT_LABELS = {
    "All":      f"All platform users\n(n = {len(all_pids):,})",
    "Engagers": f"Meaningful engagers\n(n = {len(engagers_pids):,})",
    "Scored":   f"Scored subgroup\n(n = {len(scored_pids):,})",
}
def get_cohort(name):
    if name == "All":      return master[master["c_all"]]
    if name == "Engagers": return master[master["c_eng"]]
    if name == "Scored":   return master[master["c_scr"]]
 
def med_iqr(s, dec=0):
    s = s.dropna()
    if len(s) == 0: return "—", 0
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.{dec}f} ({q1:.{dec}f}–{q3:.{dec}f})", len(s)
 
def n_pct(n, d):
    if d == 0: return "—"
    return f"{n:,} ({n/d*100:.1f}%)"
 
def med_pct(s, dec=0):
    s = s.dropna()
    if len(s) == 0: return "—"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.{dec}f}% ({q1:.{dec}f}–{q3:.{dec}f}%)"
 
rows = []
def add_section(t): rows.append({"kind":"section","label":t})
def add_row(label, vals):
    rows.append({"kind":"row","label":label,
                 "All":vals.get("All",""), "Engagers":vals.get("Engagers",""), "Scored":vals.get("Scored","")})
 
# Demographics
add_section("Demographics")
v = {}
for c in ORDER:
    coh = get_cohort(c)
    m, n = med_iqr(coh["age_num"])
    v[c] = f"{m}\n[n={n:,}, {n/len(coh)*100:.1f}%]" if len(coh) else "—"
add_row("Age — median (IQR), years", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c)
    disc = coh[coh["sex_raw"].isin(["female","male","other"])]
    fem = (disc["sex_raw"] == "female").sum()
    v[c] = f"{fem:,} ({fem/len(disc)*100:.1f}%)\nof {len(disc):,}" if len(disc) else "—"
add_row("Female sex — n (%)", v)
 
# Registration month
add_section("Registration month — n (%)")
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
    coh = get_cohort(c); m, _ = med_iqr(coh["active_days"]); v[c] = m
add_row("Distinct active days — median (IQR)", v)
 
for label, lo, hi in [("1 active day", 1, 1), ("2–3 active days", 2, 3), ("4+ active days", 4, None)]:
    v = {}
    for c in ORDER:
        coh = get_cohort(c)
        ad = coh["active_days"].fillna(0)
        n = ((ad >= lo) & (ad <= hi)).sum() if hi else (ad >= lo).sum()
        # For engagers, 1-active-day is non-applicable
        if c == "Engagers" and lo == 1 and hi == 1:
            v[c] = "—"
        else:
            v[c] = n_pct(n, len(coh))
    add_row(label, v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); m, _ = med_iqr(coh["total_msgs"]); v[c] = m
add_row("Total messages — median (IQR)", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); m, _ = med_iqr(coh["ai_convs"]); v[c] = m
add_row("AI conversations — median (IQR)", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); n = (coh["ai_convs"] >= 2).sum(); v[c] = n_pct(n, len(coh))
add_row("2+ AI conversation sessions", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); m, _ = med_iqr(coh["span_days"]); v[c] = m
add_row("Engagement span — median days (IQR)", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); v[c] = med_pct(coh["after_hours_pct"])
add_row("After-hours messages — median %", v)
 
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
        coh = get_cohort(c); n = (coh["condom_band"] == cat).sum()
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
        coh = get_cohort(c); n = (coh["hiv_test_band"] == cat).sum()
        v[c] = n_pct(n, denoms_h[c]) if denoms_h[c] else "—"
    add_row(f"  {cat}", v)
 
v = {}
for c in ORDER:
    coh = get_cohort(c); m, n = med_iqr(coh["n_partners_num"])
    v[c] = f"{m}\n[n={n:,}, {n/len(coh)*100:.1f}%]" if len(coh) else "—"
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
    coh = get_cohort(c)
    if c == "All":
        v[c] = "—"   # Not meaningful for all-users column
    else:
        m, _ = med_iqr(coh["risk_score"], dec=3); v[c] = m
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
    if c == "Scored":
        v[c] = f"{n:,} ({n/coh['c_scr'].sum()*100:.1f}% of scored)"
    else:
        v[c] = f"{n:,} ({n/len(coh)*100:.1f}% of all)"
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
print("\n" + "=" * 130)
print(f"{'Variable':<46}{COHORT_LABELS['All']:>28}{COHORT_LABELS['Engagers']:>28}{COHORT_LABELS['Scored']:>28}")
print("=" * 130)
for r in rows:
    if r["kind"] == "section":
        print(f"\n{r['label']}")
        continue
    a = r["All"].replace("\n","  ")
    e = r["Engagers"].replace("\n","  ")
    s = r["Scored"].replace("\n","  ")
    print(f"  {r['label']:<44}{a:>28}{e:>28}{s:>28}")
 
# Save CSV
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
