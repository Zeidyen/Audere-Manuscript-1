"""
SECTION 4 — Aimee and HCW engagement / referral
Clover Field Study (Aimee), South Africa — 17 March to 30 November 2025 (SAST)

Outputs:
  - figure_6a_nurse_contact.png    Two-way nurse contact rates by Aimee flag type
  - figure_6b_uptake_by_flag.png   HIV testing & PrEP uptake rates by flag type
  - supplementary_table_4.csv      Flag-level breakdown + overall HCW engagement table
  - section4_summary.json
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
})

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AUDERE_DATA_DIR", os.path.join(_HERE, "data"))
OUT_DIR  = os.environ.get("AUDERE_OUT_DIR", _HERE)
SAST = "Africa/Johannesburg"
START = pd.Timestamp("2025-03-17 00:00:00", tz=SAST)
END   = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)
def p(n): return os.path.join(DATA_DIR, n)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)
def gv(row, *keys):
    try:
        d = json.loads(row["extracted_data"]) if isinstance(row["extracted_data"], str) else row["extracted_data"]
        for k in keys: d = d[k]
        return d
    except Exception: return None

# ─── Cohort ─────────────────────────────────────────────────────────────────
print("Loading...")
msgs = pd.read_csv(p("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner"], low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs_in = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
user_msgs = msgs_in[msgs_in["role"] == "user"]
sent_pids = set(user_msgs["patient_id"].dropna().unique())
pstate = pd.read_csv(os.path.join(DATA_DIR, "ficus_patient_state_clover_fieldstudy_updated.csv"), low_memory=False)
pstate = pstate[pstate["is_test_data"]==False]
tc_pids = set(pstate[pstate["first_terms_accepted_timestamp"].notna()]["patient_id"].dropna().unique())
all_pids = sent_pids & tc_pids  # analytic cohort n=9,310
t0 = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
df = pd.DataFrame({"patient_id": sorted(all_pids)}).merge(t0.reset_index(), on="patient_id")
df["first_aimee_date"] = df["first_aimee_ts"].dt.date
df["reg_month"] = df["first_aimee_ts"].dt.month_name()
N = len(df)
print(f"Cohort: {N:,}")

# ─── HCW engagement categories ─────────────────────────────────────────────
hcw = pd.read_csv(p("ficus_hcw_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
hcw = hcw[hcw["is_test_data"] == False]
hcw["first_ts"] = to_sast(hcw["first_recorded_message_timestamp"])
hcw_in = hcw[(hcw["first_ts"] >= START) & (hcw["first_ts"] <= END)]

two_way_pids = set(hcw_in[(hcw_in["num_user_messages"] >= 1) &
                          (hcw_in["num_hcw_messages"] >= 1)]["patient_id"].unique()) & all_pids

# "HCW task generated but no patient response": tasks were created for the patient
# but they did not reciprocate in a conversation. Use the patient_tasks table to
# identify patients with any HCW outreach task in the window, then subtract two-way.
tasks = pd.read_csv(p("ficus_patient_tasks_clover_fieldstudy_updated.csv"), low_memory=False)
tasks = tasks[tasks["is_test_data"] == False]
tasks["task_ts"] = to_sast(tasks["task_created"])
tasks_in = tasks[(tasks["task_ts"] >= START) & (tasks["task_ts"] <= END)]
tasks_in = tasks_in[tasks_in["patient_id"].isin(all_pids)]

any_task_pids = set(tasks_in["patient_id"].unique())
one_way_pids = any_task_pids - two_way_pids
no_contact_pids = all_pids - two_way_pids - one_way_pids
print(f"  Two-way nurse: {len(two_way_pids):,} ({len(two_way_pids)/N*100:.1f}%)")
print(f"  Task only:     {len(one_way_pids):,} ({len(one_way_pids)/N*100:.1f}%)")
print(f"  No contact:    {len(no_contact_pids):,} ({len(no_contact_pids)/N*100:.1f}%)")

df["hcw_group"] = np.where(df["patient_id"].isin(two_way_pids), "two_way",
                  np.where(df["patient_id"].isin(one_way_pids), "task_only", "none"))

# ─── Outcomes (primary definitions, aligned with Section 3) ────────────────
pat = pd.read_csv(p("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
ctc = pd.read_csv(p("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])
st = pd.read_csv(p("ficus_self_tests_clover_fieldstudy_updated.csv"), low_memory=False)
st = st[st["is_test_data"] == False]
st["ts"] = to_sast(st["created"])
prof = pd.read_csv(p("ficus_patient_profiles_clover_fieldstudy_updated.csv"), low_memory=False)
prof = prof[prof["is_test_data"] == False]
prof["hiv_status_val"]    = prof.apply(lambda r: gv(r, "hiv_status", "value"), axis=1)
prof["last_hiv_test_val"] = prof.apply(lambda r: gv(r, "last_hiv_test", "value"), axis=1)
prof["takes_prep_val"]    = prof.apply(lambda r: gv(r, "takes_prep", "value"), axis=1)

ctc_v = ctc[ctc["test_result"].isin(["Negative","Positive","Discordant"])].copy()
ctc_v = ctc_v[(ctc_v["ts"] >= START) & (ctc_v["ts"] <= END)]
ctc_v["date"] = ctc_v["ts"].dt.date
ctc_v = ctc_v.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
s1 = set(ctc_v[ctc_v["date"] > ctc_v["first_aimee_date"]]["patient_id"].unique())

st_n = st[st["image_reviewer_interpretation"].notna()].copy()
st_n = st_n[(st_n["ts"] >= START) & (st_n["ts"] <= END)]
st_n["date"] = st_n["ts"].dt.date
st_n = st_n.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
s2 = set(st_n[st_n["date"] > st_n["first_aimee_date"]]["patient_id"].unique())

s3 = set(prof[prof["hiv_status_val"].isin(["negative","positive"])]["patient_id"].unique()) & all_pids
s4 = set(prof[prof["last_hiv_test_val"].isin(
    ["0_3_months","3_6_months","6_12_months","more_than_12_months"])]["patient_id"].unique()) & all_pids
primary_hiv = s1 | s2 | s3 | s4

prep_cbo = ctc[ctc["medication_type"] == "PrEP"].copy()
prep_cbo = prep_cbo[(prep_cbo["ts"] >= START) & (prep_cbo["ts"] <= END)]
prep_cbo["date"] = prep_cbo["ts"].dt.date
prep_cbo = prep_cbo.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
p1 = set(prep_cbo[prep_cbo["date"] > prep_cbo["first_aimee_date"]]["patient_id"].unique())
p2 = set(prof[prof["takes_prep_val"] == True]["patient_id"].unique()) & all_pids
primary_prep = p1 | p2

df["hiv"]  = df["patient_id"].isin(primary_hiv).astype(int)
df["prep"] = df["patient_id"].isin(primary_prep).astype(int)

# ─── Overall HCW group comparisons ─────────────────────────────────────────
print("\n=== HCW group comparisons (primary outcomes) ===")
table_overall = []
for grp, label in [("two_way","Two-way nurse conversation"),
                   ("task_only","HCW task generated, no patient response"),
                   ("none","No healthcare worker contact")]:
    sub = df[df["hcw_group"] == grp]
    hiv_rate  = sub["hiv"].mean()*100
    prep_rate = sub["prep"].mean()*100
    table_overall.append({
        "group": label, "n": len(sub),
        "hiv_n": int(sub["hiv"].sum()), "hiv_pct": hiv_rate,
        "prep_n": int(sub["prep"].sum()), "prep_pct": prep_rate,
    })
    print(f"  {label:<45} n={len(sub):>5}  HIV={hiv_rate:>5.1f}%  PrEP={prep_rate:>5.1f}%")

# Logistic regression: two-way vs no contact (reference)
df_lr = df[df["hcw_group"].isin(["two_way","none"])].copy()
df_lr["two_way"] = (df_lr["hcw_group"] == "two_way").astype(int)
hiv_m  = smf.logit("hiv ~ two_way + C(reg_month)", data=df_lr).fit(disp=False, method="bfgs", maxiter=200)
prep_m = smf.logit("prep ~ two_way + C(reg_month)", data=df_lr).fit(disp=False, method="bfgs", maxiter=200)
def or_ci(m, var):
    or_ = np.exp(m.params[var]); ci = np.exp(m.conf_int().loc[var]); pp = m.pvalues[var]
    return or_, ci[0], ci[1], pp
hiv_or, hiv_lo, hiv_hi, hiv_p   = or_ci(hiv_m, "two_way")
prep_or, prep_lo, prep_hi, prep_p = or_ci(prep_m, "two_way")
print(f"\nTwo-way nurse vs no contact (adjusted for registration month):")
print(f"  HIV testing:  AOR {hiv_or:.2f} (95% CI {hiv_lo:.2f}\u2013{hiv_hi:.2f}); p={hiv_p:.3g}")
print(f"  PrEP uptake:  AOR {prep_or:.2f} (95% CI {prep_lo:.2f}\u2013{prep_hi:.2f}); p={prep_p:.3g}")

# ─── Flag-level breakdowns ─────────────────────────────────────────────────
FLAGS = [
    ("Risk score",          ["concern_risk_score"]),
    ("Positive test",       ["concern_positive_test", "concern_self_reported_hiv_positive_status"]),
    ("Self-test interest",  ["interest_in_self_testing"]),
    ("PrEP/PEP interest",   ["possible_interest_in_prep_or_pep"]),
    ("Chat content",        ["concern_chat_content"]),
    ("Follow-up",           ["followup"]),
    ("Negative test",       ["concern_negative_test"]),
    ("Chat request",        ["request_to_chat"]),
]

print("\n=== Flag-level breakdown (n; nurse-contact %; HIV %; PrEP %) ===")
flag_rows = []
for label, types in FLAGS:
    pids = set(tasks_in[tasks_in["type"].isin(types)]["patient_id"].unique())
    sub = df[df["patient_id"].isin(pids)]
    n = len(sub)
    nurse_pct = (sub["patient_id"].isin(two_way_pids)).mean()*100 if n else 0
    hiv_pct   = sub["hiv"].mean()*100 if n else 0
    prep_pct  = sub["prep"].mean()*100 if n else 0
    flag_rows.append({
        "flag": label, "n": n,
        "nurse_pct": nurse_pct,
        "hiv_pct": hiv_pct, "prep_pct": prep_pct,
    })
    print(f"  {label:<22} n={n:>5}  nurse={nurse_pct:>5.1f}%  HIV={hiv_pct:>5.1f}%  PrEP={prep_pct:>5.1f}%")

flag_df = pd.DataFrame(flag_rows)
# Order by nurse contact for Figure 6A (descending)
flag_df_6a = flag_df.sort_values("nurse_pct", ascending=False).reset_index(drop=True)
# Use that same order for 6B to keep visual consistency with the draft
flag_df_6b = flag_df_6a.copy()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6A — Two-way nurse contact rate by flag type
# ═══════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 6A...")
fig, ax = plt.subplots(figsize=(10, 7))
labels = flag_df_6a["flag"].tolist()
vals = flag_df_6a["nurse_pct"].tolist()
ns = flag_df_6a["n"].tolist()
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, color="#5e7fa2", edgecolor="none")
for yi, v, n in zip(y, vals, ns):
    # Right-side percentage
    ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=13, fontweight="bold", color="#1a3a52")
    # Left-side n-label inside bar
    ax.text(2, yi, f"n={n:,}", va="center", fontsize=11, color="white")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, 100); ax.set_xlabel("Two-way nurse conversation (%)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_6a_nurse_contact.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6B — HIV testing & PrEP uptake rates by flag type
# ═══════════════════════════════════════════════════════════════════════════
print("Building Figure 6B...")
fig, ax = plt.subplots(figsize=(10, 7))
labels = flag_df_6b["flag"].tolist()
hiv_vals  = flag_df_6b["hiv_pct"].tolist()
prep_vals = flag_df_6b["prep_pct"].tolist()
y = np.arange(len(labels))[::-1]
bar_h = 0.36
# HIV bars (top, solid blue)
b1 = ax.barh(y + bar_h/2, hiv_vals,  bar_h, color="#5e7fa2", edgecolor="none", label="HIV testing uptake")
# PrEP bars (bottom, hatched green)
b2 = ax.barh(y - bar_h/2, prep_vals, bar_h, color="#5fa67f", edgecolor="white", hatch="///", label="PrEP uptake")

for yi, hv, pv in zip(y, hiv_vals, prep_vals):
    ax.text(hv + 0.4, yi + bar_h/2, f"{hv:.1f}%", va="center", fontsize=12, fontweight="bold", color="#1a3a52")
    ax.text(pv + 0.4, yi - bar_h/2, f"{pv:.1f}%", va="center", fontsize=12, color="#22593d")

ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Care uptake rate (%)")
ax.set_xlim(0, max(hiv_vals + prep_vals) * 1.18)
ax.legend(loc="lower right", frameon=True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_6b_uptake_by_flag.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── Supplementary Table 4 CSV ─────────────────────────────────────────────
flag_df_6a.to_csv(os.path.join(OUT_DIR, "supplementary_table_4.csv"), index=False)
pd.DataFrame(table_overall).to_csv(os.path.join(OUT_DIR, "supplementary_table_4_overall.csv"), index=False)

# ─── Summary JSON ──────────────────────────────────────────────────────────
summary = {
    "N": N,
    "two_way":   len(two_way_pids),
    "task_only": len(one_way_pids),
    "no_contact": len(no_contact_pids),
    "two_way_hiv_pct":   df[df["hcw_group"]=="two_way"]["hiv"].mean()*100,
    "two_way_prep_pct":  df[df["hcw_group"]=="two_way"]["prep"].mean()*100,
    "no_contact_hiv_pct":  df[df["hcw_group"]=="none"]["hiv"].mean()*100,
    "no_contact_prep_pct": df[df["hcw_group"]=="none"]["prep"].mean()*100,
    "task_only_hiv_pct":  df[df["hcw_group"]=="task_only"]["hiv"].mean()*100,
    "task_only_prep_pct": df[df["hcw_group"]=="task_only"]["prep"].mean()*100,
    "HIV_AOR":  (hiv_or, hiv_lo, hiv_hi, hiv_p),
    "PrEP_AOR": (prep_or, prep_lo, prep_hi, prep_p),
    "flags_6a": flag_df_6a.to_dict(orient="records"),
    "overall_rows": table_overall,
}
with open(os.path.join(OUT_DIR, "section4_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nDone. Wrote section4_summary.json")

# ════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS — Demographic adjustment (Supp Table 4b)
# ════════════════════════════════════════════════════════════════════════════
print("\n--- Sensitivity: adjust two-way nurse AOR for age + biological sex ---")

# Latest disclosed age + sex from risk_assessments
risk_in = risk[(risk["ts"] >= START) & (risk["ts"] <= END)]
risk_in = risk_in[risk_in["patient_id"].isin(all_pids)]
def to_num(x):
    try:
        v = float(str(x))
        if 10 <= v <= 100: return v
    except: pass
    return None
risk_in = risk_in.copy()
risk_in["age_num"] = risk_in["llm_extracted_data_age"].apply(to_num)
age_p = risk_in[risk_in["age_num"].notna()].sort_values("ts").groupby("patient_id").tail(1)[["patient_id","age_num"]]
sex_p = risk_in[risk_in["llm_extracted_data_biological_sex"].isin(["female","male"])].sort_values("ts").groupby("patient_id").tail(1)[["patient_id","llm_extracted_data_biological_sex"]].rename(columns={"llm_extracted_data_biological_sex":"sex"})

df_s = df.merge(age_p, on="patient_id", how="left").merge(sex_p, on="patient_id", how="left")
disclosing = df_s.dropna(subset=["age_num","sex"]).copy()
disclosing["two_way"] = (disclosing["hcw_group"]=="two_way").astype(int)
disclosing["female"]  = (disclosing["sex"]=="female").astype(int)

sub = disclosing[disclosing["hcw_group"].isin(["two_way","none"])].copy()
m1_hiv  = smf.logit("hiv ~ two_way + C(reg_month)",  data=sub).fit(disp=False, method="bfgs", maxiter=200)
m1_prep = smf.logit("prep ~ two_way + C(reg_month)", data=sub).fit(disp=False, method="bfgs", maxiter=200)
m2_hiv  = smf.logit("hiv ~ two_way + C(reg_month) + age_num + female",  data=sub).fit(disp=False, method="bfgs", maxiter=200)
m2_prep = smf.logit("prep ~ two_way + C(reg_month) + age_num + female", data=sub).fit(disp=False, method="bfgs", maxiter=200)

def or_ci(m, v):
    return float(np.exp(m.params[v])), float(np.exp(m.conf_int().loc[v]).iloc[0]), float(np.exp(m.conf_int().loc[v]).iloc[1]), float(m.pvalues[v])

results = {
    "N_disclosing_total": len(disclosing),
    "N_sensitivity_cohort": len(sub),
    "N_two_way_in_sub": int(sub["two_way"].sum()),
    "N_none_in_sub": int((1-sub["two_way"]).sum()),
    "median_age": float(disclosing["age_num"].median()),
    "iqr_age_lo": float(disclosing["age_num"].quantile(0.25)),
    "iqr_age_hi": float(disclosing["age_num"].quantile(0.75)),
    "pct_female": float((disclosing["sex"]=="female").mean()*100),
}
def fmtp(p): return "<0\u00b7001" if p < 0.001 else f"{p:.3f}".replace(".","\u00b7")
for k, m in [("m1_hiv",m1_hiv),("m1_prep",m1_prep),("m2_hiv",m2_hiv),("m2_prep",m2_prep)]:
    o,lo,hi,pp = or_ci(m, "two_way")
    results[k] = {"or":o, "lo":lo, "hi":hi, "p":pp, "p_fmt":fmtp(pp)}
for k, m, v in [("age_hiv",m2_hiv,"age_num"),("age_prep",m2_prep,"age_num"),
                ("sex_hiv",m2_hiv,"female"),("sex_prep",m2_prep,"female")]:
    o,lo,hi,pp = or_ci(m, v)
    results[k] = {"or":o, "lo":lo, "hi":hi, "p":pp, "p_fmt":fmtp(pp)}
with open(os.path.join(OUT_DIR, "section4_sensitivity.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"  Disclosing subset n = {len(disclosing):,}; sensitivity cohort n = {len(sub):,}")
print(f"  HIV  AOR: {results['m1_hiv']['or']:.2f} → {results['m2_hiv']['or']:.2f}  (reg-month only → +age+sex)")
print(f"  PrEP AOR: {results['m1_prep']['or']:.2f} → {results['m2_prep']['or']:.2f}")
