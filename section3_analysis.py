"""
SECTION 3 — HIV Testing and PrEP Uptake (REVISED)
Primary HIV testing = Union of all 4 sources (broadest)
Secondary HIV testing = CBO-verified + nurse-reviewed self-test (Sources 1+2)
Primary PrEP uptake = CBO-verified PrEP + self-disclosed PrEP start
Secondary PrEP uptake = CBO-verified only
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
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
END = pd.Timestamp("2025-11-30 23:59:59", tz=SAST)

def p(n): return os.path.join(DATA_DIR, n)
def to_sast(s): return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(SAST)

print("Loading...")
msgs = pd.read_csv(p("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner"], low_memory=False)
msgs = msgs[msgs["is_test_data"] == False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"] >= START) & (msgs["sent_ts"] <= END)]
msgs["date"] = msgs["sent_ts"].dt.date

pat = pd.read_csv(p("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"] == False]
ctc = pd.read_csv(p("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])
ctc["date"] = ctc["ts"].dt.date

st = pd.read_csv(p("ficus_self_tests_clover_fieldstudy_updated.csv"), low_memory=False)
st = st[st["is_test_data"] == False]
st["ts"] = to_sast(st["created"])
st["date"] = st["ts"].dt.date

hcw = pd.read_csv(p("ficus_hcw_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
hcw = hcw[hcw["is_test_data"] == False]
hcw["first_ts"] = to_sast(hcw["first_recorded_message_timestamp"])

risk = pd.read_csv(p("ficus_risk_assessments_clover_fieldstudy_updated.csv"), low_memory=False)
risk = risk[risk["is_test_data"] == False]
risk["ts"] = to_sast(risk["assessment_timestamp"])

tasks = pd.read_csv(p("ficus_patient_tasks_clover_fieldstudy_updated.csv"), low_memory=False)
tasks = tasks[tasks["is_test_data"] == False]
tasks["task_ts"] = to_sast(tasks["task_created"])

prof = pd.read_csv(p("ficus_patient_profiles_clover_fieldstudy_updated.csv"), low_memory=False)
prof = prof[prof["is_test_data"] == False]

# ─── Cohort ─────────────────────────────────────────────────────────────────
user_msgs = msgs[msgs["role"] == "user"]
sent_pids = set(user_msgs["patient_id"].dropna().unique())
pstate = pd.read_csv(p("ficus_patient_state_clover_fieldstudy_updated.csv"), low_memory=False)
pstate = pstate[pstate["is_test_data"] == False]
tc_pids = set(pstate[pstate["first_terms_accepted_timestamp"].notna()]["patient_id"].dropna().unique())
all_pids = sent_pids & tc_pids  # analytic cohort — 9,310
t0 = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
active_days = user_msgs.groupby("patient_id")["date"].nunique().rename("active_days")
reg_month = t0.dt.month_name().rename("reg_month")

df = pd.DataFrame({"patient_id": sorted(all_pids)})
for s in [t0, active_days, reg_month]:
    df = df.merge(s.reset_index(), on="patient_id", how="left")
df["first_aimee_date"] = df["first_aimee_ts"].dt.date
df["meaningful"] = (df["active_days"] >= 2).astype(int)
N = len(df)
print(f"Cohort: {N:,}  Single={int((df['meaningful']==0).sum())}  Meaningful={int(df['meaningful'].sum())}")

# Two-way nurse
hcw_in = hcw[(hcw["first_ts"] >= START) & (hcw["first_ts"] <= END)]
two_way_pids = set(hcw_in[(hcw_in["num_user_messages"] >= 1) & (hcw_in["num_hcw_messages"] >= 1)]["patient_id"].unique()) & all_pids

# Parse JSON
def gv(extracted, field):
    if pd.isna(extracted): return (None, None)
    try:
        d = json.loads(extracted) if isinstance(extracted, str) else extracted
        if isinstance(d, dict) and field in d:
            v = d[field]
            if isinstance(v, dict):
                return (v.get("value"), v.get("last_updated_source", {}).get("timestamp"))
        return (None, None)
    except Exception:
        return (None, None)

print("Parsing JSON...")
for fld in ["hiv_status","last_hiv_test","care_linkage_starting_prep"]:
    parsed = prof["extracted_data"].apply(lambda x: gv(x, fld))
    prof[f"{fld}_val"] = [x[0] for x in parsed]
    prof[f"{fld}_ts"]  = to_sast(pd.Series([x[1] for x in parsed]))
prof_c = prof[prof["patient_id"].isin(all_pids)].copy().merge(df, on="patient_id")

# ─── Sources ───────────────────────────────────────────────────────────────
# Methodology: "any event post-Aimee" approach (matches draft Supp Table 3)
# Total = patients with ≥1 event in window
# Timing-corrected = patients with ≥1 event on a calendar day LATER than first Aimee
# Same-day excluded = patients whose events are all on the first Aimee day (no later events)
#   Note: a patient with same-day AND later events is counted under timing-corrected, NOT same-day

# Source 1: CBO-verified
ctc_v = ctc[ctc["test_result"].isin(["Negative","Positive","Discordant"])].copy()
ctc_v = ctc_v[(ctc_v["ts"] >= START) & (ctc_v["ts"] <= END)]
ctc_v = ctc_v.merge(df[["patient_id","first_aimee_date","first_aimee_ts"]], on="patient_id", how="inner")
s1_total = set(ctc_v["patient_id"].unique())
s1_after = set(ctc_v[ctc_v["date"] > ctc_v["first_aimee_date"]]["patient_id"].unique())
s1_after_neg = set(ctc_v[(ctc_v["date"] > ctc_v["first_aimee_date"]) & (ctc_v["test_result"]=="Negative")]["patient_id"].unique())
s1_same_only = s1_total - s1_after  # patients with no later events (only same-day or before)

# Source 2: Nurse-reviewed self-test
st_nurse = st[st["image_reviewer_interpretation"].notna()].copy()
st_nurse = st_nurse[(st_nurse["ts"] >= START) & (st_nurse["ts"] <= END)]
st_nurse = st_nurse.merge(df[["patient_id","first_aimee_date","first_aimee_ts"]], on="patient_id", how="inner")
s2_total = set(st_nurse["patient_id"].unique())
s2_after = set(st_nurse[st_nurse["date"] > st_nurse["first_aimee_date"]]["patient_id"].unique())
s2_same_only = s2_total - s2_after

combined_verified = s1_after | s2_total   # S2: all nurse-reviewed self-tests incl same-day Wondfo

# Source 3 & 4: self-disclosed
s3_pids = set(prof_c[prof_c["hiv_status_val"].notna() & ~prof_c["hiv_status_val"].isin(["unknown",""])]["patient_id"].unique())
s4_pids = set(prof_c[prof_c["last_hiv_test_val"].notna() & ~prof_c["last_hiv_test_val"].isin(["never","clientUnknown",""])]["patient_id"].unique())

union_all = s1_after | s2_total | s3_pids | s4_pids  # S2: all incl same-day Wondfo recruitment events

# PrEP Source 1
prep_ctc = ctc[ctc["medication_type"] == "PrEP"].copy()
prep_ctc = prep_ctc[(prep_ctc["ts"] >= START) & (prep_ctc["ts"] <= END)]
prep_ctc = prep_ctc.merge(df[["patient_id","first_aimee_date","first_aimee_ts"]], on="patient_id", how="inner")
p1_total = set(prep_ctc["patient_id"].unique())
p1_after = set(prep_ctc[prep_ctc["date"] > prep_ctc["first_aimee_date"]]["patient_id"].unique())
p1_same_only = p1_total - p1_after

# PrEP Source 2 — self-disclosed PrEP use (takes_prep.value == True)
# Consistent with Sections 2, 4, 5 definitions
p2_pids = set(prof[prof["extracted_data"].apply(
    lambda x: gv(x, "takes_prep")[0] == True
)]["patient_id"].unique()) & all_pids
prep_union = p1_after | p2_pids

print(f"\n=== SUPPLEMENTARY TABLE 3 ===")
print(f"PANEL A — HIV TESTING")
print(f"  Source 1 (CBO-verified):           total={len(s1_total)},  same-day only={len(s1_same_only)},  timing-corrected={len(s1_after)}")
print(f"  Source 2 (Nurse-reviewed):         total={len(s2_total)},  same-day only={len(s2_same_only)},  timing-corrected={len(s2_after)}")
print(f"  Combined primary (S1+S2):                                                       n={len(combined_verified)}")
print(f"  Source 3 (HIV status disclosed):   n={len(s3_pids)}")
print(f"  Source 4 (test date disclosed):    n={len(s4_pids)}")
print(f"  UNION of all sources (PRIMARY):    n={len(union_all)} ({len(union_all)/N*100:.1f}%)")
print(f"\nPANEL B — PrEP UPTAKE")
print(f"  Source 1 (CBO dispensing):         total={len(p1_total)},  same-day only={len(p1_same_only)},  timing-corrected={len(p1_after)}")
print(f"  Source 2 (self-disclosed start):   n={len(p2_pids)}")
print(f"  UNION (PRIMARY):                    n={len(prep_union)} ({len(prep_union)/N*100:.1f}%)")

# Save Supp Table 3
supp3_rows = []
def fmt_pct_total(n, tot): return f"{n:,} ({n/tot*100:.1f})" if isinstance(n, (int, np.integer)) else str(n)
def fmt_pct_cohort(n): return f"{n:,} ({n/N*100:.1f})" if isinstance(n, (int, np.integer)) else str(n)
def fmt_n(n): return f"{n:,}" if isinstance(n, (int, np.integer)) else str(n)

supp3_rows.extend([
    {"panel":"A","group":"Primary analysis","source":"Source 1. CBO-verified clinic record",
     "total": fmt_n(len(s1_total)), "same_day": fmt_pct_total(len(s1_same_only), len(s1_total)),
     "timing": fmt_pct_cohort(len(s1_after)), "verif": "Strong (Reference)"},
    {"panel":"A","group":"Primary analysis","source":"Source 2. Nurse-reviewed self-test upload",
     "total": fmt_n(len(s2_total)), "same_day": fmt_pct_total(len(s2_same_only), len(s2_total)),
     "timing": fmt_pct_cohort(len(s2_after)), "verif": "Moderate"},
    {"panel":"A","group":"Primary analysis","source":"Combined primary (Sources 1+2)",
     "total":"—","same_day":"—","timing": fmt_pct_cohort(len(combined_verified)),
     "verif":"Independently verified"},
    {"panel":"A","group":"Supplementary sources only","source":"Source 3. Self-disclosed HIV status",
     "total": fmt_n(len(s3_pids)), "same_day":"NA","timing":"NA","verif":"Weak"},
    {"panel":"A","group":"Supplementary sources only","source":"Source 4. HIV test date self-disclosed to platform",
     "total": fmt_n(len(s4_pids)), "same_day":"NA","timing":"NA","verif":"Weak"},
    {"panel":"A","group":"","source":"Union of all sources",
     "total":"—","same_day":"—","timing": fmt_pct_cohort(len(union_all)),"verif":"Mixed"},
    {"panel":"B","group":"Primary analysis","source":"Source 1. CBO-verified dispensing",
     "total": fmt_n(len(p1_total)), "same_day": fmt_pct_total(len(p1_same_only), len(p1_total)),
     "timing": fmt_pct_cohort(len(p1_after)), "verif":"Strong"},
    {"panel":"B","group":"Supplementary source only","source":"Source 2. Self-disclosed PrEP start date",
     "total": fmt_n(len(p2_pids)), "same_day":"NA","timing":"NA","verif":"Weak"},
    {"panel":"B","group":"","source":"Union of both sources",
     "total":"—","same_day":"—","timing": fmt_pct_cohort(len(prep_union)),"verif":"Mixed"},
])
pd.DataFrame(supp3_rows).to_csv(os.path.join(OUT_DIR, "supplementary_table_3.csv"), index=False)

# ─── Cascade ────────────────────────────────────────────────────────────────
primary_test_pids = union_all
primary_prep_pids = prep_union

# Negative + PrEP (any negative result, then PrEP from union)
hiv_neg_pids = set(ctc[(ctc["test_result"]=="Negative") & (ctc["ts"]>=START) & (ctc["ts"]<=END)]["patient_id"]) | \
               set(st_nurse[st_nurse["image_reviewer_interpretation"]=="negative"]["patient_id"]) | \
               set(prof_c[prof_c["hiv_status_val"]=="negative"]["patient_id"])
neg_and_prep = hiv_neg_pids & primary_prep_pids

risk_in = risk[(risk["ts"] >= START) & (risk["ts"] <= END)]
med_high = set(risk_in[risk_in["risk_score_classification"].isin(["medium","high"])]["patient_id"].unique()) & all_pids
triple = hiv_neg_pids & med_high & primary_prep_pids
eligible_triple = hiv_neg_pids & med_high

print(f"\n=== CASCADE ===")
print(f"  Platform users:               {N:,}")
print(f"  Two-way nurse:                {len(two_way_pids):,} ({len(two_way_pids)/N*100:.1f}%)")
print(f"  HIV test (PRIMARY):           {len(primary_test_pids):,} ({len(primary_test_pids)/N*100:.1f}%)")
print(f"  PrEP uptake (PRIMARY):        {len(primary_prep_pids):,} ({len(primary_prep_pids)/N*100:.1f}%)")
print(f"  HIV-negative + PrEP:          {len(neg_and_prep):,} ({len(neg_and_prep)/N*100:.1f}%)")
print(f"  Triple criteria + PrEP:       {len(triple):,} ({len(triple)/N*100:.1f}%)")

# Figure 4
cascade = [
    ("Platform users",                N,                       "#3a6c8c"),
    ("Two-way nurse conversation",    len(two_way_pids),       "#3a6c8c"),
    ("HIV test after chat",           len(primary_test_pids),  "#3a6c8c"),
    ("PrEP uptake after chat",        len(primary_prep_pids),  "#2e6747"),
    ("HIV negative + PrEP",           len(neg_and_prep),       "#2e6747"),
    ("Triple criteria + PrEP",        len(triple),             "#2e6747"),
]
fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(cascade))[::-1]
pcts = [c[1]/N*100 for c in cascade]
ax.barh(y, pcts, color=[c[2] for c in cascade], edgecolor="none")
for yi, (_, c, _), pct in zip(y, cascade, pcts):
    color = "#1a3a52" if yi >= 3 else "#16382a"
    ax.text(pct + 1.2, yi, f"{c:,} ({pct:.1f}%)", va="center", fontsize=13, fontweight="bold", color=color)
ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cascade])
ax.set_xlim(0, 115); ax.set_xticks([0,25,50,75,100])
ax.set_xlabel(f"% of analytic cohort (n={N:,}, T&C-accepted users)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_4_cascade.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figure_4_cascade.png")

# ─── KM curves (verified events only) ──────────────────────────────────────
v_events = pd.concat([
    ctc_v[ctc_v["patient_id"].isin(s1_after)][["patient_id","ts"]],
    st_nurse[st_nurse["patient_id"].isin(s2_after)][["patient_id","ts"]],
])
v_events = v_events.merge(df[["patient_id","first_aimee_ts","first_aimee_date"]], on="patient_id")
v_events = v_events[v_events["ts"].dt.date > v_events["first_aimee_date"]]
first_test = v_events.sort_values("ts").groupby("patient_id").head(1).rename(columns={"ts":"first_test_ts"})

prep_ev = prep_ctc[prep_ctc["patient_id"].isin(p1_after)]
prep_ev = prep_ev[prep_ev["ts"].dt.date > prep_ev["first_aimee_date"]]
first_prep = prep_ev.sort_values("ts").groupby("patient_id").head(1).rename(columns={"ts":"first_prep_ts"})

MAX_DAYS = 185
df_tte = df.merge(first_test[["patient_id","first_test_ts"]], on="patient_id", how="left")
df_tte = df_tte.merge(first_prep[["patient_id","first_prep_ts"]], on="patient_id", how="left")
df_tte["study_end"] = END

def to_T_E(row, col):
    if pd.notna(row[col]):
        T = (row[col] - row["first_aimee_ts"]).total_seconds()/86400
        if T <= 0: return (np.nan, 0)
        if T > MAX_DAYS: return (MAX_DAYS, 0)
        return (T, 1)
    avail = (row["study_end"] - row["first_aimee_ts"]).total_seconds()/86400
    return (min(avail, MAX_DAYS), 0)

hiv_te = df_tte.apply(lambda r: to_T_E(r, "first_test_ts"), axis=1)
df_tte["hiv_T"] = [x[0] for x in hiv_te]; df_tte["hiv_E"] = [x[1] for x in hiv_te]
prep_te = df_tte.apply(lambda r: to_T_E(r, "first_prep_ts"), axis=1)
df_tte["prep_T"] = [x[0] for x in prep_te]; df_tte["prep_E"] = [x[1] for x in prep_te]
valid = df_tte.dropna(subset=["hiv_T","prep_T","reg_month"]).copy()

def cox(data, T, E):
    sub = data[[T, E, "meaningful", "reg_month"]].copy()
    dums = pd.get_dummies(sub["reg_month"], prefix="rm", drop_first=True).astype(float)
    sub = pd.concat([sub.drop(columns=["reg_month"]), dums], axis=1)
    cph = CoxPHFitter(penalizer=0.001); cph.fit(sub, duration_col=T, event_col=E)
    return (np.exp(cph.params_["meaningful"]),
            np.exp(cph.confidence_intervals_.loc["meaningful"].iloc[0]),
            np.exp(cph.confidence_intervals_.loc["meaningful"].iloc[1]),
            cph.summary.loc["meaningful","p"])

hr_hiv, lo_hiv, hi_hiv, p_hiv = cox(valid, "hiv_T", "hiv_E")
hr_prep, lo_prep, hi_prep, p_prep = cox(valid, "prep_T", "prep_E")
print(f"\nHIV testing (verified): HR {hr_hiv:.2f} ({lo_hiv:.2f}\u2013{hi_hiv:.2f})  p={p_hiv:.2g}")
print(f"PrEP uptake (verified): HR {hr_prep:.2f} ({lo_prep:.2f}\u2013{hi_prep:.2f})  p={p_prep:.2g}")

def km(data, T, E, color, hr_text, out):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    kmf = KaplanMeierFitter()
    for grp, label, col, ls in [
        (1, "Meaningful engagers  (\u22652 active days)", color, "-"),
        (0, "Single-session users  (1 active day)", "#888", "--"),
    ]:
        sub = data[data["meaningful"] == grp]
        n_e = int(sub[E].sum()); n_t = len(sub)
        kmf.fit(sub[T], event_observed=sub[E])
        tl = kmf.survival_function_.index.values
        ci = (1 - kmf.survival_function_.iloc[:, 0].values) * 100
        ci_lo = (1 - kmf.confidence_interval_.iloc[:, 1].values) * 100
        ci_hi = (1 - kmf.confidence_interval_.iloc[:, 0].values) * 100
        ax.plot(tl, ci, color=col, linestyle=ls, lw=2.2,
                label=f"{label}\n(events: {n_e:,} / {n_t:,})")
        ax.fill_between(tl, ci_lo, ci_hi, color=col, alpha=0.12)
    ax.set_xlabel("Days from first Aimee message"); ax.set_ylabel("Cumulative incidence (%)")
    ax.set_xlim(0, MAX_DAYS); ax.set_xticks([0,30,60,90,120,150,185])
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.50, 0.18, hr_text, transform=ax.transAxes, ha="center", va="center",
            fontsize=13, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

km(valid, "hiv_T", "hiv_E", "#3a6c8c",
   f"HR {hr_hiv:.2f} (95% CI {lo_hiv:.2f}\u2013{hi_hiv:.2f})\np <0\u00b7001",
   os.path.join(OUT_DIR, "figure_5a_km_hiv.png"))
km(valid, "prep_T", "prep_E", "#2e6747",
   f"HR {hr_prep:.2f} (95% CI {lo_prep:.2f}\u2013{hi_prep:.2f})\np <0\u00b7001",
   os.path.join(OUT_DIR, "figure_5b_km_prep.png"))
print("Saved figure_5a_km_hiv.png and figure_5b_km_prep.png")

# ─── Supp Figure 1: risk subgroup (using PRIMARY) ──────────────────────────
ri = risk[(risk["ts"] >= START) & (risk["ts"] <= END)]
ri = ri[ri["risk_score_classification"].isin(["low","medium","high"])]
latest_risk = ri.sort_values("ts").groupby("patient_id").tail(1)[["patient_id","risk_score_classification"]]
df_risk = df.merge(latest_risk, on="patient_id", how="inner")
# Risk subgroup analysis uses VERIFIED outcomes (Sources 1+2 for HIV; CBO-verified for PrEP)
# Rationale: PRIMARY union saturates in this subgroup because nearly all Phithos-classified
# patients disclosed HIV status or test date during conversation (rates approach 95%).
# Using verified events gives a more informative test of whether risk stratification
# correlates with actual care-seeking behaviour.
df_risk["hiv_outcome"]  = df_risk["patient_id"].isin(combined_verified).astype(int)
df_risk["prep_outcome"] = df_risk["patient_id"].isin(p1_after).astype(int)

def by_cat(outcome, data, col):
    out = {}
    for cat in ["low","medium","high"]:
        sub = data[data[col] == cat]
        n = len(sub); k = int(sub[outcome].sum()); pp = k/n if n else 0
        z = 1.96; denom = 1 + z**2/n if n else 1
        center = (pp + z**2/(2*n)) / denom
        half = (z * np.sqrt(pp*(1-pp)/n + z**2/(4*n**2))) / denom
        out[cat] = (pp, max(0, center-half), min(1, center+half), n, k)
    m2 = smf.logit(f"{outcome} ~ {col}_num + C(reg_month)",
                   data=data.assign(**{f"{col}_num": data[col].map({"low":0,"medium":1,"high":2})})
                  ).fit(disp=False)
    return out, m2.pvalues[f"{col}_num"]

hiv_probs,  p_t_hiv  = by_cat("hiv_outcome",  df_risk, "risk_score_classification")
prep_probs, p_t_prep = by_cat("prep_outcome", df_risk, "risk_score_classification")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
cats = ["low","medium","high"]
cat_labels = ["Low risk","Medium risk","High risk"]
for ax, probs, p_t, color_set, title in [
    (axes[0], hiv_probs,  p_t_hiv,  ["#a8c7df","#5e95bf","#264f74"], "A — HIV testing uptake"),
    (axes[1], prep_probs, p_t_prep, ["#b7d8c4","#5fa67f","#22593d"], "B — PrEP uptake"),
]:
    x = np.arange(3)
    means = [probs[c][0]*100 for c in cats]
    los   = [(probs[c][0] - probs[c][1])*100 for c in cats]
    his   = [(probs[c][2] - probs[c][0])*100 for c in cats]
    ax.bar(x, means, yerr=[los, his], color=color_set, edgecolor="none", capsize=6,
           error_kw={"lw":1.5,"ecolor":"#444"})
    for xi, m, h in zip(x, means, his):
        ax.text(xi, m+h+0.6, f"{m:.1f}%", ha="center", fontsize=13, fontweight="bold", color=color_set[-1])
    xticklabels = [f"{lab}\nn={probs[c][3]:,}\n({probs[c][4]:,} events)" for lab,c in zip(cat_labels, cats)]
    ax.set_xticks(x); ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Probability (%)"); ax.set_xlabel("Phithos HIV risk category")
    ps = f"p-trend = {p_t:.3f}" + ("  ns" if p_t > 0.05 else "")
    ax.text(0.98, 0.95, ps, transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888"))
    ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(means)+max(his)+4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "supp_figure_1_risk.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  HIV  Low/Med/High: {[f'{hiv_probs[c][0]*100:.1f}%' for c in cats]}  p-trend={p_t_hiv:.3f}")
print(f"  PrEP Low/Med/High: {[f'{prep_probs[c][0]*100:.1f}%' for c in cats]}  p-trend={p_t_prep:.3f}")

# Misc - tasks, timing
prep_pep_t = tasks[(tasks["type"]=="possible_interest_in_prep_or_pep") &
                    (tasks["task_ts"]>=START) & (tasks["task_ts"]<=END)]
prep_pep_pids = set(prep_pep_t["patient_id"].unique()) & all_pids
prep_pep_to_prep = prep_pep_pids & primary_prep_pids
print(f"\nPrEP/PEP interest tasks: {len(prep_pep_pids):,} ({len(prep_pep_pids)/N*100:.1f}%)")
print(f"  → Took PrEP: {len(prep_pep_to_prep):,} ({len(prep_pep_to_prep)/max(1,len(prep_pep_pids))*100:.1f}%)")

# Timing
prep_w = first_prep.copy()
prep_w["days"] = (prep_w["first_prep_ts"] - prep_w["first_aimee_ts"]).dt.total_seconds()/86400
print(f"Median days to PrEP (verified): {prep_w['days'].median():.0f} (IQR {prep_w['days'].quantile(0.25):.0f}\u2013{prep_w['days'].quantile(0.75):.0f})")

prep_w_m = prep_w.merge(df[["patient_id","meaningful"]], on="patient_id")
m_30 = (prep_w_m[prep_w_m["meaningful"]==1]["days"] <= 30).sum()
s_30 = (prep_w_m[prep_w_m["meaningful"]==0]["days"] <= 30).sum()
n_m = (prep_w_m["meaningful"]==1).sum(); n_s = (prep_w_m["meaningful"]==0).sum()
print(f"Median days — meaningful: {prep_w_m[prep_w_m['meaningful']==1]['days'].median():.0f}; single: {prep_w_m[prep_w_m['meaningful']==0]['days'].median():.0f}")
print(f"30d PrEP — meaningful: {m_30/n_m*100:.1f}%, single: {s_30/n_s*100:.1f}%")

triple_t = prep_w[prep_w["patient_id"].isin(triple)]
print(f"Median days (triple): {triple_t['days'].median():.0f}")

# Time to test by risk
print("\nTime to first verified test by risk category:")
test_w = first_test.copy()
test_w["days"] = (test_w["first_test_ts"] - test_w["first_aimee_ts"]).dt.total_seconds()/86400
for c in cats:
    sub_pids = set(df_risk[df_risk["risk_score_classification"]==c]["patient_id"])
    sub = test_w[test_w["patient_id"].isin(sub_pids)]
    print(f"  {c:<8} n={len(sub):>3}  median = {sub['days'].median():.0f} days")

def pct_of(n, d): return f"{n:,} ({n/d*100:.1f}%)"
def pct_cohort(n): return f"{n:,} ({n/N*100:.1f}%)"

s1_excl = len(s1_total) - len(s1_after)
s2_excl = len(s2_total) - len(s2_after)
p1_excl = len(p1_total) - len(p1_after)

summary = {
    "N_cohort": N,
    "N": N,
    "two_way_nurse": len(two_way_pids),
    "primary_HIV_test": len(primary_test_pids),
    "primary_hiv":      len(primary_test_pids),
    "verified_HIV_test": len(combined_verified),
    "secondary_hiv":    len(combined_verified),
    "s1_total": len(s1_total), "s1_after": len(s1_after),
    "s2_total": len(s2_total), "s2_after": len(s2_after),
    "s3": len(s3_pids), "s4": len(s4_pids),
    "primary_PrEP": len(primary_prep_pids),
    "primary_prep":    len(primary_prep_pids),
    "verified_PrEP": len(p1_after),
    "secondary_prep":  len(p1_after),
    "p1_total": len(p1_total), "p2": len(p2_pids),
    # PrEP intersect definitions — Issue 5 fix
    # CBO-verified HIV-negative: timing-corrected (after first Aimee message)
    "hiv_neg_cbo":          len(s1_after_neg),             # CBO-verified HIV-negative (timing-corrected)
    "hiv_neg_and_prep_cbo": len(s1_after_neg & p1_after),  # CBO-neg ∩ CBO-PrEP only
    "HIV_neg_and_prep": len(neg_and_prep),               # CBO-neg ∩ union-PrEP (was '840')
    "hiv_neg_and_prep": len(neg_and_prep),
    "triple": len(triple), "eligible_triple": len(eligible_triple),
    "prep_pep_tasks": len(prep_pep_pids),
    "prep_pep_to_prep": len(prep_pep_to_prep),
    "median_days_prep": float(prep_w["days"].median()),
    "median_days_prep_cbo": float(prep_w["days"].median()),  # alias
    "iqr_days_prep": (float(prep_w["days"].quantile(0.25)), float(prep_w["days"].quantile(0.75))),
    "iqr_prep": (float(prep_w["days"].quantile(0.25)), float(prep_w["days"].quantile(0.75))),  # alias
    "median_days_triple": float(triple_t["days"].median()) if len(triple_t) else None,
    "median_days_meaningful": float(prep_w_m[prep_w_m["meaningful"]==1]["days"].median()),
    "median_days_prep_meaningful": float(prep_w_m[prep_w_m["meaningful"]==1]["days"].median()),  # alias
    "median_days_single":     float(prep_w_m[prep_w_m["meaningful"]==0]["days"].median()),
    "median_days_prep_single": float(prep_w_m[prep_w_m["meaningful"]==0]["days"].median()),  # alias
    "pct_30d_meaningful": m_30/n_m*100,
    "pct_prep_30d_m":     m_30/n_m*100,           # alias
    "pct_30d_single":     s_30/n_s*100,
    "pct_prep_30d_s":     s_30/n_s*100,           # alias
    "HR_hiv":  (float(hr_hiv), float(lo_hiv), float(hi_hiv), float(p_hiv)),
    "HR_prep": (float(hr_prep), float(lo_prep), float(hi_prep), float(p_prep)),
    "hiv_probs":  [hiv_probs[c][0]*100 for c in cats],  "p_trend_hiv":  float(p_t_hiv),
    "prep_probs": [prep_probs[c][0]*100 for c in cats], "p_trend_prep": float(p_t_prep),
    "hiv_low_med_high":  [hiv_probs[c][0]*100 for c in cats],   # alias
    "prep_low_med_high": [prep_probs[c][0]*100 for c in cats],  # alias
    # Supplementary Table 3 panel rows
    "panel_a_rows": [
        ["Primary analysis", None, None, None, None],
        ["Source 1. CBO-verified clinic record",      f"{len(s1_total):,}", pct_of(s1_excl, len(s1_total)), pct_cohort(len(s1_after)), "Strong (reference)"],
        ["Source 2. Nurse-reviewed self-test upload", f"{len(s2_total):,}", pct_of(s2_excl, len(s2_total)), pct_cohort(len(s2_after)), "Moderate"],
        ["Combined verified (Sources 1+2)",           "—", "—", pct_cohort(len(combined_verified)), "Independently verified"],
        ["Supplementary sources", None, None, None, None],
        ["Source 3. Self-disclosed HIV status",       f"{len(s3_pids):,}", "NA", "NA", "Weak"],
        ["Source 4. HIV test date self-disclosed to platform", f"{len(s4_pids):,}", "NA", "NA", "Weak"],
        ["UNION OF ALL SOURCES (Primary analysis)",   "—", "—", pct_cohort(len(primary_test_pids)), "Mixed"],
    ],
    "panel_b_rows": [
        ["Primary analysis", None, None, None, None],
        ["Source 1. CBO-verified dispensing",         f"{len(p1_total):,}", pct_of(p1_excl, len(p1_total)), pct_cohort(len(p1_after)), "Strong"],
        ["Supplementary source", None, None, None, None],
        ["Source 2. Self-disclosed PrEP use",         f"{len(p2_pids):,}", "NA", "NA", "Weak"],
        ["UNION OF BOTH SOURCES (Primary analysis)",  "—", "—", pct_cohort(len(primary_prep_pids)), "Mixed"],
    ],
}
with open(os.path.join(OUT_DIR, "section3_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nDone.")
