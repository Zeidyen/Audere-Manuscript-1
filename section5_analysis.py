"""
SECTION 5 — Longitudinal risk-score change
Clover Field Study (Aimee), South Africa — 17 March to 30 November 2025 (SAST)

Two specific questions:
  Q1. Does risk-score REDUCTION predict PrEP uptake?
      Among patients with ≥2 risk assessments separated by ≥7 days, compare PrEP uptake
      by transition type (reduced / unchanged / increased risk category).
  Q2. Does NURSE INTERACTION drive risk reduction?
      Among medium/high baseline patients (only ones who can reduce), compare risk-reduction
      rates between two-way nurse contact vs other.

Outputs:
  - figure_7a_prep_by_risk_change.png      PrEP uptake by risk transition (stratified by baseline)
  - figure_7b_risk_reduction_by_nurse.png  Risk reduction by nurse interaction status
  - figure_7c_transition_sankey.png        Sankey-style flow diagram of first→last risk category
  - supplementary_table_5.csv              Transition counts and PrEP uptake rates
  - section5_summary.json
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    except: return None

# ─── Cohort ────────────────────────────────────────────────────────────────
print("Loading...")
msgs = pd.read_csv(p("ficus_messages_clover_fieldstudy_updated.csv"),
                   usecols=["sent_timestamp","role","is_test_data","api_owner"], low_memory=False)
msgs = msgs[msgs["is_test_data"]==False].rename(columns={"api_owner":"patient_id"})
msgs["sent_ts"] = to_sast(msgs["sent_timestamp"])
msgs = msgs[(msgs["sent_ts"]>=START)&(msgs["sent_ts"]<=END)]
user_msgs = msgs[msgs["role"]=="user"]
all_pids = set(user_msgs["patient_id"].dropna().unique())
t0 = user_msgs.groupby("patient_id")["sent_ts"].min().rename("first_aimee_ts")
df = pd.DataFrame({"patient_id": sorted(all_pids)}).merge(t0.reset_index(), on="patient_id")
df["first_aimee_date"] = df["first_aimee_ts"].dt.date
df["reg_month"] = df["first_aimee_ts"].dt.month_name()
N = len(df)

# ─── Risk assessments — first and last per patient ─────────────────────────
risk = pd.read_csv(p("ficus_risk_assessments_clover_fieldstudy_updated.csv"), low_memory=False)
risk = risk[risk["is_test_data"]==False]
risk["ts"] = to_sast(risk["assessment_timestamp"])
risk = risk[(risk["ts"]>=START)&(risk["ts"]<=END)]
risk = risk[risk["risk_score_classification"].isin(["low","medium","high"])]
risk = risk[risk["patient_id"].isin(all_pids)]
risk_sorted = risk.sort_values("ts")

CAT2NUM = {"low":0, "medium":1, "high":2}
first_r = risk_sorted.groupby("patient_id").head(1)[["patient_id","risk_score_classification","ts"]].rename(
    columns={"risk_score_classification":"first_cat","ts":"first_risk_ts"})
last_r  = risk_sorted.groupby("patient_id").tail(1)[["patient_id","risk_score_classification","ts"]].rename(
    columns={"risk_score_classification":"last_cat","ts":"last_risk_ts"})
n_assess = risk.groupby("patient_id").size().rename("n_assess")

trans = first_r.merge(last_r, on="patient_id").merge(n_assess.reset_index(), on="patient_id")
trans = trans[trans["n_assess"]>=2].copy()
trans["first_num"] = trans["first_cat"].map(CAT2NUM)
trans["last_num"]  = trans["last_cat"].map(CAT2NUM)
trans["span_days"] = (trans["last_risk_ts"] - trans["first_risk_ts"]).dt.total_seconds()/86400
trans["change"] = np.select(
    [trans["last_num"]<trans["first_num"], trans["last_num"]>trans["first_num"]],
    ["reduced","increased"], default="unchanged"
)
print(f"Patients with ≥2 risk assessments: {len(trans):,}")

# Filter to span ≥ 7 days (meaningful separation, not within-session refinement)
trans_m = trans[trans["span_days"]>=7].copy()
N_MEANINGFUL = len(trans_m)
print(f"With ≥7 day span (analysis cohort): {N_MEANINGFUL:,}")
print(trans_m["change"].value_counts())

# ─── Outcomes: primary PrEP uptake (CBO + self-disclosed) ──────────────────
pat = pd.read_csv(p("ficus_patients_clover_fieldstudy_updated.csv"), low_memory=False)
pat = pat[pat["is_test_data"]==False]
ctc = pd.read_csv(p("clover_connection_to_care_updated.csv"), low_memory=False)
ctc = ctc.merge(pat[["patient_id","referral_id"]], on="referral_id", how="inner")
ctc["ts"] = to_sast(ctc["service_date"])
prof = pd.read_csv(p("ficus_patient_profiles_clover_fieldstudy_updated.csv"), low_memory=False)
prof = prof[prof["is_test_data"]==False]
prof["takes_prep_val"] = prof.apply(lambda r: gv(r, "takes_prep", "value"), axis=1)

prep_cbo = ctc[ctc["medication_type"]=="PrEP"].copy()
prep_cbo = prep_cbo[(prep_cbo["ts"]>=START)&(prep_cbo["ts"]<=END)]
prep_cbo["date"] = prep_cbo["ts"].dt.date
prep_cbo = prep_cbo.merge(df[["patient_id","first_aimee_date"]], on="patient_id", how="inner")
p1 = set(prep_cbo[prep_cbo["date"]>prep_cbo["first_aimee_date"]]["patient_id"].unique())
p2 = set(prof[prof["takes_prep_val"]==True]["patient_id"].unique()) & all_pids
primary_prep = p1 | p2

trans_m["prep"] = trans_m["patient_id"].isin(primary_prep).astype(int)
trans_m = trans_m.merge(df[["patient_id","reg_month"]], on="patient_id", how="left")

# ─── HCW group ─────────────────────────────────────────────────────────────
hcw = pd.read_csv(p("ficus_hcw_conversations_clover_fieldstudy_updated.csv"), low_memory=False)
hcw = hcw[hcw["is_test_data"]==False]
hcw["first_ts"] = to_sast(hcw["first_recorded_message_timestamp"])
hcw_in = hcw[(hcw["first_ts"]>=START)&(hcw["first_ts"]<=END)]
two_way_pids = set(hcw_in[(hcw_in["num_user_messages"]>=1)&(hcw_in["num_hcw_messages"]>=1)]["patient_id"].unique()) & all_pids
trans_m["two_way"] = trans_m["patient_id"].isin(two_way_pids).astype(int)
trans_m["reduced_bin"] = (trans_m["change"]=="reduced").astype(int)

# ════════════════════════════════════════════════════════════════════════════
# Q1: PrEP uptake by risk transition
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Q1: PrEP uptake by risk transition ===")
q1_overall = []
for chg in ["reduced","unchanged","increased"]:
    sub = trans_m[trans_m["change"]==chg]
    q1_overall.append({"change":chg, "n":len(sub), "prep_n":int(sub["prep"].sum()),
                       "prep_pct":sub["prep"].mean()*100 if len(sub) else 0})
    print(f"  {chg:<11} n={len(sub):>4}  PrEP: {sub['prep'].sum()}/{len(sub)} = {sub['prep'].mean()*100:.1f}%")

# Stratified by baseline
print("\nStratified by baseline:")
q1_strat = []
for bc in ["low","medium","high"]:
    sub_b = trans_m[trans_m["first_cat"]==bc]
    for chg in ["reduced","unchanged","increased"]:
        ss = sub_b[sub_b["change"]==chg]
        if len(ss):
            q1_strat.append({"baseline":bc, "change":chg, "n":len(ss),
                              "prep_n":int(ss["prep"].sum()), "prep_pct":ss["prep"].mean()*100})
            print(f"  {bc:<7} → {chg:<11} n={len(ss):>3}  PrEP {ss['prep'].sum()}/{len(ss)} = {ss['prep'].mean()*100:.1f}%")

# Logistic regression
trans_m["change_cat"] = pd.Categorical(trans_m["change"], categories=["unchanged","reduced","increased"])
m_q1 = smf.logit("prep ~ C(change_cat) + C(first_cat) + C(reg_month)",
                 data=trans_m).fit(disp=False, method="bfgs", maxiter=200)
def or_ci(m, var):
    or_ = float(np.exp(m.params[var]))
    ci  = np.exp(m.conf_int().loc[var])
    return or_, float(ci.iloc[0]), float(ci.iloc[1]), float(m.pvalues[var])

q1_aor_red = or_ci(m_q1, "C(change_cat)[T.reduced]")
q1_aor_inc = or_ci(m_q1, "C(change_cat)[T.increased]")
print(f"\nQ1 AOR (vs unchanged, baseline-adjusted):")
print(f"  Reduced:   AOR {q1_aor_red[0]:.2f} ({q1_aor_red[1]:.2f}-{q1_aor_red[2]:.2f}), p={q1_aor_red[3]:.3g}")
print(f"  Increased: AOR {q1_aor_inc[0]:.2f} ({q1_aor_inc[1]:.2f}-{q1_aor_inc[2]:.2f}), p={q1_aor_inc[3]:.3g}")

# ════════════════════════════════════════════════════════════════════════════
# Q2: Nurse interaction → risk reduction
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Q2: Risk reduction by nurse interaction ===")
sub_can = trans_m[trans_m["first_cat"].isin(["medium","high"])].copy()
print(f"Medium/high baseline (n={len(sub_can):,}):")
q2_rows = []
for grp_label, grp_val in [("Two-way nurse contact", 1), ("No two-way contact", 0)]:
    s = sub_can[sub_can["two_way"]==grp_val]
    red = (s["change"]=="reduced").sum()
    q2_rows.append({"group":grp_label, "n":len(s), "reduced_n":int(red),
                    "reduced_pct":red/len(s)*100 if len(s) else 0})
    print(f"  {grp_label:<25} n={len(s):>4}  reduced={red} ({red/len(s)*100 if len(s) else 0:.1f}%)")

m_q2 = smf.logit("reduced_bin ~ two_way + C(first_cat) + C(reg_month)",
                 data=sub_can).fit(disp=False, method="bfgs", maxiter=200)
q2_aor = or_ci(m_q2, "two_way")
print(f"\nQ2 AOR (two-way nurse → risk reduction): {q2_aor[0]:.2f} ({q2_aor[1]:.2f}-{q2_aor[2]:.2f}), p={q2_aor[3]:.3g}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 7A — PrEP uptake by risk transition (stratified by baseline)
# ════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 7A...")
fig, ax = plt.subplots(figsize=(11, 6))
baseline_order = ["low","medium","high"]
change_order   = ["reduced","unchanged","increased"]
colors = {"reduced":"#2e6747", "unchanged":"#9bb6c9", "increased":"#a04545"}

x = np.arange(len(baseline_order))
width = 0.27
for i, chg in enumerate(change_order):
    vals, ns, errs = [], [], []
    for bc in baseline_order:
        ss = trans_m[(trans_m["first_cat"]==bc) & (trans_m["change"]==chg)]
        vals.append(ss["prep"].mean()*100 if len(ss) else np.nan)
        ns.append(len(ss))
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, color=colors[chg], edgecolor="white",
                  label=chg.capitalize(), linewidth=0.5)
    for xi, v, n in zip(x, vals, ns):
        if not np.isnan(v) and n > 0:
            ax.text(xi + offset, v + 0.8, f"{v:.0f}%\nn={n}", ha="center", fontsize=11,
                    fontweight="bold", color=colors[chg])

ax.set_xticks(x)
ax.set_xticklabels([f"{bc.capitalize()}-risk\nbaseline" for bc in baseline_order])
ax.set_ylabel("PrEP uptake (%)")
ax.set_ylim(0, 70)
ax.set_title("PrEP uptake by risk-score transition, stratified by baseline category")
ax.legend(title="Risk transition", loc="upper left", frameon=True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_7a_prep_by_risk_change.png"), dpi=150, bbox_inches="tight")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Figure 7B — Risk reduction by nurse interaction status
# ════════════════════════════════════════════════════════════════════════════
print("Building Figure 7B...")
fig, ax = plt.subplots(figsize=(10, 5.5))
groups = ["Two-way nurse contact", "No two-way contact"]
totals = [int((sub_can["two_way"]==v).sum()) for v in [1, 0]]
reds   = [int((sub_can[(sub_can["two_way"]==v) & (sub_can["reduced_bin"]==1)]).shape[0]) for v in [1, 0]]
pcts   = [r/t*100 if t else 0 for r, t in zip(reds, totals)]

x = np.arange(2)
bars = ax.bar(x, pcts, color=["#3a6c8c","#888"], width=0.5, edgecolor="none")
for xi, pp, r, t in zip(x, pcts, reds, totals):
    ax.text(xi, pp + 2, f"{pp:.1f}%", ha="center", fontsize=15, fontweight="bold",
            color="#1a3a52" if xi==0 else "#444")
    ax.text(xi, pp/2 if pp > 8 else pp + 8, f"{r}/{t}", ha="center", fontsize=12,
            color="white" if (xi==0 and pp>8) else "#222")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Risk-score reduction (%)")
ax.set_ylim(0, max(pcts)+12 if max(pcts) else 50)
ax.set_title(f"Risk-score reduction among medium- and high-risk patients (n={len(sub_can)})")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_7b_risk_reduction_by_nurse.png"), dpi=150, bbox_inches="tight")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Figure 7C — Transition flow diagram (Sankey-like, drawn with matplotlib)
# ════════════════════════════════════════════════════════════════════════════
print("Building Figure 7C...")
counts = pd.crosstab(trans_m["first_cat"], trans_m["last_cat"])
counts = counts.reindex(index=["low","medium","high"], columns=["low","medium","high"], fill_value=0)
print(counts)

fig, ax = plt.subplots(figsize=(10, 6))
cat_colors = {"low":"#7eb6a5", "medium":"#e5b769", "high":"#c46a6a"}
cat_labels = {"low":"Low risk", "medium":"Medium risk", "high":"High risk"}

# Position: left column = first_cat (x=0), right column = last_cat (x=1)
y_pos = {"low":0.85, "medium":0.50, "high":0.15}
box_height = 0.20; box_left_x = 0.08; box_right_x = 0.92; box_width = 0.10

# Compute totals to size boxes
left_totals = counts.sum(axis=1).to_dict()
right_totals = counts.sum(axis=0).to_dict()
total = counts.values.sum()

# Draw nodes (left)
for cat in ["low","medium","high"]:
    h = box_height
    rect = mpatches.FancyBboxPatch((box_left_x - box_width/2, y_pos[cat] - h/2), box_width, h,
                                    boxstyle="round,pad=0.005", linewidth=0, facecolor=cat_colors[cat])
    ax.add_patch(rect)
    ax.text(box_left_x - box_width/2 - 0.01, y_pos[cat], f"{cat_labels[cat]}\nn={left_totals[cat]}",
            ha="right", va="center", fontsize=13, fontweight="bold")

# Draw nodes (right)
for cat in ["low","medium","high"]:
    h = box_height
    rect = mpatches.FancyBboxPatch((box_right_x - box_width/2, y_pos[cat] - h/2), box_width, h,
                                    boxstyle="round,pad=0.005", linewidth=0, facecolor=cat_colors[cat])
    ax.add_patch(rect)
    ax.text(box_right_x + box_width/2 + 0.01, y_pos[cat], f"{cat_labels[cat]}\nn={right_totals[cat]}",
            ha="left", va="center", fontsize=13, fontweight="bold")

# Draw flows (ribbons)
from matplotlib.patches import PathPatch
from matplotlib.path import Path
max_flow = counts.values.max()
for first_cat in ["low","medium","high"]:
    for last_cat in ["low","medium","high"]:
        n = counts.loc[first_cat, last_cat]
        if n == 0: continue
        # Width proportional to count
        ribbon_w = (n / max_flow) * 0.12
        y1 = y_pos[first_cat]; y2 = y_pos[last_cat]
        x1 = box_left_x + box_width/2; x2 = box_right_x - box_width/2

        # Color: green if reducing, red if increasing, gray if same
        f_num = CAT2NUM[first_cat]; l_num = CAT2NUM[last_cat]
        if l_num < f_num: ribbon_color = "#2e6747"; alpha = 0.55
        elif l_num > f_num: ribbon_color = "#a04545"; alpha = 0.55
        else: ribbon_color = "#9bb6c9"; alpha = 0.40

        # Bezier curve ribbon
        ctrl_x = (x1 + x2) / 2
        verts = [
            (x1, y1 + ribbon_w/2),
            (ctrl_x, y1 + ribbon_w/2),
            (ctrl_x, y2 + ribbon_w/2),
            (x2, y2 + ribbon_w/2),
            (x2, y2 - ribbon_w/2),
            (ctrl_x, y2 - ribbon_w/2),
            (ctrl_x, y1 - ribbon_w/2),
            (x1, y1 - ribbon_w/2),
            (x1, y1 + ribbon_w/2),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor=ribbon_color, edgecolor="none", alpha=alpha)
        ax.add_patch(patch)

        # Label flow at midpoint if non-trivial
        if n >= 5:
            ax.text((x1+x2)/2, (y1+y2)/2, str(n), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#111",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))

ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
ax.set_axis_off()
ax.text(box_left_x, 0.98, "First assessment", ha="center", fontsize=13, fontweight="bold", color="#444")
ax.text(box_right_x, 0.98, "Last assessment", ha="center", fontsize=13, fontweight="bold", color="#444")

# Legend
legend_handles = [
    mpatches.Patch(color="#2e6747", alpha=0.55, label="Risk reduced"),
    mpatches.Patch(color="#9bb6c9", alpha=0.40, label="Unchanged"),
    mpatches.Patch(color="#a04545", alpha=0.55, label="Risk increased"),
]
ax.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.title(f"Risk-category transitions over time (n={len(trans_m)} patients, ≥7 days between assessments)",
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figure_7c_transition_sankey.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── Supplementary Table 5 ─────────────────────────────────────────────────
print("\nBuilding Supplementary Table 5...")
def fmtp(p): return "<0\u00b7001" if p < 0.001 else f"{p:.3f}".replace(".","\u00b7")

# Save key numbers
summary = {
    "N_total_repeat": len(trans),
    "N_meaningful":   N_MEANINGFUL,
    "median_span":    float(trans_m["span_days"].median()),
    "iqr_span_lo":    float(trans_m["span_days"].quantile(0.25)),
    "iqr_span_hi":    float(trans_m["span_days"].quantile(0.75)),
    "q1_overall":     q1_overall,
    "q1_strat":       q1_strat,
    "q1_aor_reduced": {"or":q1_aor_red[0], "lo":q1_aor_red[1], "hi":q1_aor_red[2], "p":q1_aor_red[3], "p_fmt":fmtp(q1_aor_red[3])},
    "q1_aor_increased": {"or":q1_aor_inc[0], "lo":q1_aor_inc[1], "hi":q1_aor_inc[2], "p":q1_aor_inc[3], "p_fmt":fmtp(q1_aor_inc[3])},
    "q2_rows":        q2_rows,
    "q2_aor":         {"or":q2_aor[0], "lo":q2_aor[1], "hi":q2_aor[2], "p":q2_aor[3], "p_fmt":fmtp(q2_aor[3])},
    "transition_matrix": counts.to_dict(),
    "N_baseline_low":    int((trans_m["first_cat"]=="low").sum()),
    "N_baseline_medium": int((trans_m["first_cat"]=="medium").sum()),
    "N_baseline_high":   int((trans_m["first_cat"]=="high").sum()),
}
with open(os.path.join(OUT_DIR, "section5_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

# CSV of transition counts
pd.DataFrame(q1_strat).to_csv(os.path.join(OUT_DIR, "supplementary_table_5.csv"), index=False)
print("Saved section5_summary.json and supplementary_table_5.csv")
