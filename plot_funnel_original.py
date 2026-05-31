"""Original 6-step engagement funnel — corrected numbers, same style."""
import matplotlib.pyplot as plt
import numpy as np

steps = [
    ("Sent a message",                  9958),
    ("Accepted T&C\n(analytic cohort)", 9310),
    ("Engaged with product",            6502),
    ("Meaningful engagement\n(≥2 active days)", 5452),
    ("Met engagement target",           4546),
    ("Received Phithos\nrisk score",    1260),
]

N      = steps[1][1]   # anchor = analytic cohort 9,310
labels = [s[0] for s in steps]
values = [s[1] for s in steps]
pcts   = [v / N * 100 for v in values]

n_steps  = len(steps)
fig_w, fig_h = 10, 13
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

slab_h   = 1.5
slab_gap = 0.20
total_h  = n_steps * (slab_h + slab_gap)
max_half = 3.8
min_half = 0.80

colors = ["#5a9dbf","#4a8db5","#3e80a8","#30709a","#22608c","#14507e"]

def slab_half(val):
    return min_half + (max_half - min_half) * (val / steps[0][1])

for i, (label, val, pct) in enumerate(zip(labels, values, pcts)):
    y_top = total_h - i * (slab_h + slab_gap)
    y_bot = y_top - slab_h
    hw    = slab_half(val)
    next_val = values[i + 1] if i < n_steps - 1 else val
    hw_bot   = slab_half(next_val)

    trap_x = [-hw, hw, hw_bot, -hw_bot]
    trap_y = [y_top, y_top, y_bot, y_bot]
    ax.fill(trap_x, trap_y, color=colors[i], zorder=2, linewidth=0)

    mid_y = (y_top + y_bot) / 2
    ax.text(0, mid_y + 0.22, f"{val:,}",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", zorder=3)

    # % relative to analytic cohort (Step 2) for all steps
    pct_label = f"{pct:.1f}%" if i > 0 else f"{val/steps[0][1]*100:.1f}% of sent"
    ax.text(0, mid_y - 0.30, pct_label,
            ha="center", va="center", fontsize=9.5,
            color="white", alpha=0.92, zorder=3)

    ax.annotate(
        label,
        xy=(hw, mid_y), xytext=(max_half + 0.55, mid_y),
        fontsize=10.5, va="center", color="#1a1a1a",
        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
        zorder=4,
    )

    # No dropout annotations

ax.set_xlim(-max_half - 0.2, max_half + 3.8)
ax.set_ylim(-0.5, total_h + 0.9)
ax.axis("off")

ax.text(0, total_h + 0.6,
        "Engagement funnel — Aimee, Clover Field Study",
        ha="center", va="bottom", fontsize=14, fontweight="bold", color="#111")
ax.text(0, total_h + 0.15,
        "17 March – 30 November 2025  |  Analytic cohort (T&C accepted): n=9,310",
        ha="center", va="bottom", fontsize=9, color="#555")

plt.tight_layout(pad=0.5)
plt.savefig("/home/claude/funnel/engagement_funnel_plot.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Done")
