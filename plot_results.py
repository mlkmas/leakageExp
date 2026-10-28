import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
s = pd.read_csv(f"{OUT}/results_summary.csv")

# fixed categorical order; low-capacity dashed, high-capacity solid; distinct markers (secondary encoding)
SERIES = [
    ("Logistic regression",    "#2a78d6", "o", "--"),
    ("AdaBoost (stumps)",      "#eb6834", "x", "--"),
    ("Shallow tree (depth 3)", "#1baf7a", "s", "--"),
    ("Neural network (MLP)",   "#eda100", "^", "-"),
    ("Random forest",          "#e87ba4", "D", "-"),
    ("Deep tree (unlimited)",  "#008300", "v", "-"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=True)
for ax, metric, title in [(axes[0], "accuracy_mean", "Test accuracy"), (axes[1], "auc_mean", "Test AUC")]:
    x_ref = [0, 100]
    ax.plot(x_ref, [0.5, 0.5], color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(100, 0.505, "chance (true performance)", color=MUTED, ha="right", va="bottom", fontsize=8.5)
    if metric == "accuracy_mean":
        ax.plot(x_ref, [0.5, 1.0], color="#b9b8b2", lw=1, ls="-", zorder=1)
        ax.text(62, 0.84, "perfect memorizer\n0.5 + 0.5 × contamination", color=MUTED, fontsize=8.5,
                rotation=0, ha="right")
    for name, col, mk, ls in SERIES:
        d = s[s.model == name].sort_values("contamination")
        ax.plot(d.contamination * 100, d[metric], color=col, marker=mk, ms=6, lw=2, ls=ls,
                label=name, zorder=3, markeredgecolor="white" if mk not in ("x",) else col, markeredgewidth=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Contamination: share of training rows copied from the test set (%)", color=MUTED)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel("Score on the full test set", color=MUTED)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Random data, so the true accuracy is 50%: every point above the dotted line is leakage",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0.11, 1, 0.95))
fig.savefig(f"{OUT}/accuracy_vs_contamination.png", facecolor="white")
print("saved")
