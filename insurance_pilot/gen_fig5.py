"""
Supplemental: Generate fig5 comparison plot for Insurance Pilot vs Student Performance.
Run after the main pilot experiment completes.
"""
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
COLORS = {"Pearson": "#4C72B0", "F-test": "#DD8452",
          "LassoCV": "#55A868", "RFE":    "#C44E52"}
METHODS = list(COLORS.keys())

# --- Load Insurance pilot CV data ---
ins_cv = pd.read_csv(os.path.join(OUTPUT_DIR, "cv_curve_data.csv"))
k_range = sorted(ins_cv["k"].unique())

# --- Load Student Performance CV data ---
student_path = os.path.join(BASE_DIR, "..", "results", "feature_count_comparison.csv")
df_st = pd.read_csv(student_path)
print("Student cols:", df_st.columns.tolist())
# Columns: Method, Number of Features, ..., CV RMSE, CV R2, ...
method_col = "Method"
k_col      = "Number of Features"
rmse_col   = "CV RMSE"

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Dataset Comparison: CV RMSE vs Number of Features\n"
             "(Student Performance  vs  Insurance [PILOT, n=5,000])",
             fontsize=12, fontweight="bold")

# ---- Left: Student Performance ----
ax = axes[0]
ax.set_title("Student Performance Dataset")
method_map = {
    "Pearson":  "Pearson Correlation",
    "F-test":   "F-test",
    "LassoCV":  "LassoCV",
    "RFE":      "RFECV",
}
for method, label in method_map.items():
    sub = df_st[df_st[method_col].str.contains(label.split()[0], case=False, na=False)]
    sub = sub.sort_values(k_col)
    if not sub.empty:
        ax.plot(sub[k_col], sub[rmse_col], "o-",
                color=COLORS[method], label=method, linewidth=2, markersize=4)
ax.set_xlabel("Number of Features (k)")
ax.set_ylabel("CV RMSE")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Right: Insurance (Pilot) ----
ax = axes[1]
ax.set_title("[PILOT TEST] Insurance Dataset (n=5,000, 3-Fold CV)")
for method in METHODS:
    sub = ins_cv[ins_cv["method"] == method].sort_values("k")
    means = sub["rmse_mean"].values
    stds  = sub["rmse_std"].values
    c = COLORS[method]
    ax.plot(k_range, means, "o-", color=c, label=method, linewidth=2, markersize=4)
    ax.fill_between(k_range, means - stds, means + stds, alpha=0.15, color=c)
ax.set_xlabel("Number of Features (k)")
ax.set_ylabel("CV RMSE")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(OUTPUT_DIR, "fig5_comparison_student_vs_insurance.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")
print("Done.")
