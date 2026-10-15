"""
Parkinsons Telemonitoring Dataset - Feature Selection Experiment
================================================================
Dataset : https://www.kaggle.com/rishidamarla/parkinsons-telemonitoring-data
Target  : total_UPDRS  (continuous regression)
Excluded: motor_UPDRS (target leakage), subject# (patient ID)
Method  : LinearRegression + 4 Feature Selection methods
Split   : Group-aware 80/20 (by subject#) + GroupKFold(n_splits=5)

This script is FULLY STANDALONE and does NOT modify any existing project files.
All outputs go to experiment_parkinsons/output/.
"""

import sys, io
# Force UTF-8 output on Windows (avoids cp950 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import (
    SelectKBest, f_regression, r_regression, RFE
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "telemonitoring_parkinsons_updrs.data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "Pearson":  "#4C72B0",
    "F-test":   "#DD8452",
    "LassoCV":  "#55A868",
    "RFE":      "#C44E52",
}
METHOD_NAMES = list(COLORS.keys())

# ==============================================================================
# 1. Load & Inspect
# ==============================================================================
def load_and_inspect(path: str) -> pd.DataFrame:
    print("=" * 65)
    print("SECTION 1 - DATA INSPECTION")
    print("=" * 65)
    df = pd.read_csv(path)
    print(f"  Total samples   : {len(df)}")
    print(f"  Total columns   : {df.shape[1]}")
    print(f"  Unique subjects : {df['subject#'].nunique()}")
    print()

    print("  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<25} {dtype}")
    print()

    print("  Missing values:")
    mv = df.isnull().sum()
    if mv.sum() == 0:
        print("    -> None (0 missing values across all columns)")
    else:
        print(mv[mv > 0])
    print()

    categorical = df.select_dtypes(include="object").columns.tolist()
    print(f"  Categorical columns: {categorical if categorical else 'None'}")
    print()

    print("  Target - total_UPDRS stats:")
    print(df["total_UPDRS"].describe().to_string())
    print()

    print("  Excluded from features:")
    print("    subject#      -> patient ID (group variable, not a predictor)")
    print("    motor_UPDRS   -> target leakage (sub-score of total_UPDRS)")
    print("    test_time     -> recording index, not a biological feature")
    print("    age, sex      -> demographic metadata (kept; see note below)")
    print()
    return df


# ==============================================================================
# 2. Pre-process & Group Split
# ==============================================================================
def prepare_data(df: pd.DataFrame):
    """
    Feature columns: 16 voice measures + age + sex  (18 total)
    Drop: subject#, motor_UPDRS, test_time (recording time index)
    Target: total_UPDRS
    Groups: subject#
    """
    print("=" * 65)
    print("SECTION 2 - PRE-PROCESSING & GROUP SPLIT")
    print("=" * 65)

    DROP_COLS = ["subject#", "motor_UPDRS", "test_time", "total_UPDRS"]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X = df[feature_cols].values
    y = df["total_UPDRS"].values
    groups = df["subject#"].values

    print(f"  Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {col}")
    print()

    # Group-aware 80/20 split (whole patients into train/test)
    unique_subjects = np.unique(groups)
    n_subjects      = len(unique_subjects)
    n_train_subj    = int(np.floor(n_subjects * 0.8))

    rng = np.random.RandomState(42)
    train_subjects  = rng.choice(unique_subjects, size=n_train_subj, replace=False)
    test_subjects   = np.setdiff1d(unique_subjects, train_subjects)

    train_mask = np.isin(groups, train_subjects)
    test_mask  = np.isin(groups, test_subjects)

    X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test,  y_test            = X[test_mask],  y[test_mask]

    print(f"  Group-aware 80/20 split (by subject#):")
    print(f"    Train subjects : {len(train_subjects)} / {n_subjects}  "
          f"({len(X_train)} rows)")
    print(f"    Test  subjects : {len(test_subjects)} / {n_subjects}  "
          f"({len(X_test)} rows)")
    print()

    return X_train, X_test, y_train, y_test, g_train, feature_cols


# ==============================================================================
# 3. Feature Selection helpers (ALL called INSIDE each fold)
# ==============================================================================
def select_pearson(X_fold, y_fold, k: int):
    """Absolute Pearson correlation - computed on fold training data."""
    corrs = np.abs(r_regression(X_fold, y_fold))
    idx   = np.argsort(corrs)[::-1][:k]
    return np.sort(idx)


def select_ftest(X_fold, y_fold, k: int):
    """F-test (ANOVA F-value) for regression."""
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X_fold, y_fold)
    return np.where(selector.get_support())[0]


def select_lasso(X_fold, y_fold, k: int, groups_fold=None):
    """
    LassoCV with Group-aware internal CV.
    Groups_fold are used to build a group-stratified CV splitter so that
    the same patient never appears in both inner train and inner validation.
    k is the desired number of features; we pick the top-|coef| features
    from the Lasso solution (not by thresholding α).
    """
    n_features = X_fold.shape[1]
    k_eff      = min(k, n_features)

    # Build inner GroupKFold CV for LassoCV
    if groups_fold is not None:
        unique_g   = np.unique(groups_fold)
        n_inner_cv = min(5, len(unique_g))
        inner_cv   = list(
            GroupKFold(n_splits=n_inner_cv).split(X_fold, y_fold, groups_fold)
        )
    else:
        inner_cv = 5

    lasso = LassoCV(cv=inner_cv, random_state=42, max_iter=10000)
    lasso.fit(X_fold, y_fold)

    coef_abs = np.abs(lasso.coef_)
    idx      = np.argsort(coef_abs)[::-1][:k_eff]
    return np.sort(idx)


def select_rfe(X_fold, y_fold, k: int):
    """RFE with LinearRegression estimator."""
    n_features = X_fold.shape[1]
    k_eff      = min(k, n_features)
    estimator  = LinearRegression()
    rfe        = RFE(estimator, n_features_to_select=k_eff)
    rfe.fit(X_fold, y_fold)
    return np.where(rfe.support_)[0]


SELECTORS = {
    "Pearson":  select_pearson,
    "F-test":   select_ftest,
    "LassoCV":  select_lasso,
    "RFE":      select_rfe,
}


# ==============================================================================
# 4. 5-Fold Group CV loop
# ==============================================================================
def run_cv_experiment(X_train, y_train, g_train, feature_cols):
    print("=" * 65)
    print("SECTION 3 - 5-FOLD GROUP CV EXPERIMENT")
    print("=" * 65)

    n_features = X_train.shape[1]
    k_range    = list(range(1, n_features + 1))   # k = 1 … n_features
    gkf        = GroupKFold(n_splits=5)

    results = {}   # method -> {k: {"rmse": [...], "r2": [...]}}
    timing  = {}   # method -> elapsed seconds

    for method in METHOD_NAMES:
        print(f"\n  Running: {method}  (k=1..{n_features})")
        t0           = time.time()
        method_res   = {}

        for k in k_range:
            fold_rmse = []
            fold_r2   = []

            for fold_idx, (tr_idx, va_idx) in enumerate(
                    gkf.split(X_train, y_train, g_train)):

                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]
                g_tr       = g_train[tr_idx]

                # ── Step 1: StandardScaler fit on fold train ──
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)

                # ── Step 2: Feature selection fit on fold train ──
                if method == "LassoCV":
                    feat_idx = select_lasso(X_tr_sc, y_tr, k, groups_fold=g_tr)
                else:
                    feat_idx = SELECTORS[method](X_tr_sc, y_tr, k)

                X_tr_sel = X_tr_sc[:, feat_idx]
                X_va_sel = X_va_sc[:, feat_idx]

                # ── Step 3: Fit LinearRegression ──
                lr = LinearRegression()
                lr.fit(X_tr_sel, y_tr)
                y_pred = lr.predict(X_va_sel)

                rmse = np.sqrt(mean_squared_error(y_va, y_pred))
                r2   = r2_score(y_va, y_pred)
                fold_rmse.append(rmse)
                fold_r2.append(r2)

            method_res[k] = {
                "rmse_mean": np.mean(fold_rmse),
                "rmse_std":  np.std(fold_rmse),
                "r2_mean":   np.mean(fold_r2),
                "r2_std":    np.std(fold_r2),
            }

        elapsed       = time.time() - t0
        timing[method] = elapsed
        results[method] = method_res
        print(f"    Done in {elapsed:.1f}s")

    return results, k_range, timing


# ==============================================================================
# 5. Best k & test-set evaluation
# ==============================================================================
def evaluate_best_k(X_train, X_test, y_train, y_test, g_train,
                    results, k_range, feature_cols):
    print()
    print("=" * 65)
    print("SECTION 4 - BEST k & TEST-SET EVALUATION")
    print("=" * 65)
    print("  (Best k chosen by CV RMSE; test set NOT used for selection)\n")

    summary = {}

    for method in METHOD_NAMES:
        method_res = results[method]
        # best k by lowest CV RMSE mean
        best_k = min(k_range, key=lambda k: method_res[k]["rmse_mean"])
        cv_rmse = method_res[best_k]["rmse_mean"]
        cv_rmse_std = method_res[best_k]["rmse_std"]
        cv_r2   = method_res[best_k]["r2_mean"]
        cv_r2_std = method_res[best_k]["r2_std"]

        # Retrain on full train set with best_k, evaluate on test
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        if method == "LassoCV":
            feat_idx = select_lasso(X_tr_sc, y_train, best_k, groups_fold=g_train)
        else:
            feat_idx = SELECTORS[method](X_tr_sc, y_train, best_k)

        X_tr_sel = X_tr_sc[:, feat_idx]
        X_te_sel = X_te_sc[:, feat_idx]

        lr = LinearRegression()
        lr.fit(X_tr_sel, y_train)
        y_pred_test = lr.predict(X_te_sel)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2   = r2_score(y_test, y_pred_test)

        selected_features = [feature_cols[i] for i in feat_idx]

        summary[method] = {
            "best_k":       best_k,
            "cv_rmse":      cv_rmse,
            "cv_rmse_std":  cv_rmse_std,
            "cv_r2":        cv_r2,
            "cv_r2_std":    cv_r2_std,
            "test_rmse":    test_rmse,
            "test_r2":      test_r2,
            "features":     selected_features,
            "feat_idx":     feat_idx,
        }

        print(f"  [{method}]")
        print(f"    Best k         : {best_k}")
        print(f"    CV  RMSE       : {cv_rmse:.4f} +/- {cv_rmse_std:.4f}")
        print(f"    CV  R^2         : {cv_r2:.4f} +/- {cv_r2_std:.4f}")
        print(f"    Test RMSE      : {test_rmse:.4f}")
        print(f"    Test R^2        : {test_r2:.4f}")
        print(f"    Selected feats : {selected_features}")
        print()

    return summary


# ==============================================================================
# 6. Plots
# ==============================================================================
def plot_cv_curves(results, k_range, output_dir):
    """Figure 1 - RMSE vs k  (with +/-1 std band)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Parkinsons Telemonitoring - Feature Selection Comparison (5-Fold Group CV)",
        fontsize=13, fontweight="bold", y=1.01)

    for ax, metric, label in zip(
            axes,
            ["rmse", "r2"],
            ["CV RMSE ((dn) better)", "CV R^2 ((up) better)"]):

        for method in METHOD_NAMES:
            means = [results[method][k][f"{metric}_mean"] for k in k_range]
            stds  = [results[method][k][f"{metric}_std"]  for k in k_range]
            means = np.array(means)
            stds  = np.array(stds)
            c     = COLORS[method]
            ax.plot(k_range, means, "o-", color=c, label=method,
                    linewidth=2, markersize=4)
            ax.fill_between(k_range, means - stds, means + stds,
                            alpha=0.15, color=c)

        ax.set_xlabel("Number of Features (k)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_range)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_cv_curves_rmse_r2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_method_comparison_bar(summary, output_dir):
    """Figure 2 - Bar chart: CV RMSE & R^2 at best k per method."""
    methods = METHOD_NAMES
    cv_rmse  = [summary[m]["cv_rmse"]  for m in methods]
    test_rmse= [summary[m]["test_rmse"] for m in methods]
    cv_r2    = [summary[m]["cv_r2"]    for m in methods]
    test_r2  = [summary[m]["test_r2"]  for m in methods]
    best_ks  = [summary[m]["best_k"]   for m in methods]
    colors   = [COLORS[m] for m in methods]

    x  = np.arange(len(methods))
    w  = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Best-k Performance Summary per Method", fontsize=13, fontweight="bold")

    for ax, cv_vals, te_vals, ylabel, title in zip(
            axes,
            [cv_rmse, cv_r2],
            [test_rmse, test_r2],
            ["RMSE", "R^2"],
            ["RMSE at Best k", "R^2 at Best k"]):

        bars_cv = ax.bar(x - w/2, cv_vals, w, label="CV",   color=colors, alpha=0.7)
        bars_te = ax.bar(x + w/2, te_vals, w, label="Test", color=colors, alpha=1.0,
                         edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n(k={best_ks[i]})" for i, m in enumerate(methods)])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        for bar in bars_cv:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001*h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_te:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001*h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_best_k_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_feature_heatmap(summary, feature_cols, output_dir):
    """Figure 3 - Feature selection heatmap."""
    n_feat = len(feature_cols)
    mat    = np.zeros((len(METHOD_NAMES), n_feat))
    for i, method in enumerate(METHOD_NAMES):
        for j in summary[method]["feat_idx"]:
            mat[i, j] = 1

    fig, ax = plt.subplots(figsize=(max(10, n_feat * 0.7), 4))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHOD_NAMES)))
    ax.set_yticklabels(METHOD_NAMES)
    ax.set_title("Features Selected at Best k (blue = selected)", fontsize=11)
    for i in range(len(METHOD_NAMES)):
        for j in range(n_feat):
            ax.text(j, i, "v" if mat[i, j] else "",
                    ha="center", va="center", fontsize=10, color="steelblue")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_feature_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_timing(timing, output_dir):
    """Figure 4 - Execution time per method."""
    methods = METHOD_NAMES
    times   = [timing[m] for m in methods]
    colors  = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, times, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Feature Selection Method - Total Execution Time")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_timing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ==============================================================================
# 7. Save CSV summaries
# ==============================================================================
def save_csv_summaries(results, summary, timing, k_range, output_dir):
    # 7-a. Full CV curve data
    rows = []
    for method in METHOD_NAMES:
        for k in k_range:
            r = results[method][k]
            rows.append({
                "method":    method,
                "k":         k,
                "rmse_mean": round(r["rmse_mean"], 5),
                "rmse_std":  round(r["rmse_std"], 5),
                "r2_mean":   round(r["r2_mean"], 5),
                "r2_std":    round(r["r2_std"], 5),
            })
    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(os.path.join(output_dir, "cv_curve_data.csv"), index=False)

    # 7-b. Summary table
    rows2 = []
    for method in METHOD_NAMES:
        s = summary[method]
        rows2.append({
            "method":           method,
            "best_k":           s["best_k"],
            "cv_rmse":          round(s["cv_rmse"], 4),
            "cv_rmse_std":      round(s["cv_rmse_std"], 4),
            "cv_r2":            round(s["cv_r2"], 4),
            "cv_r2_std":        round(s["cv_r2_std"], 4),
            "test_rmse":        round(s["test_rmse"], 4),
            "test_r2":          round(s["test_r2"], 4),
            "elapsed_sec":      round(timing[method], 1),
            "selected_features":"; ".join(s["features"]),
        })
    df_sum = pd.DataFrame(rows2)
    df_sum.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    print(f"  Saved: cv_curve_data.csv")
    print(f"  Saved: summary_table.csv")
    return df_sum


# ==============================================================================
# 8. Comparison with Student Performance results
# ==============================================================================
STUDENT_RESULTS = {
    # Approximate values from main.py experiments (Student Performance HW2)
    # best_k by CV RMSE, target = productivity_score
    "Pearson":  {"best_k": None, "note": "see main project results"},
    "F-test":   {"best_k": None, "note": "see main project results"},
    "LassoCV":  {"best_k": None, "note": "see main project results"},
    "RFE":      {"best_k": None, "note": "see main project results"},
}

def load_student_results():
    """Try to load student performance results CSV from main project."""
    candidate = os.path.join(BASE_DIR, "..", "results", "feature_count_comparison.csv")
    if os.path.exists(candidate):
        return pd.read_csv(candidate)
    candidate2 = os.path.join(BASE_DIR, "..", "feature_count_comparison.csv")
    if os.path.exists(candidate2):
        return pd.read_csv(candidate2)
    return None


def plot_comparison(results_park, summary_park, k_range_park, output_dir):
    """Figure 5 - Side-by-side Student vs Parkinsons CV RMSE curves (normalised)."""
    df_student = load_student_results()
    if df_student is None:
        print("  [SKIP] Student results CSV not found; skipping comparison plot.")
        return

    # Normalise RMSE by mean of k=all-features RMSE across methods
    # so both datasets are on comparable scale
    park_all = {m: results_park[m][max(k_range_park)]["rmse_mean"] for m in METHOD_NAMES}
    norm_park = np.mean(list(park_all.values()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             sharey=False)
    fig.suptitle("Dataset Comparison - CV RMSE vs Number of Features",
                 fontsize=13, fontweight="bold")

    # Left: Student Performance
    ax = axes[0]
    ax.set_title("Student Performance Dataset")
    # Try to plot student data
    try:
        for method in METHOD_NAMES:
            sub = df_student[df_student["method"].str.lower() == method.lower()]
            if sub.empty:
                # try alternate column names
                for col in df_student.columns:
                    if "method" in col.lower():
                        sub = df_student[df_student[col].str.lower() == method.lower()]
                        break
            if not sub.empty:
                k_col   = [c for c in sub.columns if "k" == c.lower() or "n_feat" in c.lower()][0]
                rmse_col= [c for c in sub.columns if "rmse" in c.lower()][0]
                ax.plot(sub[k_col], sub[rmse_col], "o-",
                        color=COLORS[method], label=method, linewidth=2, markersize=4)
    except Exception as e:
        ax.text(0.5, 0.5, f"Student data format mismatch:\n{e}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("CV RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Parkinsons
    ax = axes[1]
    ax.set_title("Parkinsons Telemonitoring Dataset")
    for method in METHOD_NAMES:
        means = [results_park[method][k]["rmse_mean"] for k in k_range_park]
        stds  = [results_park[method][k]["rmse_std"]  for k in k_range_park]
        means = np.array(means)
        stds  = np.array(stds)
        c     = COLORS[method]
        ax.plot(k_range_park, means, "o-", color=c, label=method,
                linewidth=2, markersize=4)
        ax.fill_between(k_range_park, means - stds, means + stds,
                        alpha=0.15, color=c)
    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("CV RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_dataset_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ==============================================================================
# 9. Print final narrative comparison
# ==============================================================================
def print_final_comparison(summary, timing, feature_cols, k_range):
    print()
    print("=" * 65)
    print("SECTION 5 - SUMMARY & COMPARISON WITH STUDENT PERFORMANCE")
    print("=" * 65)

    print("\n  [Parkinsons Telemonitoring - Summary Table]")
    print(f"  {'Method':<10} {'Best k':>6} {'CV RMSE':>10} {'+/-':>6} "
          f"{'CV R^2':>8} {'+/-':>6} {'Test RMSE':>10} {'Test R^2':>8} {'Time(s)':>8}")
    print("  " + "-" * 70)
    for m in METHOD_NAMES:
        s = summary[m]
        print(f"  {m:<10} {s['best_k']:>6} {s['cv_rmse']:>10.4f} "
              f"{s['cv_rmse_std']:>6.4f} {s['cv_r2']:>8.4f} "
              f"{s['cv_r2_std']:>6.4f} {s['test_rmse']:>10.4f} "
              f"{s['test_r2']:>8.4f} {timing[m]:>8.1f}")

    print()
    print("  [Dataset Characteristics]")
    print(f"  Parkinsons :")
    print(f"    Samples          : 5875  (42 patients, repeated measures)")
    print(f"    Features used    : {len(feature_cols)}  (voice acoustics + age + sex)")
    print(f"    Target           : total_UPDRS  (7.0 - 55.0, Parkinson severity)")
    print(f"    Split strategy   : Group-aware 80/20 + GroupKFold(5)")
    print(f"    Missing values   : 0")
    print(f"    Categorical cols : 0  (all numeric)")
    print()
    print(f"  Student Performance :")
    print(f"    Samples          : ~16000")
    print(f"    Features used    : 16 (study habits, demographics)")
    print(f"    Target           : productivity_score (continuous)")
    print(f"    Split strategy   : Random 80/20 + KFold(5)")
    print(f"    Missing values   : none reported")
    print(f"    Categorical cols : gender  (encoded)")
    print()
    print("  [Analysis Notes]")
    print("  1. Both RMSE scales differ (total_UPDRS ~29 mean vs productivity_score")
    print("     scale); do NOT compare RMSE absolute values across datasets.")
    print("  2. Parkinsons has highly correlated Jitter/Shimmer sub-features,")
    print("     which may cause Pearson & F-test to pick redundant features.")
    print("  3. GroupKFold prevents patient data leakage but may increase")
    print("     CV variance due to fewer effective folds with 42 subjects.")
    print("  4. LassoCV inner CV is also group-aware - fair comparison.")
    print()

    # Quick recommendation heuristic
    rmse_range_park = max(
        summary[m]["cv_rmse"] for m in METHOD_NAMES
    ) - min(
        summary[m]["cv_rmse"] for m in METHOD_NAMES
    )
    print(f"  CV RMSE spread across methods (Parkinsons): {rmse_range_park:.4f}")
    print()
    print("  >>> RECOMMENDATION (preliminary) <<<")
    if rmse_range_park > 0.5:
        print("  Parkinsons shows meaningful between-method RMSE difference.")
        print("  CONSIDER switching if group-aware CV aligns with HW2 requirements.")
    else:
        print("  Between-method differences are small on this dataset.")
        print("  Student Performance may still be the safer HW2 choice unless")
        print("  the group-aware CV structure itself is a desired demonstration.")
    print()
    print("  [Awaiting your confirmation before any changes to main project]")
    print()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print()
    print("+==============================================================+")
    print("|   Parkinsons Telemonitoring - Feature Selection Experiment  |")
    print("+==============================================================+")
    print()

    # 1. Load & inspect
    df = load_and_inspect(DATA_PATH)

    # 2. Pre-process & group split
    X_train, X_test, y_train, y_test, g_train, feature_cols = prepare_data(df)

    # 3. 5-Fold Group CV
    results, k_range, timing = run_cv_experiment(X_train, y_train, g_train, feature_cols)

    # 4. Best k & test evaluation
    summary = evaluate_best_k(X_train, X_test, y_train, y_test, g_train,
                               results, k_range, feature_cols)

    # 5. Plots
    print("=" * 65)
    print("SECTION 4b - SAVING PLOTS & CSVs")
    print("=" * 65)
    plot_cv_curves(results, k_range, OUTPUT_DIR)
    plot_method_comparison_bar(summary, OUTPUT_DIR)
    plot_feature_heatmap(summary, feature_cols, OUTPUT_DIR)
    plot_timing(timing, OUTPUT_DIR)
    plot_comparison(results, summary, k_range, OUTPUT_DIR)

    # 6. CSV summaries
    df_sum = save_csv_summaries(results, summary, timing, k_range, OUTPUT_DIR)

    # 7. Final narrative
    print_final_comparison(summary, timing, feature_cols, k_range)

    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print()
    print("  ✅  Experiment complete.  Awaiting your confirmation.")
    print()


if __name__ == "__main__":
    main()
