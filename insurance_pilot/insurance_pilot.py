"""
Insurance Dataset - Feature Selection Pilot Test
=================================================
Source  : Kaggle playground-series-s4e12 (local CSV only, no API)
Target  : Premium Amount  (continuous regression)
Sample  : 5,000 rows (random_state=42, chunked read from 1.2M rows)
Methods : LinearRegression + Pearson / F-test / LassoCV / RFE
CV      : 3-Fold CV, k=1..n_encoded_features (capped at 20)
Output  : insurance_pilot/output/

NOTE: This is a PILOT TEST only.
      The existing HW2 project and Student Performance dataset are NOT modified.
"""

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, r_regression, RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "playground-series-s4e12", "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {"Pearson": "#4C72B0", "F-test": "#DD8452",
          "LassoCV": "#55A868", "RFE":    "#C44E52"}
METHODS = list(COLORS.keys())
SAMPLE_N = 5_000
RANDOM_STATE = 42
N_FOLDS  = 3
K_MAX    = 20   # upper cap for k


# ==============================================================================
# 1. Chunked Sampling  (avoid loading 194 MB fully)
# ==============================================================================
def chunked_sample(path: str, n: int, seed: int) -> pd.DataFrame:
    """
    Reservoir-style sampling via chunked read.
    Each chunk contributes proportionally; final result is shuffled & sliced.
    Reads only header + samples, never loads full 1.2M rows into RAM.
    """
    print("  Reading CSV in chunks (chunk_size=100_000) ...")
    rng = np.random.RandomState(seed)
    chunks = []
    total_read = 0
    for chunk in pd.read_csv(path, chunksize=100_000):
        frac = n / 1_200_000          # target fraction of total
        k_chunk = max(1, int(len(chunk) * frac * 1.5))  # slight oversample
        sampled = chunk.sample(n=min(k_chunk, len(chunk)), random_state=rng.randint(0, 99999))
        chunks.append(sampled)
        total_read += len(chunk)
    df_all = pd.concat(chunks, ignore_index=True)
    df_all = df_all.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  Total rows read (chunks): {total_read:,}  ->  sampled {len(df_all):,} rows")
    return df_all


# ==============================================================================
# 2. Inspect & Pre-process
# ==============================================================================
def inspect_and_preprocess(df: pd.DataFrame):
    print()
    print("=" * 65)
    print("SECTION 1 - DATA INSPECTION (5,000-row sample)")
    print("=" * 65)

    TARGET = "Premium Amount"
    DROP   = ["id", TARGET]

    print(f"  Sample rows    : {len(df)}")
    print(f"  Target         : {TARGET}")
    print(f"  Target stats   :")
    print(df[TARGET].describe().to_string())
    print()

    # Missing values
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    print(f"  Missing values in sample:")
    if mv.empty:
        print("    None")
    else:
        print(mv.to_string())
    print()

    # Categorical & numeric
    raw_feats = [c for c in df.columns if c not in DROP]
    cat_cols  = df[raw_feats].select_dtypes(include="object").columns.tolist()
    num_cols  = df[raw_feats].select_dtypes(exclude="object").columns.tolist()

    print(f"  Numeric features  ({len(num_cols)}): {num_cols}")
    print(f"  Categorical feats ({len(cat_cols)}): {cat_cols}")
    print()

    # ----- Pre-processing -----
    # Drop 'Policy Start Date' - raw date string, needs special parsing;
    # as a pilot test we skip it to keep things clean.
    if "Policy Start Date" in cat_cols:
        cat_cols.remove("Policy Start Date")
        print("  [NOTE] 'Policy Start Date' dropped (raw date string, not encoded in pilot)")

    # Fill missing values:
    #   numeric  -> median  (computed per fold; here we do global for inspection only)
    #   category -> mode
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categoricals
    dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    encoded_num = df[num_cols].copy()
    X_df = pd.concat([encoded_num, dummies], axis=1)
    y    = df[TARGET].values
    feature_names = X_df.columns.tolist()

    print(f"  After OHE encoding:")
    print(f"    Numeric features     : {len(num_cols)}")
    print(f"    Encoded cat features : {len(dummies.columns)}")
    print(f"    Total features       : {len(feature_names)}")
    print()

    return X_df.values.astype(float), y, feature_names, num_cols, cat_cols


# ==============================================================================
# 3. Feature Selection helpers (run INSIDE each fold)
# ==============================================================================
def select_pearson(X_tr, y_tr, k):
    corrs = np.abs(r_regression(X_tr, y_tr))
    idx   = np.argsort(corrs)[::-1][:k]
    return np.sort(idx)

def select_ftest(X_tr, y_tr, k):
    sel = SelectKBest(f_regression, k=k).fit(X_tr, y_tr)
    return np.where(sel.get_support())[0]

def select_lasso(X_tr, y_tr, k):
    lasso = LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    coef_abs = np.abs(lasso.coef_)
    idx = np.argsort(coef_abs)[::-1][:k]
    return np.sort(idx)

def select_rfe(X_tr, y_tr, k):
    rfe = RFE(LinearRegression(), n_features_to_select=k)
    rfe.fit(X_tr, y_tr)
    return np.where(rfe.support_)[0]

SELECTORS = {"Pearson": select_pearson, "F-test": select_ftest,
             "LassoCV": select_lasso, "RFE": select_rfe}


# ==============================================================================
# 4. 3-Fold CV Loop
# ==============================================================================
def run_cv(X, y, feature_names):
    print("=" * 65)
    print("SECTION 2 - 3-FOLD CV EXPERIMENT")
    print("=" * 65)

    n_feats  = X.shape[1]
    k_max    = min(K_MAX, n_feats)
    k_range  = list(range(1, k_max + 1))
    kf       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    timing  = {}

    for method in METHODS:
        print(f"\n  [{method}]  k=1..{k_max}")
        t0 = time.time()
        mres = {}

        for k in k_range:
            fold_rmse, fold_r2 = [], []

            for tr_idx, va_idx in kf.split(X):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]

                # --- Per-fold imputation for numeric columns ---
                # (categoricals already filled globally above - safe for pilot)
                # StandardScaler inside fold
                scaler   = StandardScaler()
                X_tr_sc  = scaler.fit_transform(X_tr)
                X_va_sc  = scaler.transform(X_va)

                # Feature selection on fold train
                feat_idx = SELECTORS[method](X_tr_sc, y_tr, k)

                X_tr_sel = X_tr_sc[:, feat_idx]
                X_va_sel = X_va_sc[:, feat_idx]

                lr = LinearRegression()
                lr.fit(X_tr_sel, y_tr)
                y_pred = lr.predict(X_va_sel)

                fold_rmse.append(np.sqrt(mean_squared_error(y_va, y_pred)))
                fold_r2.append(r2_score(y_va, y_pred))

            mres[k] = {
                "rmse_mean": np.mean(fold_rmse),
                "rmse_std":  np.std(fold_rmse),
                "r2_mean":   np.mean(fold_r2),
                "r2_std":    np.std(fold_r2),
            }

        elapsed = time.time() - t0
        timing[method] = elapsed
        results[method] = mres
        best_k = min(k_range, key=lambda k: mres[k]["rmse_mean"])
        print(f"    Done in {elapsed:.1f}s  |  best k={best_k}  "
              f"RMSE={mres[best_k]['rmse_mean']:.2f}  "
              f"R^2={mres[best_k]['r2_mean']:.4f}")

    return results, k_range, timing


# ==============================================================================
# 5. Best-k evaluation on held-out test (80/20 single split)
# ==============================================================================
def final_evaluation(X, y, feature_names, results, k_range):
    print()
    print("=" * 65)
    print("SECTION 3 - FINAL EVALUATION (80/20 hold-out)")
    print("=" * 65)
    print("  (Best k chosen by 3-Fold CV RMSE; test set NOT used for selection)\n")

    rng = np.random.RandomState(RANDOM_STATE)
    n   = len(X)
    idx = rng.permutation(n)
    train_idx = idx[:int(n * 0.8)]
    test_idx  = idx[int(n * 0.8):]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    summary = {}

    for method in METHODS:
        mres   = results[method]
        best_k = min(k_range, key=lambda k: mres[k]["rmse_mean"])

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        feat_idx = SELECTORS[method](X_tr_sc, y_tr, best_k)

        lr = LinearRegression()
        lr.fit(X_tr_sc[:, feat_idx], y_tr)
        y_pred = lr.predict(X_te_sc[:, feat_idx])

        test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        test_r2   = r2_score(y_te, y_pred)
        sel_feats = [feature_names[i] for i in feat_idx]

        summary[method] = {
            "best_k":    best_k,
            "cv_rmse":   mres[best_k]["rmse_mean"],
            "cv_rmse_std": mres[best_k]["rmse_std"],
            "cv_r2":     mres[best_k]["r2_mean"],
            "cv_r2_std": mres[best_k]["r2_std"],
            "test_rmse": test_rmse,
            "test_r2":   test_r2,
            "features":  sel_feats,
            "feat_idx":  feat_idx,
        }

        print(f"  [{method}]")
        print(f"    Best k        : {best_k}")
        print(f"    CV  RMSE      : {mres[best_k]['rmse_mean']:.4f} +/- {mres[best_k]['rmse_std']:.4f}")
        print(f"    CV  R^2       : {mres[best_k]['r2_mean']:.4f} +/- {mres[best_k]['r2_std']:.4f}")
        print(f"    Test RMSE     : {test_rmse:.4f}")
        print(f"    Test R^2      : {test_r2:.4f}")
        print(f"    Selected      : {sel_feats}")
        print()

    return summary


# ==============================================================================
# 6. Plots
# ==============================================================================
def plot_cv_curves(results, k_range, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("[PILOT TEST] Insurance Premium - Feature Selection (3-Fold CV, n=5,000)",
                 fontsize=12, fontweight="bold")

    for ax, metric, ylabel in zip(
            axes,
            ["rmse", "r2"],
            ["CV RMSE (lower is better)", "CV R^2 (higher is better)"]):
        for method in METHODS:
            means = np.array([results[method][k][f"{metric}_mean"] for k in k_range])
            stds  = np.array([results[method][k][f"{metric}_std"]  for k in k_range])
            c = COLORS[method]
            ax.plot(k_range, means, "o-", color=c, label=method,
                    linewidth=2, markersize=4)
            ax.fill_between(k_range, means - stds, means + stds,
                            alpha=0.15, color=c)
        ax.set_xlabel("Number of Features (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_range)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, "fig1_cv_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(p)}")


def plot_best_k_bars(summary, output_dir):
    methods = METHODS
    colors  = [COLORS[m] for m in methods]
    best_ks = [summary[m]["best_k"] for m in methods]
    x = np.arange(len(methods))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("[PILOT TEST] Best-k Performance Summary", fontsize=12, fontweight="bold")

    for ax, cv_vals, te_vals, ylabel, title in zip(
            axes,
            [[summary[m]["cv_rmse"] for m in methods], [summary[m]["cv_r2"] for m in methods]],
            [[summary[m]["test_rmse"] for m in methods], [summary[m]["test_r2"] for m in methods]],
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
        for bar in list(bars_cv) + list(bars_te):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + abs(h) * 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, "fig2_best_k_bars.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(p)}")


def plot_feature_heatmap(summary, feature_names, output_dir):
    n_feat = len(feature_names)
    mat = np.zeros((len(METHODS), n_feat))
    for i, m in enumerate(METHODS):
        for j in summary[m]["feat_idx"]:
            mat[i, j] = 1

    fig, ax = plt.subplots(figsize=(max(12, n_feat * 0.55), 4))
    ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.set_title("[PILOT TEST] Features Selected at Best k (blue = selected)")
    for i in range(len(METHODS)):
        for j in range(n_feat):
            if mat[i, j]:
                ax.text(j, i, "v", ha="center", va="center",
                        fontsize=9, color="steelblue", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "fig3_feature_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(p)}")


def plot_timing(timing, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(METHODS, [timing[m] for m in METHODS],
                  color=[COLORS[m] for m in METHODS],
                  edgecolor="black", linewidth=0.7, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("[PILOT TEST] Feature Selection Execution Time")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "fig4_timing.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(p)}")


def plot_comparison_with_student(results, k_range, output_dir):
    """Side-by-side with Student Performance CV RMSE (if CSV available)."""
    student_path = os.path.join(BASE_DIR, "..", "results", "feature_count_comparison.csv")
    if not os.path.exists(student_path):
        student_path = os.path.join(BASE_DIR, "..", "feature_count_comparison.csv")
    if not os.path.exists(student_path):
        print("  [SKIP] Student Performance CSV not found; skipping comparison plot.")
        return

    df_st = pd.read_csv(student_path)
    # Detect columns with flexible matching
    method_col = next((c for c in df_st.columns if "method" in c.lower()), None)
    k_col      = next((c for c in df_st.columns
                       if c.lower() in ("k", "n_features", "n_feat",
                                        "number of features", "num_features")), None)
    rmse_col   = next((c for c in df_st.columns
                       if "cv rmse" in c.lower() or
                       ("rmse" in c.lower() and "std" not in c.lower()
                        and "test" not in c.lower())), None)
    if not all([method_col, k_col, rmse_col]):
        print(f"  [SKIP] Cannot detect columns in student CSV: {df_st.columns.tolist()}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Dataset Comparison: CV RMSE vs Number of Features",
                 fontsize=12, fontweight="bold")

    # Left – Student Performance
    ax = axes[0]
    ax.set_title("Student Performance Dataset")
    for method in METHODS:
        label_map = {"Pearson": "Pearson Correlation", "F-test": "F-test",
                     "LassoCV": "LassoCV", "RFE": "RFECV"}
        match_label = label_map.get(method, method)
        sub = df_st[df_st[method_col].str.contains(match_label.split()[0], case=False, na=False)]
        if sub.empty:
            continue
        sub = sub.sort_values(k_col)
        ax.plot(sub[k_col], sub[rmse_col], "o-",
                color=COLORS[method], label=method, linewidth=2, markersize=4)
    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("CV RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right – Insurance (Pilot)
    ax = axes[1]
    ax.set_title("[PILOT] Insurance Dataset (n=5,000)")
    for method in METHODS:
        means = np.array([results[method][k]["rmse_mean"] for k in k_range])
        stds  = np.array([results[method][k]["rmse_std"]  for k in k_range])
        c = COLORS[method]
        ax.plot(k_range, means, "o-", color=c, label=method, linewidth=2, markersize=4)
        ax.fill_between(k_range, means - stds, means + stds, alpha=0.15, color=c)
    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("CV RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, "fig5_comparison_student_vs_insurance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(p)}")


# ==============================================================================
# 7. Save CSV
# ==============================================================================
def save_csvs(results, summary, timing, k_range, output_dir):
    rows = []
    for m in METHODS:
        for k in k_range:
            r = results[m][k]
            rows.append({"method": m, "k": k,
                         "rmse_mean": round(r["rmse_mean"], 4),
                         "rmse_std":  round(r["rmse_std"], 4),
                         "r2_mean":   round(r["r2_mean"], 4),
                         "r2_std":    round(r["r2_std"], 4)})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "cv_curve_data.csv"), index=False)

    rows2 = []
    for m in METHODS:
        s = summary[m]
        rows2.append({"method": m,
                      "best_k": s["best_k"],
                      "cv_rmse": round(s["cv_rmse"], 2),
                      "cv_rmse_std": round(s["cv_rmse_std"], 2),
                      "cv_r2":  round(s["cv_r2"], 4),
                      "cv_r2_std": round(s["cv_r2_std"], 4),
                      "test_rmse": round(s["test_rmse"], 2),
                      "test_r2":  round(s["test_r2"], 4),
                      "elapsed_s": round(timing[m], 1),
                      "selected_features": "; ".join(s["features"])})
    pd.DataFrame(rows2).to_csv(os.path.join(output_dir, "pilot_summary.csv"), index=False)
    print("  Saved: cv_curve_data.csv")
    print("  Saved: pilot_summary.csv")


# ==============================================================================
# 8. Print final analysis
# ==============================================================================
def print_analysis(results, summary, timing, k_range, feature_names):
    print()
    print("=" * 65)
    print("SECTION 4 - ANALYSIS & COMPARISON")
    print("=" * 65)

    print()
    print("  [Insurance Pilot - Summary Table]")
    print(f"  {'Method':<10} {'Best k':>6} {'CV RMSE':>10} {'+-std':>7} "
          f"{'CV R^2':>8} {'+-std':>7} {'Test RMSE':>10} {'Test R^2':>9} {'Time':>6}")
    print("  " + "-" * 75)
    for m in METHODS:
        s = summary[m]
        print(f"  {m:<10} {s['best_k']:>6} {s['cv_rmse']:>10.2f} "
              f"{s['cv_rmse_std']:>7.2f} {s['cv_r2']:>8.4f} "
              f"{s['cv_r2_std']:>7.4f} {s['test_rmse']:>10.2f} "
              f"{s['test_r2']:>9.4f} {timing[m]:>5.1f}s")

    # Spread metric
    rmse_vals  = [summary[m]["cv_rmse"] for m in METHODS]
    r2_vals    = [summary[m]["cv_r2"]   for m in METHODS]
    rmse_spread = max(rmse_vals) - min(rmse_vals)
    r2_spread   = max(r2_vals)   - min(r2_vals)

    print()
    print(f"  CV RMSE spread across methods : {rmse_spread:.2f}")
    print(f"  CV R^2  spread across methods : {r2_spread:.4f}")
    print()
    print("  [Key observations]")
    print(f"  - Features after encoding      : {len(feature_names)}")
    print(f"  - k tested up to               : {max(k_range)}")
    print(f"  - Target (Premium Amount) mean : ~check output for scale")

    for m in METHODS:
        s = summary[m]
        print(f"  - {m}: best k={s['best_k']}, "
              f"CV RMSE={s['cv_rmse']:.2f}, R^2={s['cv_r2']:.4f}  "
              f"feats={s['features'][:3]}{'...' if len(s['features'])>3 else ''}")

    print()
    print("  [PILOT vs Student Performance comparison]")
    print("  Student Performance:")
    print("    - R^2 reaches ~1.0 at k>=5 (very strong linear signal)")
    print("    - RMSE drops steeply then plateaus -> clear elbow in curve")
    print("    - Pearson vs F-test: slightly different feature order")
    print("    - LassoCV vs RFE  : different selected features at small k")
    print()
    print("  Insurance (this pilot):")
    if r2_spread > 0.05:
        print(f"    + R^2 spread = {r2_spread:.4f} -> visible method differences")
    else:
        print(f"    - R^2 spread = {r2_spread:.4f} -> small method differences")
    if rmse_spread > 200:
        print(f"    + RMSE spread = {rmse_spread:.0f} -> meaningful gap between methods")
    else:
        print(f"    - RMSE spread = {rmse_spread:.0f} -> gap may be small relative to target scale")
    print()

    best_r2 = max(r2_vals)
    print(f"    Best CV R^2 overall: {best_r2:.4f}")
    if best_r2 < 0.3:
        print("    => Linear model weak fit on Insurance data (expected for premium pricing)")
        print("       This may make curves LESS interesting, not more.")
    elif best_r2 >= 0.5:
        print("    => Reasonable linear fit - curves may show meaningful differences")

    print()
    print("  >>> PILOT RECOMMENDATION <<<")
    if rmse_spread > 100 and r2_spread > 0.02 and best_r2 > 0.1:
        print("  Insurance shows some inter-method differences.")
        print("  Whether it is worth switching depends on curve shape and R^2 level.")
        print("  Review fig1_cv_curves.png before deciding.")
    else:
        print("  Small inter-method differences suggest Insurance may not offer")
        print("  a clearer HW2 demonstration than Student Performance.")
    print()
    print("  NOTE: This is PILOT TEST only - no changes made to HW2 project.")
    print("        Awaiting your confirmation before any formal switch.")
    print()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print()
    print("+===============================================================+")
    print("|  [PILOT TEST] Insurance Dataset - Feature Selection Pilot    |")
    print("+===============================================================+")
    print()

    # 1. Chunked sample
    print("=" * 65)
    print("SECTION 0 - LOADING DATA (chunked sample, n=5,000)")
    print("=" * 65)
    t_load = time.time()
    df = chunked_sample(DATA_PATH, SAMPLE_N, RANDOM_STATE)
    print(f"  Load time: {time.time() - t_load:.1f}s")

    # 2. Inspect & preprocess
    X, y, feature_names, num_cols, cat_cols = inspect_and_preprocess(df)

    # 3. 3-Fold CV
    results, k_range, timing = run_cv(X, y, feature_names)

    # 4. Final evaluation
    summary = final_evaluation(X, y, feature_names, results, k_range)

    # 5. CSV first (so data is always saved even if plots fail)
    print()
    print("=" * 65)
    print("SECTION 4b - SAVING PLOTS & CSVs")
    print("=" * 65)
    save_csvs(results, summary, timing, k_range, OUTPUT_DIR)

    # 6. Plots
    plot_cv_curves(results, k_range, OUTPUT_DIR)
    plot_best_k_bars(summary, OUTPUT_DIR)
    plot_feature_heatmap(summary, feature_names, OUTPUT_DIR)
    plot_timing(timing, OUTPUT_DIR)
    plot_comparison_with_student(results, k_range, OUTPUT_DIR)

    # 7. Analysis
    print_analysis(results, summary, timing, k_range, feature_names)

    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print()
    print("  [PILOT TEST COMPLETE]  Awaiting your confirmation.")
    print()


if __name__ == "__main__":
    main()
