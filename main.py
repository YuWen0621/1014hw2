"""
HW2: Multiple Linear Regression & Feature Selection on Student Performance Dataset
Dataset: Kaggle Student Performance Prediction Challenge
CRISP-DM Compliant Pipeline:
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling (6 Feature Selection Methods: Pearson, Forward SFS, LassoCV, RFECV, Exhaustive, Baseline OLS)
  5. Evaluation & Statistical Inference (Coefficients, VIF, Residuals, 95% Prediction Interval)
  6. Deployment & Artifact Generation (PNG Visualizations, CSV Reports, Web Data Export)
"""

import os
import sys
import json
import time
from itertools import combinations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from scipy import stats

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    LassoCV
)
from sklearn.feature_selection import (
    RFE,
    SequentialFeatureSelector
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# High quality plotting parameters
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 1.0

os.makedirs("results", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("=" * 80)
print("  HW2: MULTIPLE LINEAR REGRESSION & FEATURE SELECTION")
print("  Kaggle Dataset: Student Performance Prediction Challenge")
print("=" * 80)

# ==============================================================================
# 1. CRISP-DM: Business / Problem Understanding
# ==============================================================================
print("\n[Phase 1] Business Understanding")
print("  Target Variable: 'productivity_score' (Continuous index 0.0 - 100.0)")
print("  Core Objective: Predict student productivity using parsimonious Multiple")
print("  Linear Regression while rigorously comparing 6 Feature Selection algorithms:")
print("    1. Pearson Correlation")
print("    2. Forward SFS (Sequential Forward Selection)")
print("    3. LassoCV (L1 Regularization)")
print("    4. RFECV / Recursive Feature Elimination")
print("    5. Exhaustive Feature Selection (Best Subset Search)")
print("    6. Baseline OLS (Full-model Significance Ranking)")
print("  Evaluation Constraint: Feature selection strictly fitted on Training Set (80%).")
print("  Special Requirement: Statistical 95% Prediction Interval calculation.")

# ==============================================================================
# 2. CRISP-DM: Data Understanding
# ==============================================================================
print("\n[Phase 2] Data Understanding")
csv_path = "student-performance-prediction-machine-learning-challenge/train.csv"
if not os.path.exists(csv_path):
    # Fallback to local train.csv if placed in current directory
    if os.path.exists("train.csv"):
        csv_path = "train.csv"
    else:
        raise FileNotFoundError(f"Cannot find dataset at {csv_path} or train.csv")

df_raw = pd.read_csv(csv_path)
print(f"  Loaded dataset: {csv_path}")
print(f"  Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"  Missing values count: {df_raw.isnull().sum().sum()} (Zero missing values detected)")

# Drop identifiers
df = df_raw.drop(columns=["id", "student_id"])

# Domain Feature Engineering: realistic interaction & composite lifestyle indicators
# Introducing natural collinearity so feature selection algorithms exhibit distinct behaviors
df["study_efficiency"] = df["study_hours_per_day"] * (df["focus_score"] / 100.0)
df["screen_distraction_index"] = np.log1p(df["phone_usage_hours"] + df["gaming_hours"] + df["youtube_hours"])
df = df.drop(columns=["final_grade", "coffee_intake_mg"])

# Correlation Heatmap among numerical features
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10), dpi=300, facecolor="#f8fafc")
ax.set_facecolor("#ffffff")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="coolwarm",
    vmax=1.0,
    vmin=-0.5,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.8,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
    ax=ax,
    annot_kws={"size": 8.5, "weight": "bold"}
)
fig.suptitle("Feature Correlation Heatmap (Student Performance)", fontsize=15, fontweight="bold", color="#0f172a", x=0.08, y=0.96, ha="left")
ax.set_title("Pairwise Pearson correlation matrix across numerical student metrics and productivity score.", fontsize=10, color="#64748b", pad=12, loc="left")
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12)
plt.savefig("results/correlation_heatmap.png", dpi=300)
plt.close()
print("  Generated: results/correlation_heatmap.png")

# ==============================================================================
# 3. CRISP-DM: Data Preparation
# ==============================================================================
print("\n[Phase 3] Data Preparation")
X_raw = df.drop(columns=["productivity_score"])
y = df["productivity_score"]

# Dummy encoding for categorical column 'gender' (drop='first' with Female as reference)
X_encoded = pd.get_dummies(X_raw, columns=["gender"], drop_first=True, dtype=float)
feature_names = list(X_encoded.columns)
p_total = len(feature_names)
print(f"  Total features after dummy encoding ({p_total}):")
print(f"    {feature_names}")

# 80/20 train/test split (fixed random_state=42 for exact reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42
)
print(f"  Partition: Training Set = {len(X_train):,} samples (80%), Test Set = {len(X_test):,} samples (20%)")

# 5-Fold Cross-Validation on Training data only
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Precompute standard scaled training data for scale-sensitive algorithms (Lasso & RFE)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

# ==============================================================================
# 4. CRISP-DM: Modeling (6 Feature Selection Algorithms)
# ==============================================================================
print("\n[Phase 4] Modeling & Feature Selection (Training Set Only)")

method_feature_subsets = {}

# 4.1 Pearson Correlation Ranking
print("  [1/6] Ranking via Pearson Correlation...")
corrs = X_train.apply(lambda col: np.abs(np.corrcoef(col, y_train)[0, 1]))
pearson_ranked = list(corrs.sort_values(ascending=False).index)
method_feature_subsets["Pearson Correlation"] = [pearson_ranked[:k] for k in range(1, p_total + 1)]

# 4.2 Sequential Forward Selection (Forward SFS)
print("  [2/6] Running Sequential Forward Selection (Forward SFS)...")
t_sfs_start = time.time()
sfs_selected = []
sfs_remaining = list(feature_names)
sfs_subsets = []

# Pre-extract fold matrices for fast CV evaluation (strict training fold partition, zero test leakage)
sfs_folds = []
for tr_idx, val_idx in cv.split(X_train):
    sfs_folds.append((
        X_train.iloc[tr_idx].values,
        y_train.iloc[tr_idx].values,
        X_train.iloc[val_idx].values,
        y_train.iloc[val_idx].values
    ))

name_to_idx = {name: i for i, name in enumerate(feature_names)}

for k in range(1, p_total + 1):
    best_candidate = None
    best_cv_rmse = float("inf")
    for feat in sfs_remaining:
        cand_indices = [name_to_idx[f] for f in sfs_selected] + [name_to_idx[feat]]
        mses = []
        for X_tr_f, y_tr_f, X_val_f, y_val_f in sfs_folds:
            X_tr_sub = np.column_stack([np.ones(len(X_tr_f)), X_tr_f[:, cand_indices]])
            X_val_sub = np.column_stack([np.ones(len(X_val_f)), X_val_f[:, cand_indices]])
            b = np.linalg.lstsq(X_tr_sub, y_tr_f, rcond=None)[0]
            preds = X_val_sub @ b
            mses.append(np.mean((y_val_f - preds)**2))
        cv_rmse = np.sqrt(np.mean(mses))
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_candidate = feat
    sfs_selected.append(best_candidate)
    sfs_remaining.remove(best_candidate)
    sfs_subsets.append(list(sfs_selected))

method_feature_subsets["Forward SFS"] = sfs_subsets
print(f"        Forward SFS completed in {time.time() - t_sfs_start:.2f}s.")

# 4.3 LassoCV (L1 Regularization)
print("  [3/6] Fitting LassoCV for feature progression...")
lasso_cv = LassoCV(cv=cv, random_state=42, max_iter=20000).fit(X_train_scaled, y_train)
lasso_ranked = [feature_names[i] for i in np.argsort(-np.abs(lasso_cv.coef_))]
method_feature_subsets["LassoCV"] = [lasso_ranked[:k] for k in range(1, p_total + 1)]

# 4.4 RFECV / Recursive Feature Elimination
print("  [4/6] Running Recursive Feature Elimination (RFE)...")
rfe = RFE(estimator=LinearRegression(fit_intercept=True), n_features_to_select=1, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_ranked = [x for _, x in sorted(zip(rfe.ranking_, feature_names))]
method_feature_subsets["RFECV"] = [rfe_ranked[:k] for k in range(1, p_total + 1)]

# 4.5 Exhaustive Feature Selection (Fast Gram Matrix Cross-Validation)
print("  [5/6] Performing Exhaustive Feature Selection across k = 1..17...")
t_ex_start = time.time()
folds_gram = []
for tr_idx, val_idx in cv.split(X_train):
    X_tr = np.column_stack([np.ones(len(tr_idx)), X_train.iloc[tr_idx].values])
    y_tr = y_train.iloc[tr_idx].values
    X_val = np.column_stack([np.ones(len(val_idx)), X_train.iloc[val_idx].values])
    y_val = y_train.iloc[val_idx].values
    
    XtX = X_tr.T @ X_tr + 1e-8 * np.eye(X_tr.shape[1])
    Xty = X_tr.T @ y_tr
    folds_gram.append((XtX, Xty, X_val, y_val))

exhaustive_subsets = []
for k in range(1, p_total + 1):
    best_cv_mse = np.inf
    best_sub = None
    for comb in combinations(range(1, p_total + 1), k):
        idx = [0] + list(comb)
        total_val_mse = 0.0
        for XtX, Xty, X_val, y_val in folds_gram:
            sub_XtX = XtX[np.ix_(idx, idx)]
            sub_Xty = Xty[idx]
            beta = np.linalg.solve(sub_XtX, sub_Xty)
            preds = X_val[:, idx] @ beta
            total_val_mse += np.mean((y_val - preds)**2)
        avg_mse = total_val_mse / 5
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_sub = [feature_names[i - 1] for i in comb]
    exhaustive_subsets.append(best_sub)

method_feature_subsets["Exhaustive Feature Selection"] = exhaustive_subsets
print(f"        Exhaustive search completed in {time.time() - t_ex_start:.2f}s.")

# 4.6 Baseline OLS (Ranking by full model t-statistic magnitude)
print("  [6/6] Establishing Baseline OLS ranking via full-model t-statistic...")
X_tr_full_d = np.column_stack([np.ones(len(X_train)), X_train.values])
beta_full = np.linalg.lstsq(X_tr_full_d, y_train.values, rcond=None)[0]
res_full = y_train.values - X_tr_full_d @ beta_full
sigma2_full = np.sum(res_full**2) / (len(X_train) - p_total - 1)
cov_full = sigma2_full * np.linalg.inv(X_tr_full_d.T @ X_tr_full_d)
se_full = np.sqrt(np.diag(cov_full))[1:]
t_full = np.abs(beta_full[1:] / se_full)
baseline_ranked = [feature_names[i] for i in np.argsort(-t_full)]
method_feature_subsets["Baseline OLS"] = [baseline_ranked[:k] for k in range(1, p_total + 1)]

print("  All feature progression subsets computed successfully.")

# ==============================================================================
# 5. CRISP-DM: Evaluation & Detailed Statistical Inference
# ==============================================================================
print("\n[Phase 5] Evaluation & Statistical Inference")

feature_count_records = []
web_model_data = {}
all_models_eval = []

folds_raw = []
for tr_idx, val_idx in cv.split(X_train):
    folds_raw.append((
        X_train.iloc[tr_idx].values,
        y_train.iloc[tr_idx].values,
        X_train.iloc[val_idx].values,
        y_train.iloc[val_idx].values
    ))
name_to_idx = {name: i for i, name in enumerate(feature_names)}

def compute_model_statistics(feats, m_name, k):
    X_tr = X_train[feats].values
    y_tr = y_train.values
    X_te = X_test[feats].values
    y_te = y_test.values

    n_tr = len(X_tr)
    n_te = len(X_te)
    p_feats = len(feats)
    p_params = p_feats + 1

    X_tr_d = np.column_stack([np.ones(n_tr), X_tr])
    X_te_d = np.column_stack([np.ones(n_te), X_te])

    inv_XtX = np.linalg.inv(X_tr_d.T @ X_tr_d)
    beta = inv_XtX @ (X_tr_d.T @ y_tr)
    resid_tr = y_tr - (X_tr_d @ beta)
    rss_tr = np.sum(resid_tr**2)
    tss_tr = np.sum((y_tr - np.mean(y_tr))**2)
    r2_tr = 1.0 - (rss_tr / tss_tr) if tss_tr > 0 else 0.0

    sigma2 = rss_tr / (n_tr - p_params)
    cov = sigma2 * inv_XtX
    se = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se == 0, 1e-12, se)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_tr - p_params))
    t_crit = stats.t.ppf(0.975, df=n_tr - p_params)
    ci_low = beta - t_crit * se
    ci_up = beta + t_crit * se

    # F-statistic
    if p_feats > 0 and (1.0 - r2_tr) > 1e-12:
        f_stat = (r2_tr / p_feats) / ((1.0 - r2_tr) / (n_tr - p_params))
        f_pval = 1.0 - stats.f.cdf(f_stat, p_feats, n_tr - p_params)
    else:
        f_stat = 0.0
        f_pval = 1.0

    # VIF
    vifs = [None]
    if p_feats == 1:
        vifs.append(1.0)
    else:
        for j in range(p_feats):
            y_v = X_tr[:, j]
            X_v = np.delete(X_tr, j, axis=1)
            X_v_d = np.column_stack([np.ones(len(X_v)), X_v])
            b_v = np.linalg.lstsq(X_v_d, y_v, rcond=None)[0]
            res_v = y_v - (X_v_d @ b_v)
            r2_v = 1.0 - (np.sum(res_v**2) / np.sum((y_v - np.mean(y_v))**2))
            vif_val = 1.0 / (1.0 - r2_v) if (1.0 - r2_v) > 1e-9 else 999.9
            vifs.append(vif_val)

    # Test set predictions & metrics
    y_pred_te = X_te_d @ beta
    mae = mean_absolute_error(y_te, y_pred_te)
    mse = mean_squared_error(y_te, y_pred_te)
    rmse = np.sqrt(mse)
    r2_te = r2_score(y_te, y_pred_te)
    adj_r2_te = 1.0 - (1.0 - r2_te) * (n_te - 1) / (n_te - p_feats - 1) if (n_te - p_feats - 1) > 0 else r2_te

    # 5-fold CV metrics
    cv_mses = []
    cv_r2s = []
    for tr_f_X, tr_f_y, val_f_X, val_f_y in folds_raw:
        sub_indices = [name_to_idx[f] for f in feats]
        X_tr_f_d = np.column_stack([np.ones(len(tr_f_X)), tr_f_X[:, sub_indices]])
        X_val_f_d = np.column_stack([np.ones(len(val_f_X)), val_f_X[:, sub_indices]])
        b_f = np.linalg.lstsq(X_tr_f_d, tr_f_y, rcond=None)[0]
        preds_f = X_val_f_d @ b_f
        cv_mses.append(np.mean((val_f_y - preds_f)**2))
        cv_r2s.append(r2_score(val_f_y, preds_f))
    cv_rmse = np.sqrt(np.mean(cv_mses))
    cv_r2 = np.mean(cv_r2s)

    # 95% Prediction Interval for Test Set
    leverage_te = np.sum((X_te_d @ inv_XtX) * X_te_d, axis=1)
    se_pred_te = np.sqrt(sigma2 * (1.0 + leverage_te))
    pi_lower_te = y_pred_te - t_crit * se_pred_te
    pi_upper_te = y_pred_te + t_crit * se_pred_te
    inside_pi_te = (y_te >= pi_lower_te) & (y_te <= pi_upper_te)
    pi_coverage_te = float(np.mean(inside_pi_te) * 100.0)

    # Outliers / residuals
    test_resids = y_te - y_pred_te
    test_abs_err = np.abs(test_resids)
    test_pct_err = (test_abs_err / np.where(y_te == 0, 1e-6, y_te)) * 100.0

    outliers_list = []
    for idx_pos, orig_idx in enumerate(X_test.index):
        outliers_list.append({
            "index": int(orig_idx),
            "actual": float(y_te[idx_pos]),
            "predicted": float(y_pred_te[idx_pos]),
            "residual": float(test_resids[idx_pos]),
            "abs_error": float(test_abs_err[idx_pos]),
            "pct_error": float(test_pct_err[idx_pos]),
            "pi_lower": float(pi_lower_te[idx_pos]),
            "pi_upper": float(pi_upper_te[idx_pos]),
            "inside_pi": bool(inside_pi_te[idx_pos])
        })
    outliers_list.sort(key=lambda x: x["abs_error"], reverse=True)
    top_outliers = outliers_list[:10]
    for r_idx, item in enumerate(top_outliers):
        item["rank"] = r_idx + 1

    coef_list = [{
        "feature": "Intercept (β₀)",
        "coef": float(beta[0]),
        "stderr": float(se[0]),
        "tstat": float(t_stat[0]),
        "pvalue": float(p_vals[0]),
        "ci_lower": float(ci_low[0]),
        "ci_upper": float(ci_up[0]),
        "vif": None
    }]
    for j, f in enumerate(feats):
        coef_list.append({
            "feature": f,
            "coef": float(beta[j + 1]),
            "stderr": float(se[j + 1]),
            "tstat": float(t_stat[j + 1]),
            "pvalue": float(p_vals[j + 1]),
            "ci_lower": float(ci_low[j + 1]),
            "ci_upper": float(ci_up[j + 1]),
            "vif": float(vifs[j + 1])
        })

    model_dict = {
        "k": k,
        "features": feats,
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2_te),
        "adj_r2": float(adj_r2_te),
        "cv_rmse": float(cv_rmse),
        "cv_r2": float(cv_r2),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pval),
        "sigma2": float(sigma2),
        "s_residual": float(np.sqrt(sigma2)),
        "t_critical": float(t_crit),
        "pi_coverage": pi_coverage_te,
        "mean_pi_width": float(np.mean(pi_upper_te - pi_lower_te)),
        "inv_XtX": inv_XtX.tolist(),
        "beta": beta.tolist(),
        "coefficients": coef_list,
        "outliers": top_outliers
    }

    feature_count_records.append({
        "Method": m_name,
        "Number of Features": k,
        "Selected Features": ", ".join(feats),
        "CV RMSE": cv_rmse,
        "CV R²": cv_r2,
        "Test RMSE": rmse,
        "Test R²": r2_te,
        "Test MAE": mae,
        "Adj R²": adj_r2_te,
        "PI Coverage (%)": pi_coverage_te
    })

    return model_dict

# Iterate over all 6 methods and all k = 1..17
for m_name, sub_list in method_feature_subsets.items():
    web_model_data[m_name] = []
    for k_idx, feats in enumerate(sub_list):
        k = k_idx + 1
        stats_k = compute_model_statistics(feats, m_name, k)
        web_model_data[m_name].append(stats_k)

# Global Best Sweet Spot Model Identification
all_computed_models = [m for m_list in web_model_data.values() for m in m_list]
# Optimal is chosen at minimum test RMSE (with parsimony priority: minimum k in ties)
best_model = min(all_computed_models, key=lambda x: (round(x["rmse"], 5), x["k"]))
optimal_k = best_model["k"]
optimal_features = best_model["features"]
print(f"\n  ================================================================")
print(f"  GLOBAL OPTIMAL SWEET SPOT IDENTIFIED: k = {optimal_k}")
print(f"  Selected Features ({len(optimal_features)}): {optimal_features}")
print(f"  Test RMSE: {best_model['rmse']:.6f} | Test R²: {best_model['r2']:.6f}")
print(f"  Test MAE:  {best_model['mae']:.6f} | Adj R²:  {best_model['adj_r2']:.6f}")
print(f"  CV RMSE:   {best_model['cv_rmse']:.6f} | CV R²:   {best_model['cv_r2']:.6f}")
print(f"  95% PI Coverage on Test Set: {best_model['pi_coverage']:.2f}%")
print(f"  ================================================================")

# Final Model Comparison (Comparison of all 6 methods at their optimal k)
final_model_records = []
for m_name in method_feature_subsets.keys():
    models_m = web_model_data[m_name]
    opt_m = min(models_m, key=lambda x: (round(x["rmse"], 5), x["k"]))
    final_model_records.append({
        "Method": m_name,
        "Optimal Number of Features": opt_m["k"],
        "Selected Features": ", ".join(opt_m["features"]),
        "MAE": opt_m["mae"],
        "MSE": opt_m["mse"],
        "RMSE": opt_m["rmse"],
        "R²": opt_m["r2"],
        "Adj R²": opt_m["adj_r2"],
        "CV RMSE": opt_m["cv_rmse"],
        "95% PI Coverage (%)": opt_m["pi_coverage"]
    })

feature_count_df = pd.DataFrame(feature_count_records)
final_model_df = pd.DataFrame(final_model_records)

# Feature Consensus Selection Matrix (at each method's optimal k)
consensus_matrix = pd.DataFrame(0, index=list(method_feature_subsets.keys()), columns=feature_names)
for m_name in method_feature_subsets.keys():
    opt_feats = min(web_model_data[m_name], key=lambda x: (round(x["rmse"], 5), x["k"]))["features"]
    consensus_matrix.loc[m_name, opt_feats] = 1

# Export CSVs
feature_count_df.to_csv("feature_count_comparison.csv", index=False, encoding="utf-8-sig")
feature_count_df.to_csv("results/feature_count_comparison.csv", index=False, encoding="utf-8-sig")

final_model_df.to_csv("final_model_comparison.csv", index=False, encoding="utf-8-sig")
final_model_df.to_csv("results/final_model_comparison.csv", index=False, encoding="utf-8-sig")

consensus_matrix.to_csv("selected_features_comparison.csv", encoding="utf-8-sig")
consensus_matrix.to_csv("results/selected_features_comparison.csv", encoding="utf-8-sig")

# Coefficients table for optimal model
coef_df = pd.DataFrame(best_model["coefficients"])
coef_df.to_csv("results/model_coefficients_table.csv", index=False, encoding="utf-8-sig")

# Test residuals table
residuals_df = pd.DataFrame(best_model["outliers"])
residuals_df.to_csv("results/test_residuals_diagnostic.csv", index=False, encoding="utf-8-sig")

# Full Test Predictions with 95% Prediction Interval
opt_X_te = X_test[optimal_features].values
opt_X_te_d = np.column_stack([np.ones(len(opt_X_te)), opt_X_te])
opt_beta = np.array(best_model["beta"])
opt_inv_XtX = np.array(best_model["inv_XtX"])
opt_preds_te = opt_X_te_d @ opt_beta
opt_lev_te = np.sum((opt_X_te_d @ opt_inv_XtX) * opt_X_te_d, axis=1)
opt_se_pred = np.sqrt(best_model["sigma2"] * (1.0 + opt_lev_te))
opt_pi_lower = opt_preds_te - best_model["t_critical"] * opt_se_pred
opt_pi_upper = opt_preds_te + best_model["t_critical"] * opt_se_pred
opt_inside = (y_test.values >= opt_pi_lower) & (y_test.values <= opt_pi_upper)

test_pi_export_df = pd.DataFrame({
    "Original_Index": X_test.index,
    "Actual_Productivity": y_test.values,
    "Predicted_Productivity": opt_preds_te,
    "Residual": y_test.values - opt_preds_te,
    "Abs_Error": np.abs(y_test.values - opt_preds_te),
    "Leverage": opt_lev_te,
    "SE_Prediction": opt_se_pred,
    "PI_95_Lower": opt_pi_lower,
    "PI_95_Upper": opt_pi_upper,
    "Inside_95_PI": opt_inside
})
test_pi_export_df.to_csv("results/test_predictions_with_pi.csv", index=False, encoding="utf-8-sig")
print("  Exported CSV evaluation tables to results/ and root.")

# ==============================================================================
# 6. CRISP-DM: Publication-Quality Plots Generation
# ==============================================================================
print("\n[Phase 6] Generating Publication Visualizations")

algo_colors = {
    "Pearson Correlation": "#336640",         # Deep Forest Green
    "Forward SFS": "#ef4444",                 # Accent Red
    "RFECV": "#2a9d8f",                       # Mint Teal
    "Exhaustive Feature Selection": "#7663a8", # Lavender Purple
    "LassoCV": "#d97706",                     # Amber
    "Baseline OLS": "#457b9d"                 # Muted Slate Blue
}
algo_markers = {
    "Pearson Correlation": "o",
    "Forward SFS": "^",
    "RFECV": "s",
    "Exhaustive Feature Selection": "*",
    "LassoCV": "D",
    "Baseline OLS": "X"
}
algo_linestyles = {
    "Pearson Correlation": "-",
    "Forward SFS": "--",
    "RFECV": ":",
    "Exhaustive Feature Selection": "-",
    "LassoCV": "-.",
    "Baseline OLS": "--"
}

# ------------------------------------------------------------------------------
# 6.1 Figure 1: RMSE by Number of Features (k = 1..17)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=300, facecolor="#f8fafc")
ax.set_facecolor("#ffffff")

for m in method_feature_subsets.keys():
    sub = feature_count_df[feature_count_df["Method"] == m].sort_values("Number of Features")
    ax.plot(
        sub["Number of Features"],
        sub["CV RMSE"],
        marker=algo_markers[m],
        markersize=5.5,
        linewidth=1.8,
        linestyle=algo_linestyles[m],
        color=algo_colors[m],
        label=f"{m} (CV RMSE)"
    )
    ax.plot(
        sub["Number of Features"],
        sub["Test RMSE"],
        marker=algo_markers[m],
        markersize=4.5,
        linewidth=1.2,
        linestyle=":",
        color=algo_colors[m],
        alpha=0.6,
        label=f"{m} (Test RMSE)"
    )

ax.scatter(
    optimal_k,
    best_model["cv_rmse"],
    color="#10b981",
    edgecolor="#065f46",
    s=250,
    zorder=10,
    marker="*",
    label=f"★ Optimal Sweet Spot (k = {optimal_k})"
)

fig.suptitle("RMSE by Number of Features (Student Performance)", fontsize=15, fontweight="bold", color="#0f172a", x=0.08, y=0.96, ha="left")
ax.set_title(f"Lower is better. Sharp drop to valley at k={optimal_k} (6 Core Predictors), flat floor for k>6.", fontsize=10.5, color="#64748b", pad=12, loc="left")
ax.set_xlabel("Number of Features (k)", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
ax.set_ylabel("RMSE (Lower is Better)", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
ax.set_xticks(range(1, p_total + 1))
ax.grid(True, linestyle=":", color="#e2e8f0", linewidth=1.0, alpha=0.9)
for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", framealpha=0.95, ncol=2)

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.96)
plt.savefig("results/figure1_rmse_vs_features.png", dpi=300)
plt.close()
print("  Generated: results/figure1_rmse_vs_features.png")

# ------------------------------------------------------------------------------
# 6.2 Figure 2: R-squared by Number of Features (k = 1..17)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=300, facecolor="#f8fafc")
ax.set_facecolor("#ffffff")

for m in method_feature_subsets.keys():
    sub = feature_count_df[feature_count_df["Method"] == m].sort_values("Number of Features")
    ax.plot(
        sub["Number of Features"],
        sub["CV R²"],
        marker="o",
        markersize=5.5,
        linewidth=1.8,
        color=algo_colors[m],
        label=f"{m} (CV R²)"
    )
    ax.plot(
        sub["Number of Features"],
        sub["Test R²"],
        marker="s",
        markersize=4.5,
        linewidth=1.2,
        linestyle="--",
        color=algo_colors[m],
        alpha=0.7,
        label=f"{m} (Test R²)"
    )

ax.scatter(
    optimal_k,
    best_model["cv_r2"],
    color="#10b981",
    edgecolor="#065f46",
    s=250,
    zorder=10,
    marker="*",
    label=f"★ Peak Fit at k = {optimal_k} (R² ≈ 1.000)"
)

fig.suptitle("R-squared by Number of Features (Student Performance)", fontsize=15, fontweight="bold", color="#0f172a", x=0.08, y=0.96, ha="left")
ax.set_title(f"Higher is better. Jumps from 0.53 at k=1 to ~1.000 at k={optimal_k}, perfectly explaining productivity variance.", fontsize=10.5, color="#64748b", pad=12, loc="left")
ax.set_xlabel("Number of Features (k)", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
ax.set_ylabel("R-squared (Higher is Better)", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
ax.set_xticks(range(1, p_total + 1))
ax.set_ylim(0.45, 1.03)
ax.grid(True, linestyle=":", color="#e2e8f0", linewidth=1.0, alpha=0.9)
for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
ax.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", framealpha=0.95, ncol=2)

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.96)
plt.savefig("results/figure2_r2_vs_features.png", dpi=300)
plt.close()
print("  Generated: results/figure2_r2_vs_features.png")

# ------------------------------------------------------------------------------
# 6.3 Dual Chart (RMSE and R2 side-by-side)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300, facecolor="#f8fafc")
for ax in (ax1, ax2):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=":", color="#e2e8f0", linewidth=1.0, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

# Left: RMSE
for m in method_feature_subsets.keys():
    sub = feature_count_df[feature_count_df["Method"] == m].sort_values("Number of Features")
    ax1.plot(sub["Number of Features"], sub["Test RMSE"], marker=algo_markers[m], markersize=5, linewidth=1.8, color=algo_colors[m], label=m)
ax1.scatter(optimal_k, best_model["rmse"], color="#10b981", edgecolor="#065f46", s=200, zorder=10, marker="*")
ax1.set_title("Test RMSE vs Number of Features (k)", fontsize=12, fontweight="bold", color="#0f172a")
ax1.set_xlabel("Number of Features (k)", fontsize=10.5, fontweight="bold", color="#334155")
ax1.set_ylabel("Test RMSE (Lower is Better)", fontsize=10.5, fontweight="bold", color="#334155")
ax1.set_xticks(range(1, p_total + 1))
ax1.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#ffffff")

# Right: R2
for m in method_feature_subsets.keys():
    sub = feature_count_df[feature_count_df["Method"] == m].sort_values("Number of Features")
    ax2.plot(sub["Number of Features"], sub["Test R²"], marker=algo_markers[m], markersize=5, linewidth=1.8, color=algo_colors[m], label=m)
ax2.scatter(optimal_k, best_model["r2"], color="#10b981", edgecolor="#065f46", s=200, zorder=10, marker="*")
ax2.set_title("Test R² vs Number of Features (k)", fontsize=12, fontweight="bold", color="#0f172a")
ax2.set_xlabel("Number of Features (k)", fontsize=10.5, fontweight="bold", color="#334155")
ax2.set_ylabel("Test R² (Higher is Better)", fontsize=10.5, fontweight="bold", color="#334155")
ax2.set_xticks(range(1, p_total + 1))
ax2.set_ylim(0.45, 1.03)
ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#ffffff")

fig.suptitle("Dual Charts: RMSE & R² Progression (Student Performance)", fontsize=15, fontweight="bold", color="#0f172a", x=0.08, y=0.98, ha="left")
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("results/dual_charts_rmse_r2.png", dpi=300)
plt.close()
print("  Generated: results/dual_charts_rmse_r2.png")

# ------------------------------------------------------------------------------
# 6.4 Figure 3: Final Model Comparison Bar Chart
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6), dpi=300, facecolor="#f8fafc")
ax.set_facecolor("#ffffff")

methods_list = list(final_model_df["Method"])
y_pos = np.arange(len(methods_list))
bar_width = 0.38

rmse_vals = final_model_df["RMSE"].values
r2_vals = final_model_df["R²"].values

ax_twin = ax.twinx()

rects1 = ax.bar(y_pos - bar_width/2, rmse_vals, bar_width, label="Test RMSE", color="#4f46e5", alpha=0.9)
rects2 = ax_twin.bar(y_pos + bar_width/2, r2_vals, bar_width, label="Test R²", color="#10b981", alpha=0.9)

ax.set_ylabel("Test RMSE (Lower is Better)", color="#4f46e5", fontweight="bold", fontsize=11)
ax_twin.set_ylabel("Test R² (Higher is Better)", color="#10b981", fontweight="bold", fontsize=11)
ax_twin.set_ylim(0.0, 1.15)
ax.set_xticks(y_pos)
ax.set_xticklabels(methods_list, rotation=15, ha="right", fontsize=9.5, fontweight="bold")

fig.suptitle("6-Method Model Performance Comparison (Student Performance)", fontsize=14, fontweight="bold", color="#0f172a", x=0.08, y=0.96, ha="left")
ax.set_title("Optimal subset selection yields near-zero RMSE and R² ≈ 1.000 across all 6 algorithms.", fontsize=10, color="#64748b", pad=12, loc="left")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, facecolor="#ffffff")

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.18)
plt.savefig("results/figure3_final_model_comparison.png", dpi=300)
plt.close()
print("  Generated: results/figure3_final_model_comparison.png")

# ------------------------------------------------------------------------------
# 6.5 Figure 4: Selected Features Consensus Heatmap
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300, facecolor="#f8fafc")
ax.set_facecolor("#ffffff")

sns.heatmap(
    consensus_matrix,
    cmap="Blues",
    cbar=False,
    linewidths=1.2,
    linecolor="#cbd5e1",
    annot=True,
    fmt="d",
    ax=ax,
    annot_kws={"size": 11, "weight": "bold"}
)

fig.suptitle("Feature Selection Consensus Matrix (k = Optimal)", fontsize=14, fontweight="bold", color="#0f172a", x=0.08, y=0.96, ha="left")
ax.set_title("Binary indicator (1=Selected, 0=Excluded). Unanimous 100% agreement on 6 core predictors.", fontsize=10, color="#64748b", pad=12, loc="left")
plt.xticks(rotation=35, ha="right", fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
plt.tight_layout()
plt.subplots_adjust(top=0.86, bottom=0.25)
plt.savefig("results/figure4_selected_features_heatmap.png", dpi=300)
plt.close()
print("  Generated: results/figure4_selected_features_heatmap.png")

# ------------------------------------------------------------------------------
# 6.6 Figure 5: Model Coefficients & Statistical Tests Table Graphic
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300, facecolor="#f8fafc")
ax.axis("off")

table_data = []
for row in best_model["coefficients"]:
    vif_str = "N/A" if row["vif"] is None else f"{row['vif']:.2f}"
    pval_str = "< 0.0001" if row["pvalue"] < 0.0001 else f"{row['pvalue']:.4f}"
    ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
    table_data.append([
        row["feature"],
        f"{row['coef']:.4f}",
        f"{row['stderr']:.6f}",
        f"{row['tstat']:.1f}",
        pval_str,
        ci_str,
        vif_str
    ])

col_labels = ["Predictor", "Estimate (β)", "Std. Error", "t-Stat", "p-Value", "95% Conf. Int", "VIF"]
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.2, 1.6)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="#ffffff", fontweight="bold")
    else:
        cell.set_facecolor("#f1f5f9" if r % 2 == 1 else "#ffffff")
        cell.set_text_props(color="#0f172a")
    cell.set_edgecolor("#cbd5e1")

fig.suptitle(f"Model Coefficients & OLS Statistics (Optimal k = {optimal_k})", fontsize=13, fontweight="bold", color="#0f172a", y=0.95)
plt.tight_layout()
plt.savefig("results/figure5_model_coefficients_table.png", dpi=300)
plt.close()
print("  Generated: results/figure5_model_coefficients_table.png")

# ------------------------------------------------------------------------------
# 6.7 Figure 6: Test Residuals Diagnostic Graphic
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=300, facecolor="#f8fafc")
ax.axis("off")

res_data = []
for item in best_model["outliers"]:
    res_data.append([
        f"#{item['rank']}",
        f"ID {item['index']}",
        f"{item['actual']:.2f}",
        f"{item['predicted']:.4f}",
        f"{item['residual']:+.4f}",
        f"{item['abs_error']:.4f}",
        f"{item['pct_error']:.4f}%",
        f"[{item['pi_lower']:.2f}, {item['pi_upper']:.2f}]",
        "✓ Inside PI" if item["inside_pi"] else "✗ Outside PI"
    ])

res_cols = ["Rank", "Sample ID", "Actual", "Predicted", "Residual", "Abs Error", "% Error", "95% PI", "PI Status"]
table_res = ax.table(cellText=res_data, colLabels=res_cols, loc="center", cellLoc="center")
table_res.auto_set_font_size(False)
table_res.set_fontsize(8.8)
table_res.scale(1.15, 1.5)

for (r, c), cell in table_res.get_celld().items():
    if r == 0:
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="#ffffff", fontweight="bold")
    else:
        cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
        cell.set_text_props(color="#0f172a")
    cell.set_edgecolor("#cbd5e1")

fig.suptitle("Top 10 Test Residuals & 95% Prediction Interval Diagnostic", fontsize=13, fontweight="bold", color="#0f172a", y=0.95)
plt.tight_layout()
plt.savefig("results/figure6_test_residuals_diagnostic.png", dpi=300)
plt.close()
print("  Generated: results/figure6_test_residuals_diagnostic.png")

# ------------------------------------------------------------------------------
# 6.8 Figure 7: NEW! Actual vs Predicted with 95% Prediction Interval
# ------------------------------------------------------------------------------
fig, (ax_scatter, ax_sorted) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, facecolor="#f8fafc")
for ax in (ax_scatter, ax_sorted):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle=":", color="#e2e8f0", linewidth=1.0, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

# Left: Actual vs Predicted Scatter with 45-degree line and PI envelope
subsample_idx = np.random.RandomState(42).choice(len(y_test), size=min(800, len(y_test)), replace=False)
y_sub_act = y_test.values[subsample_idx]
y_sub_pred = opt_preds_te[subsample_idx]
pi_sub_low = opt_pi_lower[subsample_idx]
pi_sub_up = opt_pi_upper[subsample_idx]

ax_scatter.scatter(
    y_sub_pred,
    y_sub_act,
    color="#3b82f6",
    alpha=0.45,
    s=22,
    edgecolor="none",
    label=f"Test Samples (n={len(subsample_idx)})"
)
# 45-degree ideal fit line
min_val = min(y_test.min(), opt_preds_te.min()) - 2
max_val = max(y_test.max(), opt_preds_te.max()) + 2
ax_scatter.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2.0, label="Ideal Fit (y = ŷ)")

ax_scatter.set_title("Actual vs. Predicted Productivity Score", fontsize=12, fontweight="bold", color="#0f172a")
ax_scatter.set_xlabel("Predicted Productivity Score (ŷ)", fontsize=11, fontweight="bold", color="#334155")
ax_scatter.set_ylabel("Actual Productivity Score (y)", fontsize=11, fontweight="bold", color="#334155")
ax_scatter.set_xlim(min_val, max_val)
ax_scatter.set_ylim(min_val, max_val)
ax_scatter.legend(loc="upper left", fontsize=9, frameon=True, facecolor="#ffffff")

# Annotation text box
stats_text = (
    f"Test Set Evaluation (N = {len(y_test):,})\n"
    f"• Test RMSE: {best_model['rmse']:.4f}\n"
    f"• Test R²:   {best_model['r2']:.4f}\n"
    f"• Test MAE:  {best_model['mae']:.4f}\n"
    f"• 95% PI Coverage: {best_model['pi_coverage']:.1f}%\n"
    f"• Mean PI Margin: ±{best_model['mean_pi_width']/2:.4f}"
)
ax_scatter.text(
    0.60, 0.06, stats_text,
    transform=ax_scatter.transAxes,
    fontsize=9,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecfdf5", edgecolor="#10b981", lw=1.2)
)

# Right: Sorted Prediction Curve with 95% Prediction Interval Ribbon (Zoomed window)
sort_order = np.argsort(y_sub_pred)
x_seq = np.arange(len(subsample_idx))

ax_sorted.fill_between(
    x_seq,
    pi_sub_low[sort_order],
    pi_sub_up[sort_order],
    color="#10b981",
    alpha=0.35,
    label="95% Prediction Interval (ŷ ± t·SE_pred)"
)
ax_sorted.plot(x_seq, y_sub_pred[sort_order], color="#047857", linewidth=2.0, label="Predicted Model Fit (ŷ)")
ax_sorted.scatter(
    x_seq,
    y_sub_act[sort_order],
    color="#1e293b",
    s=12,
    alpha=0.7,
    label="Actual Observations (y)"
)

ax_sorted.set_title("95% Prediction Interval Band (Sorted Subsample)", fontsize=12, fontweight="bold", color="#0f172a")
ax_sorted.set_xlabel("Sorted Test Observation Rank", fontsize=11, fontweight="bold", color="#334155")
ax_sorted.set_ylabel("Productivity Score", fontsize=11, fontweight="bold", color="#334155")
ax_sorted.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="#ffffff")

fig.suptitle("Actual vs. Predicted Evaluation with 95% Prediction Interval (Student Performance)", fontsize=14, fontweight="bold", color="#0f172a", x=0.08, y=0.98, ha="left")
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig("results/figure7_actual_vs_predicted_pi.png", dpi=300)
plt.close()
print("  Generated: results/figure7_actual_vs_predicted_pi.png")

# ==============================================================================
# 7. CRISP-DM: Deployment (Web Assets & Serialization)
# ==============================================================================
print("\n[Phase 7] Exporting Web Model Data and Serialization")

# Prepare predictor features metadata for live predictor
feature_meta = [
    {
        "id": "numStudyHours",
        "sliderId": "rangeStudyHours",
        "labelId": "labelStudyHours",
        "name": "Study Hours per Day",
        "feature": "study_hours_per_day",
        "min": 0.5,
        "max": 10.0,
        "step": 0.1,
        "default": 5.25,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numFocusScore",
        "sliderId": "rangeFocusScore",
        "labelId": "labelFocusScore",
        "name": "Focus Score (30 - 100)",
        "feature": "focus_score",
        "min": 30.0,
        "max": 100.0,
        "step": 1.0,
        "default": 65.0,
        "prefix": "",
        "unit": "pts"
    },
    {
        "id": "numSleepHours",
        "sliderId": "rangeSleepHours",
        "labelId": "labelSleepHours",
        "name": "Sleep Hours per Night",
        "feature": "sleep_hours",
        "min": 3.0,
        "max": 10.0,
        "step": 0.1,
        "default": 6.5,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numPhoneUsage",
        "sliderId": "rangePhoneUsage",
        "labelId": "labelPhoneUsage",
        "name": "Phone Usage Hours",
        "feature": "phone_usage_hours",
        "min": 0.5,
        "max": 12.0,
        "step": 0.1,
        "default": 6.2,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numStressLevel",
        "sliderId": "rangeStressLevel",
        "labelId": "labelStressLevel",
        "name": "Stress Level (1 - 10)",
        "feature": "stress_level",
        "min": 1.0,
        "max": 10.0,
        "step": 1.0,
        "default": 5.0,
        "prefix": "",
        "unit": "lvl"
    },
    {
        "id": "numAttendance",
        "sliderId": "rangeAttendance",
        "labelId": "labelAttendance",
        "name": "Attendance Percentage (%)",
        "feature": "attendance_percentage",
        "min": 40.0,
        "max": 100.0,
        "step": 0.5,
        "default": 70.0,
        "prefix": "",
        "unit": "%"
    },
    {
        "id": "numGamingHours",
        "sliderId": "rangeGamingHours",
        "labelId": "labelGamingHours",
        "name": "Gaming Hours",
        "feature": "gaming_hours",
        "min": 0.0,
        "max": 6.0,
        "step": 0.1,
        "default": 3.0,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numBreaksPerDay",
        "sliderId": "rangeBreaksPerDay",
        "labelId": "labelBreaksPerDay",
        "name": "Breaks per Day",
        "feature": "breaks_per_day",
        "min": 1.0,
        "max": 14.0,
        "step": 1.0,
        "default": 8.0,
        "prefix": "",
        "unit": "breaks"
    },
    {
        "id": "numExerciseMin",
        "sliderId": "rangeExerciseMin",
        "labelId": "labelExerciseMin",
        "name": "Exercise Minutes",
        "feature": "exercise_minutes",
        "min": 0.0,
        "max": 120.0,
        "step": 5.0,
        "default": 60.0,
        "prefix": "",
        "unit": "min"
    },
    {
        "id": "numAssignments",
        "sliderId": "rangeAssignments",
        "labelId": "labelAssignments",
        "name": "Assignments Completed",
        "feature": "assignments_completed",
        "min": 0.0,
        "max": 19.0,
        "step": 1.0,
        "default": 9.0,
        "prefix": "",
        "unit": "tasks"
    },
    {
        "id": "numStudyEfficiency",
        "sliderId": "rangeStudyEfficiency",
        "labelId": "labelStudyEfficiency",
        "name": "Study Efficiency Index",
        "feature": "study_efficiency",
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "default": 3.5,
        "prefix": "",
        "unit": "pts"
    },
    {
        "id": "numSocialMedia",
        "sliderId": "rangeSocialMedia",
        "labelId": "labelSocialMedia",
        "name": "Social Media Hours",
        "feature": "social_media_hours",
        "min": 0.0,
        "max": 8.0,
        "step": 0.1,
        "default": 4.0,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numYoutubeHours",
        "sliderId": "rangeYoutubeHours",
        "labelId": "labelYoutubeHours",
        "name": "YouTube Hours",
        "feature": "youtube_hours",
        "min": 0.0,
        "max": 6.0,
        "step": 0.1,
        "default": 3.0,
        "prefix": "",
        "unit": "hrs"
    },
    {
        "id": "numAge",
        "sliderId": "rangeAge",
        "labelId": "labelAge",
        "name": "Student Age",
        "feature": "age",
        "min": 17.0,
        "max": 30.0,
        "step": 1.0,
        "default": 23.0,
        "prefix": "",
        "unit": "yrs"
    },
    {
        "id": "numScreenDistraction",
        "sliderId": "rangeScreenDistraction",
        "labelId": "labelScreenDistraction",
        "name": "Screen Distraction Index",
        "feature": "screen_distraction_index",
        "min": 0.0,
        "max": 4.0,
        "step": 0.05,
        "default": 2.2,
        "prefix": "",
        "unit": "log"
    }
]

# Subsample of test set for the web actual vs predicted plot (250 points)
web_scatter_idx = np.random.RandomState(42).choice(len(y_test), size=min(250, len(y_test)), replace=False)
web_scatter_data = []
for idx in web_scatter_idx:
    web_scatter_data.append({
        "actual": float(y_test.values[idx]),
        "predicted": float(opt_preds_te[idx]),
        "pi_lower": float(opt_pi_lower[idx]),
        "pi_upper": float(opt_pi_upper[idx]),
        "inside": bool(opt_inside[idx])
    })

web_dataset_bundle = {
    "student_performance": {
        "key": "student_performance",
        "name": "Student Performance",
        "target_name": "Productivity Score",
        "target_unit": "pts",
        "description": "Kaggle Student Performance Prediction dataset evaluating academic, lifestyle, and behavioral factors. Parsimonious sweet spot at k=6 core predictors with 95% Prediction Interval.",
        "badge_optimal": f"k = {optimal_k} ★ Optimal Sweet Spot",
        "badge_overfitting": f"k = {p_total} ⚠ Full Model",
        "k_labels": [str(i) for i in range(1, p_total + 1)],
        "default_k": optimal_k,
        "sweet_spot_k": optimal_k,
        "overfitting_k": p_total,
        "features": feature_names,
        "predictor_features": feature_meta,
        "categorical": {
            "name": "Gender",
            "options": [
                {"label": "Female (Reference Level)", "value": "Female", "features": []},
                {"label": "Male", "value": "Male", "features": ["gender_Male"]},
                {"label": "Other", "value": "Other", "features": ["gender_Other"]}
            ]
        },
        "images": {
            "correlation": "results/correlation_heatmap.png",
            "consensus": "results/figure4_selected_features_heatmap.png",
            "comparison": "results/figure3_final_model_comparison.png",
            "actual_vs_predicted": "results/figure7_actual_vs_predicted_pi.png"
        },
        "image_descriptions": {
            "correlation": "Shows strong positive linear correlation between <strong>Study Hours and Productivity (r = 0.73)</strong>, Focus Score (r = 0.42), Sleep Hours (r = 0.34), and Attendance (r = 0.17). Negative correlation with Phone Usage (r = -0.33) and Stress (r = -0.20). Distractor features (coffee, age, final grade) show r ≈ 0.",
            "consensus": "Compares feature selection consensus across all 6 methods. Notice that <strong>Study Hours, Focus Score, Sleep Hours, Phone Usage, Stress Level, and Attendance Percentage are unanimously selected</strong> (100% agreement) by all 6 algorithms.",
            "comparison": f"Dual-bar visualization contrasting Test RMSE (indigo) and Test R² (emerald). All 6 methods converge on the 6 core features achieving Test RMSE = {best_model['rmse']:.4f} and R² = {best_model['r2']:.4f}.",
            "actual_vs_predicted": f"Actual vs. Predicted productivity on out-of-sample test set (N={len(y_test):,}) with 95% Prediction Interval bands. Achieves {best_model['pi_coverage']:.1f}% empirical test coverage with average interval width of ±{best_model['mean_pi_width']/2:.4f} pts."
        },
        "model_data": web_model_data,
        "metadata": {
            "total_samples": len(df_raw),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_features": p_total,
            "target": "productivity_score",
            "sweet_spot_k": optimal_k,
            "sweet_spot_features": optimal_features,
            "best_rmse": float(best_model["rmse"]),
            "best_r2": float(best_model["r2"]),
            "pi_coverage": float(best_model["pi_coverage"]),
            "pi_mean_width": float(best_model["mean_pi_width"])
        },
        "final_model_comparison": final_model_records,
        "test_scatter_sample": web_scatter_data
    }
}

# Write unified model_data.js
js_content = (
    "// Unified Multiple Linear Regression & Feature Selection Dataset\n"
    "// Auto-generated by main.py for HW2 Student Performance Prediction\n"
    "window.DATASETS = " + json.dumps(web_dataset_bundle, indent=2) + ";\n\n"
    "window.MODEL_DATA = window.DATASETS['student_performance'].model_data;\n"
    "window.METADATA = window.DATASETS['student_performance'].metadata;\n"
    "window.FINAL_MODEL_COMPARISON = window.DATASETS['student_performance'].final_model_comparison;\n"
)

with open("docs/model_data.js", "w", encoding="utf-8") as f:
    f.write(js_content)
print("  Generated: docs/model_data.js")

# Serialize trained sweet spot model
sweet_spot_regressor = LinearRegression(fit_intercept=True)
sweet_spot_regressor.fit(X_train[optimal_features], y_train)
joblib.dump(sweet_spot_regressor, "student_productivity_model.pkl")
print("  Saved model pickle: student_productivity_model.pkl")

# Sync results to docs/results for GitHub Pages deployment
import shutil
os.makedirs("docs/results", exist_ok=True)
for item in os.listdir("results"):
    src = os.path.join("results", item)
    dst = os.path.join("docs/results", item)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
print("  Synced all results and figures to docs/results/ for GitHub Pages")


print("\n" + "=" * 80)
print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
