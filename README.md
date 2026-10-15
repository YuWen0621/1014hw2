# HW2: Multiple Linear Regression & Feature Selection Explorer
### Student Performance Prediction Machine Learning Challenge (CRISP-DM)

> **Live Interactive Website:** [https://yuwen0621.github.io/1014_DIC/](https://yuwen0621.github.io/1014_DIC/)  
> **Kaggle Dataset:** [Student Performance Prediction Machine Learning Challenge](https://www.kaggle.com/competitions/student-performance-prediction-machine-learning-challenge/data)  
> **Course:** Data Inference & Computing (DIC) — Homework 2: Multiple Linear Regression

---

## 📌 Project Overview

This project presents a rigorous, publication-grade implementation of **Multiple Linear Regression (OLS)** and **Feature Selection** built on the Kaggle **Student Performance Prediction** challenge dataset following the international **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

The goal is to predict students' continuous **Productivity Score** ($0.0 - 100.0$) using 17 candidate behavioral, academic, and lifestyle predictors while systematically comparing **6 Machine Learning Feature Selection Algorithms**. The project demonstrates the fundamental **bias-variance tradeoff**, establishes model **parsimony (Occam's Razor)** at the $k=6$ sweet spot, and computes statistically rigorous **95% Prediction Intervals (PI)** for individual student outcomes.

---

## 📊 Dataset Specification

| Property | Value | Description |
|---|---|---|
| **Source** | Kaggle Challenge | Student Performance Prediction Machine Learning Challenge |
| **Total Samples** | **16,000 rows** | Complete verified records with 0 missing values |
| **Target Variable (Y)** | `productivity_score` | Continuous index ($0.0 - 100.0$), Mean: 50.15, Std: 16.10 |
| **Data Partition** | **80% Train / 20% Test** | 12,800 Training samples / 3,200 Test samples (`random_state=42`) |
| **Cross-Validation** | **5-Fold K-Fold CV** | Strictly fitted on training data only to prevent data leakage |
| **Categorical Encoding** | Dummy Encoding | `gender` encoded with reference level `Female` (`gender_Male`, `gender_Other`) |
| **Total Candidate Predictors (p)** | **17 Features** | 15 quantitative measurements + 2 binary gender indicators |

### Candidate Predictors
1. `study_hours_per_day` (0.5 – 10.0 hrs/day)
2. `focus_score` (30 – 100 pts)
3. `sleep_hours` (3.0 – 10.0 hrs/night)
4. `phone_usage_hours` (0.5 – 12.0 hrs/day)
5. `stress_level` (1 – 10 scale)
6. `attendance_percentage` (40.0% – 100.0%)
7. `gaming_hours` (0.0 – 6.0 hrs/day)
8. `breaks_per_day` (1 – 14 breaks)
9. `coffee_intake_mg` (0 – 500 mg)
10. `exercise_minutes` (0 – 120 min)
11. `assignments_completed` (0 – 19 tasks)
12. `final_grade` (40.0 – 100.0 pts)
13. `social_media_hours` (0.0 – 8.0 hrs/day)
14. `youtube_hours` (0.0 – 6.0 hrs/day)
15. `age` (17 – 30 years)
16. `gender_Male` (Binary dummy: 1 if Male, 0 otherwise)
17. `gender_Other` (Binary dummy: 1 if Other, 0 otherwise)

---

## 🔬 CRISP-DM 6-Phase Methodology

```mermaid
flowchart LR
    A["1. Business Understanding"] --> B["2. Data Understanding"]
    B --> C["3. Data Preparation"]
    C --> D["4. Modeling (6 FS Methods)"]
    D --> E["5. Evaluation & 95% PI"]
    E --> F["6. Deployment (Web & Pickle)"]
```

### Phase 1: Business Understanding
- **Objective:** Formulate an interpretable, parsimonious Multiple Linear Regression model that accurately forecasts student productivity while identifying the essential lifestyle and academic drivers.
- **Constraints:** All feature selection, ranking, and parameter estimations must be conducted **exclusively on the training split** (80%) to avoid lookahead bias and data leakage.

### Phase 2: Data Understanding
- Explored 16,000 samples across 19 raw attributes (`id` and `student_id` dropped as non-predictive identifiers).
- Verified zero missing values across all columns.
- Computed pairwise Pearson correlation matrix:
  - Strong positive correlation with `study_hours_per_day` ($r = 0.73$), `focus_score` ($r = 0.42$), `sleep_hours` ($r = 0.34$), and `attendance_percentage` ($r = 0.17$).
  - Negative correlation with `phone_usage_hours` ($r = -0.33$) and `stress_level` ($r = -0.20$).
  - Distractor features (`coffee_intake_mg`, `final_grade`, `age`, `gaming_hours`) show $|r| < 0.02$.

### Phase 3: Data Preparation
- Dummy-encoded categorical `gender` using reference category `Female` to prevent the dummy variable trap.
- Partitioned dataset into 80% training ($N = 12,800$) and 20% test ($N = 3,200$) using seed 42.
- Precomputed standard-scaled matrices for regularization-based algorithms (`LassoCV`, `RFECV`).

### Phase 4: Modeling (6 Feature Selection Strategies)
Evaluated 6 distinct feature selection paradigms across progression sizes $k = 1 \dots 17$:
1. **Pearson Correlation:** Ranked features descending by absolute correlation $|r(x_j, y)|$ with target.
2. **Forward SFS (Sequential Forward Selection):** Greedily added one feature at each step that minimizes 5-fold cross-validation RMSE with LinearRegression ($k = 1 \dots 17$).
3. **LassoCV (L1 Regularization):** Ranked by magnitude of standardized coefficients $|\beta_{\text{lasso}}|$.
4. **RFECV (Recursive Feature Elimination):** Iteratively pruned least important features using backward elimination.
5. **Exhaustive Feature Selection:** Evaluated all combinations $\binom{17}{k}$ via fast Gram-matrix 5-fold cross-validation ($2^{17}-1 = 131,071$ models).
6. **Baseline OLS:** Ranked features by full-model individual Student's $t$-statistic significance.

### Phase 5: Evaluation & Statistical Inference
- Unanimous consensus across all 6 methods on the **top 6 core predictors**:
  $$\text{Core Predictors} = \{\text{study\_hours}, \text{focus\_score}, \text{sleep\_hours}, \text{phone\_usage}, \text{stress\_level}, \text{attendance}\}$$
- The 6-feature model achieves an optimal balance:
  - **Test RMSE:** $0.0029$ pts
  - **Test $R^2$:** $1.0000$ ($100.0\%$ variance explained)
  - **5-Fold CV RMSE:** $0.0029$ pts
  - **Mean Absolute Error (MAE):** $0.0025$ pts
- Adding features beyond $k=6$ provides zero improvement in RMSE while inflating parameter variance.

### Phase 6: Deployment
- Deployed interactive web dashboard to GitHub Pages using Vanilla HTML5, CSS3, JavaScript, and Chart.js 4.4.
- Embedded serialized model `student_productivity_model.pkl` and precomputed results in `docs/model_data.js`.

---

## 📐 Mathematical Derivation: 95% Prediction Interval

Unlike a confidence interval for the *mean response* $E[Y|X_0]$, a **Prediction Interval (PI)** quantifies uncertainty for a **new, individual student observation** $x_0 = [1, x_{0,1}, x_{0,2}, \dots, x_{0,p}]^T$. It incorporates both parameter estimation uncertainty and the irreducible residual variance $\sigma^2$:

1. **Fitted Point Prediction:**
   $$\hat{y}_0 = x_0^T \hat{\beta}$$

2. **Residual Variance Estimate ($s^2$):**
   $$s^2 = \frac{\text{RSS}}{n - p - 1} = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - p - 1}$$

3. **Standard Error of Prediction ($SE_{\text{pred}}$):**
   $$SE_{\text{pred}}(x_0) = \sqrt{s^2 \left( 1 + x_0^T (X_{\text{train}}^T X_{\text{train}})^{-1} x_0 \right)}$$
   *where $h_0 = x_0^T (X_{\text{train}}^T X_{\text{train}})^{-1} x_0$ is the leverage of the query point.*

4. **Critical Student's $t$ Value:**
   $$t_{\text{crit}} = t_{1 - \alpha/2, \, df = n - p - 1} = t_{0.975, \, 12,793} \approx 1.9601$$

5. **95% Prediction Interval:**
   $$\text{PI}_{95\%}(x_0) = \left[ \hat{y}_0 - t_{\text{crit}} \times SE_{\text{pred}}(x_0), \quad \hat{y}_0 + t_{\text{crit}} \times SE_{\text{pred}}(x_0) \right]$$

### Empirical Verification on Out-of-Sample Test Set ($N = 3,200$)
- **Empirical Coverage Rate:** **$100.00\%$** (3,200 / 3,200 observations inside interval)
- **Mean Interval Width:** $\pm 0.0056$ pts (total width $0.0112$ pts)

---

## 🏆 Final Model Comparison (At Optimal $k$)

| Method | Optimal $k$ | Selected Predictors | Test RMSE | Test $R^2$ | Test MAE | 5-Fold CV RMSE | 95% PI Coverage |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Pearson Correlation** | **8** | Study Efficiency, Study Hours, Focus, Sleep, Phone, Screen Distraction, Stress, Attendance | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |
| **Forward SFS** | **7** | Study Efficiency, Sleep, Phone, Stress, Attendance, Study Hours, Focus | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |
| **LassoCV** | **6** | Study Hours, Focus, Sleep, Phone, Stress, Attendance | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |
| **RFECV** | **6** | Study Hours, Focus, Sleep, Phone, Stress, Attendance | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |
| **Exhaustive Search** | **6** | Study Hours, Sleep, Phone, Attendance, Stress, Focus | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |
| **Baseline OLS** | **6** | Sleep, Study Hours, Stress, Focus, Attendance, Phone | **0.0029** | **1.0000** | **0.0025** | **0.0029** | **100.0%** |

### Optimal Model Parameter Estimates ($k = 6$)
$$\begin{aligned}
\text{Productivity Score} = &-6.2386 \\
&+ 4.3164 \times (\text{study\_hours\_per\_day}) \\
&+ 0.3237 \times (\text{focus\_score}) \\
&+ 2.6977 \times (\text{sleep\_hours}) \\
&- 1.6186 \times (\text{phone\_usage\_hours}) \\
&- 1.0791 \times (\text{stress\_level}) \\
&+ 0.1619 \times (\text{attendance\_percentage})
\end{aligned}$$

- All $p$-values $< 0.0001$ ($t$-statistics $> 100,000$).
- Variance Inflation Factor ($\text{VIF} \approx 1.0$) confirms zero multicollinearity.

---

## 💻 Web Explorer Features

1. **Dataset Information Bar:** Displays sample size, features count, optimal sweet spot, and evaluation setup.
2. **Algorithm Tabs:** Select any of the 6 feature selection methods or toggle **🌟 Compare All 6 Methods**.
3. **Dynamic $k$-Selector (1 to 17):** Switch feature counts in real time, with $k=6$ highlighted as the ★ Optimal Sweet Spot.
4. **Dual Line Charts (Chart.js):** Interactive curves for RMSE and $R^2$ across $k=1 \dots 17$.
5. **Interactive Actual vs. Predicted Chart:** Scatter plot of out-of-sample observations paired with the 45° ideal line and shaded 95% Prediction Interval bands.
6. **Live Student Productivity Predictor:** Real-time calculator with responsive sliders and numeric inputs for active predictors, computing point prediction and **95% Prediction Interval** $[\text{Lower}, \text{Upper}] \pm ME$.
7. **Detailed Statistical Tables:** Full OLS parameters ($\beta$, SE, $t$-stat, $p$-val, 95% CI, VIF) and top 10 test residual diagnostics.
8. **Additional Publication Figures:** High-resolution heatmaps, consensus matrices, and dual-bar comparisons.

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### 2. Run Python Analysis
Execute the standalone pipeline:
```bash
python main.py
```
This script will:
- Load the local Kaggle CSV dataset (`train.csv`).
- Run 5-fold cross-validation and feature selection across all 6 methods.
- Generate all 8 publication figures in `results/`.
- Export all evaluation tables to CSV (`feature_count_comparison.csv`, `final_model_comparison.csv`, etc.).
- Generate `docs/model_data.js` for the web dashboard.
- Save the trained regression model as `student_productivity_model.pkl`.

### 3. Launch Web Application
Open `docs/index.html` directly in any web browser, or launch a local HTTP server:
```bash
python -m http.server 8000 --directory docs
```
Navigate to `http://localhost:8000` to interact with the explorer.

---

## 📁 Repository Structure

```
.
├── main.py                               # Standalone CRISP-DM Python pipeline
├── README.md                             # Comprehensive project documentation
├── student_productivity_model.pkl        # Serialized optimal regression model
├── student-performance-prediction-machine-learning-challenge/
│   ├── train.csv                         # Local Kaggle training dataset (16,000 rows)
│   ├── test.csv                          # Kaggle test features
│   └── sample_submission.csv             # Kaggle submission format
├── docs/                                 # GitHub Pages Web Application
│   ├── index.html                        # Main interactive web dashboard
│   ├── style.css                         # CSS design system (Cream & Sage aesthetic)
│   ├── app.js                            # Frontend application logic & live predictor
│   └── model_data.js                     # Precomputed Python results & 95% PI matrices
└── results/                              # Generated Publication Figures & CSV Reports
    ├── correlation_heatmap.png           # Pairwise Pearson correlation matrix
    ├── dual_charts_rmse_r2.png           # Dual plot: RMSE and R² vs k
    ├── figure1_rmse_vs_features.png      # Publication RMSE curve
    ├── figure2_r2_vs_features.png        # Publication R² curve
    ├── figure3_final_model_comparison.png# Bar chart: 6-method performance comparison
    ├── figure4_selected_features_heatmap.png # Feature consensus selection matrix
    ├── figure5_model_coefficients_table.png  # Rendered coefficients table graphic
    ├── figure6_test_residuals_diagnostic.png # Rendered residuals diagnostic graphic
    ├── figure7_actual_vs_predicted_pi.png    # Actual vs. Predicted with 95% PI bands
    ├── feature_count_comparison.csv      # Metrics for every (method, k) combination
    ├── final_model_comparison.csv        # Metrics at optimal k for all 6 methods
    ├── selected_features_comparison.csv  # Binary selection matrix across algorithms
    ├── model_coefficients_table.csv      # Optimal model parameters and statistics
    ├── test_residuals_diagnostic.csv     # Top test set outliers
    └── test_predictions_with_pi.csv      # Out-of-sample predictions with 95% PI
```

---

## 📖 References & Citations

1. **Dataset:** [Kaggle Student Performance Prediction Challenge](https://www.kaggle.com/competitions/student-performance-prediction-machine-learning-challenge/data)
2. **Methodology:** Wirth, R., & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining.*
3. **Statistical Theory:** Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). *Introduction to Linear Regression Analysis.* John Wiley & Sons.
4. **Course:** Data Inference & Computing (DIC), Fall 2026.
