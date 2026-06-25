# Credit Default Prediction
Python · Scikit-learn · XGBoost · SHAP · Imbalanced-learn

---

## The Problem

A lender's biggest question before issuing a loan is simple: will this borrower pay it back?
Getting that answer right - at scale, across hundreds of thousands of applications - requires
more than intuition or simple rules. It requires a model that can learn complex patterns
across borrower profiles and generalize to new applicants.

This project builds and compares three classification models to predict whether a borrower
will default, using origination-time features only - the information a lender would actually
have before approving a loan.

---

## The Data

Lending Club loan data via [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
266,262 resolved loans (fully paid or defaulted) drawn from the full 2.2M record dataset.
Loans with ambiguous outcomes (Current, In Grace Period) were excluded - the model only
trains on loans where the final outcome is known.

**Class distribution:**
- Non-Default: 208,944 (78.47%)
- Default: 57,318 (21.53%)

21.53% default rate - imbalanced but not extreme. Handled with SMOTE on the training set.

> Note: Due to memory constraints (8GB RAM), Random Forest was trained with
> `max_depth=10` and `max_samples=0.5`. All other models are unconstrained.

---

## Features

18 origination-time features were selected - no post-issuance data included:

| Feature | Description |
|---|---|
| `loan_amnt` | Loan amount requested |
| `term` | Loan term (36 or 60 months) |
| `int_rate` | Interest rate assigned |
| `installment` | Monthly payment amount |
| `grade` / `sub_grade` | Lending Club risk grade |
| `emp_length` | Employment length (0–10+ years) |
| `home_ownership` | Rent / Own / Mortgage |
| `annual_inc` | Annual income |
| `verification_status` | Income verification status |
| `purpose` | Loan purpose |
| `dti` | Debt-to-income ratio |
| `delinq_2yrs` | Delinquencies in past 2 years |
| `open_acc` | Number of open credit lines |
| `pub_rec` | Public derogatory records |
| `revol_bal` | Revolving credit balance |
| `revol_util` | Revolving utilization rate |
| `total_acc` | Total credit lines ever |

---

## Modeling Approach

Three classifiers were trained and compared:

**Logistic Regression** — interpretable baseline. Trained on SMOTE-balanced data with
standard scaling. Strong performance relative to complexity.

**Random Forest** — ensemble of 50 decision trees with depth and sample constraints
for memory efficiency.

**XGBoost** — gradient boosted trees. Best overall performance. Used for SHAP
explainability analysis.

All models evaluated on a held-out test set (20% of data, stratified). SMOTE applied
to training set only — test set reflects true class distribution.

---

## Results

| Model | AUC | KS Statistic |
|---|---|---|
| XGBoost | 0.7104 | 0.3016 |
| Logistic Regression | 0.7055 | 0.3005 |
| Random Forest | 0.6942 | 0.2794 |

AUC of ~0.71 is realistic for credit default prediction on public data — industry models
using proprietary bureau data (credit scores, payment history) typically reach 0.75–0.85.
The KS statistic of 0.30 for XGBoost meets the industry threshold for an acceptable
scorecard (KS > 0.20 is considered viable; KS > 0.30 is good).

A notable finding: Logistic Regression (AUC 0.7055) nearly matches XGBoost (AUC 0.7104),
suggesting the relationship between origination features and default is largely linear —
consistent with how traditional scorecards are built in practice.

---

## Explainability — SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed for the XGBoost model to
understand which features drive individual predictions.

**Top drivers of default risk across the portfolio:**
- `int_rate` — strongest continuous predictor; higher rates signal riskier borrowers
- `loan grade` — grade F/G borrowers have substantially higher default probability
- `dti` — higher debt burden increases default risk
- `term` — 60-month loans default more than 36-month loans
- `annual_inc` — higher income mildly protective

**Highest-risk borrower in test sample (predicted default probability: 81.1%):**

Starting from a base rate of 2.9%, this borrower's profile pushed predicted default
probability to 81.1%, driven primarily by Grade F (+0.74), interest rate of 26.99%
(+0.39), debt consolidation purpose (+0.35), and renter status (+0.32).

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Data loading, cleaning, null handling, encoding, EDA visualizations |
| `02_modeling.ipynb` | Train/test split, SMOTE, model training, ROC curves, KS statistic |
| `03_explainability.ipynb` | SHAP summary plot, feature importance, waterfall for highest-risk borrower |

---

## Tech Stack

| | |
|---|---|
| Python 3.12 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Modeling, preprocessing, evaluation |
| XGBoost | Gradient boosting |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| matplotlib / seaborn | Visualization |

---

## Repo Structure

```
credit-default-prediction/
├── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── screenshots/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── default_rate_by_grade.png
│   ├── default_rate_by_dti.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
└── data/
    └── data_source.md
```

---

## Data Source

Lending Club Loan Data via [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
Raw CSV not included — download and place `loan.csv` in `/data` before running notebooks.

---

## Related Projects

This project extends the SQL and Power BI analysis in the
[Credit Risk Scorecard Dashboard](https://github.com/ang-fran/credit-risk-scorecard),
which explores the same dataset from a business intelligence perspective.
Together they form an end-to-end credit risk analytics pipeline:

> SQL pipeline → Business dashboard → Predictive model → Explainability

---

*Angela-Frances Ibhade · MSc Statistics, Brock University (2026)*
