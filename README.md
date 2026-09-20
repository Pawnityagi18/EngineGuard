# EngineGuard

## Explainable AI-Based Predictive Maintenance and Remaining Useful Life Estimation for Turbofan Engines

EngineGuard is an AI-based predictive-maintenance system built around the NASA C-MAPSS FD001 turbofan-engine dataset.

The system uses historical engine operating data and sensor measurements to:

- estimate **Remaining Useful Life (RUL)** in engine cycles,
- estimate **maintenance risk**,
- identify engines that may require maintenance soon,
- compare different temporal sensor-history windows,
- explain model predictions using **Explainable AI (SHAP)**,
- and present the results through an interactive web dashboard.

> **Scope:** The reported research findings apply to the evaluated NASA C-MAPSS FD001 dataset and the experimental configuration described below. The results should not be treated as universally optimal for other datasets or engine types.

---

## 1. Problem Statement

Traditional maintenance strategies can be:

- **Corrective:** maintain after failure.
- **Preventive:** maintain at a predefined interval.
- **Predictive:** use data to estimate the current condition and remaining life before failure.

EngineGuard focuses on the predictive-maintenance approach.

The central question is:

> **Can historical engine sensor data be used to estimate how much useful operating life remains and support maintenance decisions before failure?**

---

## 2. Main Objectives

1. Predict the **Remaining Useful Life (RUL)** of turbofan engines.
2. Convert RUL estimates into a practical **maintenance-risk decision**.
3. Compare machine-learning models for RUL regression.
4. Investigate the effect of different **temporal feature windows**.
5. Explain model predictions using **SHAP-based Explainable AI**.
6. Study the importance of cycle information using a **feature ablation experiment**.
7. Present results through the EngineGuard dashboard.

---

## 3. Dataset

### NASA C-MAPSS FD001

The project uses the **NASA C-MAPSS FD001** subset.

The dataset contains engine trajectories observed across operating cycles with multiple operating-condition and sensor variables.

### Key concepts

- **Engine / unit:** one simulated turbofan engine trajectory.
- **Cycle:** one operating-time step.
- **Sensor:** a measured engine variable.
- **RUL (Remaining Useful Life):** estimated number of cycles remaining before failure.

No physical IoT hardware is required for this project because the sensor history is already provided by the benchmark dataset.

---

## 4. System Architecture

```text
NASA C-MAPSS FD001
        |
        v
Data Loading
        |
        v
RUL Target Creation
        |
        v
Feature Selection
        |
        v
Temporal Feature Engineering
(5 / 10 / 20 / 30 cycle windows)
        |
        +-----------------------+
        |                       |
        v                       v
RUL Regression           Maintenance Classification
        |                       |
        v                       v
Predicted RUL             Risk Probability
        |                       |
        +-----------+-----------+
                    |
                    v
              Evaluation
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Metrics     SHAP    Error Analysis
                    |
                    v
             EngineGuard Dashboard
```

---

## 5. Machine-Learning Tasks

### RUL Prediction — Regression

RUL is a continuous numeric quantity, so the primary problem is a **regression problem**.

The project compares:

- **Random Forest Regressor**
- **Histogram Gradient Boosting Regressor**

Model selection is based on the validation **NASA score** used by the project.

### Maintenance Decision — Classification

The maintenance component converts model outputs into a binary decision:

- `0` → maintenance not predicted
- `1` → maintenance predicted

The system also uses a probability threshold to create the maintenance queue.

---

## 6. Feature Engineering

The current feature-engineering pipeline uses causal rolling statistics calculated separately for each engine trajectory.

For a temporal window of `N` cycles, the system derives sensor-history features such as:

- rolling mean
- rolling standard deviation

The evaluated windows are:

```text
5 cycles
10 cycles
20 cycles
30 cycles
```

A causal window uses only the current and previous observations, avoiding future information during prediction.

Example:

```text
Current cycle = 50

Valid history:
46, 47, 48, 49, 50

Not used:
51, 52, ...
```

---

# 7. Research Questions

### RQ1 — Temporal History

> How does the length of recent sensor history affect RUL prediction performance?

### RQ2 — Explainability

> Which features most strongly influence the model's RUL predictions?

### RQ3 — Cycle Information

> How much does `time_in_cycles` contribute to predictive performance?

---

# 8. Research Experiment 1 — Temporal Window Comparison

The same experimental setup was used while changing only the temporal feature window.

| Window | Selected Model | MAE ↓ | RMSE ↓ | R² ↑ | NASA Score ↓ |
|---:|---|---:|---:|---:|---:|
| 5 cycles | Random Forest | 13.49 | 18.61 | 0.799 | 630.41 |
| 10 cycles | Random Forest | 14.35 | 19.31 | 0.784 | 707.65 |
| 20 cycles | Random Forest | 13.72 | 18.58 | 0.800 | 706.23 |
| **30 cycles** | **Hist. Gradient Boosting** | **13.15** | **16.91** | **0.834** | **453.17** |

### Observation

Among the four evaluated windows, the **30-cycle configuration produced the strongest observed test RUL performance**:

- MAE = **13.15 cycles**
- RMSE = **16.91 cycles**
- R² = **0.834**
- NASA Score = **453.17**

This does not mean that 30 cycles is universally optimal. It is the strongest observed configuration within the evaluated FD001 setup.

---

# 9. Research Experiment 2 — Explainable AI with SHAP

**SHAP (SHapley Additive exPlanations)** is used to interpret model predictions.

The analysis was performed over **100 test engines** using the 30-cycle model.

### Top global features

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | `time_in_cycles` | 11.097 |
| 2 | `sensor_3_mean_30` | 9.084 |
| 3 | `sensor_2_mean_30` | 4.120 |
| 4 | `sensor_11_std_30` | 2.198 |
| 5 | `sensor_14_std_30` | 2.111 |

The global SHAP analysis shows that `time_in_cycles` and `sensor_3_mean_30` were the most influential variables, by mean absolute SHAP value, in the evaluated test set.

> SHAP explains **model behavior**. A high SHAP contribution does not by itself prove that a feature is a physical cause of engine failure.

### Engine 1 case study

- Actual RUL: **112 cycles**
- Predicted RUL: **122.65 cycles**
- Prediction error: **+10.65 cycles**

The largest individual SHAP contribution for this prediction came from `time_in_cycles`.

---

# 10. Research Experiment 3 — Ablation Study

An ablation study removes one feature and measures how model performance changes.

### 30-cycle configuration

| Metric | With `time_in_cycles` | Without `time_in_cycles` |
|---|---:|---:|
| Features | 48 | 47 |
| Selected Model | Hist. Gradient Boosting | Hist. Gradient Boosting |
| MAE ↓ | **13.15** | 13.27 |
| RMSE ↓ | **16.91** | 17.65 |
| R² ↑ | **0.834** | 0.820 |
| NASA Score ↓ | **453.17** | 687.37 |

### Observation

Removing `time_in_cycles` degraded the observed test performance, particularly the NASA score.

This suggests that cycle-index information contributes predictive information in the evaluated FD001 setup.

The result should be interpreted as **model-level predictive dependence**, not as proof of physical causation.

---

# 11. Maintenance Decision Results

The current 30-cycle configuration produced the following test classification results:

| Metric | Value |
|---|---:|
| Maintenance threshold | 11% |
| True Positives (TP) | 25 |
| True Negatives (TN) | 70 |
| False Positives (FP) | 5 |
| False Negatives (FN) | 0 |
| Classification accuracy | 95% |
| Expected value* | $7.0M |

\*The expected-value figure comes from the project's illustrative maintenance-cost assumptions. It is not a measured real-world financial saving.

The observed test classification accuracy is:

```text
(TP + TN) / Total
= (25 + 70) / 100
= 95%
```

---

# 12. Evaluation Metrics

### MAE — Mean Absolute Error

Average absolute difference between predicted and actual RUL.

**Lower is better.**

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly than MAE.

**Lower is better.**

### R² — Coefficient of Determination

Measures how much of the variation in the target is explained by the model.

**Higher is better.**

### NASA Score

An asymmetric RUL evaluation metric used in the C-MAPSS predictive-maintenance setting.

**Lower is better.**

### Classification Accuracy

For the maintenance classifier:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

# 13. EngineGuard Dashboard

The web dashboard provides four main views.

### Fleet Overview

Shows:

- number of engines assessed,
- predicted fleet RUL summary,
- maintenance queue,
- risk distribution,
- search and filtering.

### Maintenance

Shows:

- engines in the maintenance queue,
- priority categories,
- predicted RUL,
- maintenance risk.

### Model Health

Shows:

- MAE,
- RMSE,
- R²,
- NASA score,
- model comparison,
- feature influence.

### Research Findings

Shows:

- temporal-window comparison,
- ablation-study results,
- SHAP findings,
- research conclusion.

---

# 14. Technology Stack

### Machine Learning / Data

- Python
- NumPy
- pandas
- scikit-learn
- SHAP
- joblib

### Project / Environment

- `uv`
- pytest

### Dashboard

- React
- TypeScript
- Vinext
- Vite
- Recharts

### Data Format

- CSV / structured tabular data
- JSON dashboard export

---

# 15. Project Structure

```text
nasa-cmapss-predictive-maintenance/
│
├── data/
│   └── raw/
│
├── artifacts/
│   ├── metrics.json
│   └── fd001_models.joblib
│
├── research/
│   ├── baseline_50_estimators.json
│   ├── window_5.json
│   ├── window_10.json
│   ├── window_20.json
│   ├── window_30.json
│   ├── window_30_without_time.json
│   ├── shap_analysis.py
│   ├── shap_global_feature_importance.csv
│   ├── shap_engine_1.csv
│   ├── shap_global_importance.png
│   ├── shap_engine_1_explanation.png
│   └── plot_shap.py
│
├── scripts/
│   └── export_dashboard_data.py
│
├── src/
│   └── cmapss_maintenance/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       └── modeling.py
│
├── web/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── public/
│   │   └── data/
│   │       └── dashboard.json
│   └── tests/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 16. Reproducibility

### Run the Python test suite

From the project root:

```bash
uv run --with pytest pytest
```

Expected baseline result:

```text
6 passed
```

### Train / run the FD001 experiment

```bash
uv run cmapss-maintenance run --estimators 50 --skip-download
```

### Export dashboard data

```bash
uv run python scripts/export_dashboard_data.py
```

### Run SHAP analysis

```bash
uv run python research/shap_analysis.py
```

### Generate SHAP plots

```bash
uv run python research/plot_shap.py
```

### Run the dashboard in development mode on Windows Git Bash

From `web/`:

```bash
export WRANGLER_LOG_PATH=.wrangler/wrangler.log
npx vinext dev
```

Then open:

```text
http://localhost:3000
```

---

# 17. Research Conclusion

The experiments demonstrate that temporal feature-window selection affects turbofan RUL prediction performance. Among the evaluated 5-, 10-, 20-, and 30-cycle windows, the 30-cycle configuration produced the strongest observed test results, achieving an MAE of 13.15 cycles, RMSE of 16.91 cycles, R² of 0.834, and NASA score of 453.17.

The results also show that a longer history does not automatically improve performance. The selected regression model changed with the temporal representation, with Histogram Gradient Boosting selected for the 30-cycle configuration according to the validation NASA score.

The ablation experiment showed that removing `time_in_cycles` degraded test performance, while SHAP analysis identified `time_in_cycles` and `sensor_3_mean_30` as the most influential features by mean absolute SHAP value across the evaluated test engines.

Overall, the study indicates that temporal feature representation, feature selection, and model interpretability are important components of turbofan RUL prediction. The conclusions are limited to the evaluated NASA C-MAPSS FD001 dataset and experimental configuration.

---

# 18. Current Limitations

- Evaluation is currently focused on **FD001**.
- The experiments use a fixed set of tested temporal windows: 5, 10, 20, and 30 cycles.
- Reported results are from the current experimental configuration and should not be generalized automatically to other datasets.
- The maintenance-value calculation uses illustrative project cost assumptions.
- SHAP describes model behavior and does not establish physical causality.

---

# 19. Future Work

Possible future extensions include:

- evaluation on additional C-MAPSS subsets such as FD002, FD003, and FD004,
- robustness checks across multiple random seeds,
- uncertainty estimation and prediction intervals,
- richer temporal models such as recurrent or attention-based architectures,
- more detailed fleet-level SHAP analysis,
- deployment-oriented monitoring and alerting.

---

## Acknowledgement

This project is a research and application extension of an existing NASA C-MAPSS predictive-maintenance implementation. The original reference repository is:

https://github.com/Saroswat/nasa-cmapss-predictive-maintenance

The EngineGuard research extensions include configurable temporal windows, SHAP-based explainability, feature ablation analysis, research-result artifacts, and an integrated research dashboard.
