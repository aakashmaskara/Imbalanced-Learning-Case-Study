# Handling Imbalanced Dataset — Case Study

Exploratory Data Analysis (EDA) and modeling workflow for a highly **imbalanced classification** problem, focusing on practical strategies to improve minority-class detection while controlling false alarms.

## Introduction

This case study applies **EDA + imbalanced-learning techniques** to a real-world classification task.  
We analyze the dataset, engineer features, and compare multiple strategies (resampling, class weighting, thresholding, calibration) to separate the **minority (positive) class** from the **majority (negative) class**.

Two risks to balance:
1. Missing minority cases (low recall) → **business loss / risk exposure**
2. Triggering too many false positives → **operational cost / alert fatigue**

## Business Understanding

Imbalanced problems are common in domains like **fraud detection, churn, faults, and medical alerts**.  
The goal is to **maximize catches of the minority class** with acceptable precision, so decisions lead to action rather than noise.

## Business Objectives

- Improve **minority-class recall** and **PR-AUC**, while monitoring precision/operational load.  
- Deliver an **auditable pipeline** with interpretable metrics and a tunable decision threshold.  
- Provide guidance on **when to use resampling vs class weights vs calibration**.

## Analytical Approach

1. **Data Understanding**
   - Inspect schema, missingness, target distribution; create a robust train/validation/test split.

2. **EDA**
   - Univariate / segmented univariate by class; leakage checks; stability across splits.

3. **Feature Engineering**
   - Handle categories (one-hot/target-encode as appropriate), scale relevant numerics, create ratios/deltas if meaningful.

4. **Imbalanced-Learning Strategies**
   - **Class weights** (e.g., `class_weight='balanced'`)  
   - **Resampling**: `SMOTE` / `SMOTEENN` / undersampling (evaluated carefully on train only)  
   - **Threshold tuning** using PR/ROC trade-offs and cost-sensitive curves  
   - **Probability calibration** (Platt/Isotonic) when needed for decisioning

5. **Models**
   - Baselines (Logistic Regression), tree-based (Random Forest / XGBoost/LightGBM if used), compare calibrated vs uncalibrated.

6. **Evaluation**
   - Prioritize **PR-AUC**, **Recall (minority)**, **Precision**, **F1**, **ROC AUC**, **Confusion Matrix** at chosen threshold.  
   - Report gains vs a naive baseline.

## Tools & Libraries

- **Python** (Jupyter Notebook)  
- **pandas**, **numpy**, **scikit-learn**, **matplotlib**, **seaborn**  
- **imblearn** (SMOTE/SMOTEENN) if resampling is used

## Key Insights

This workflow is designed to surface:
- Which features are most discriminative for the minority class  
- How resampling vs class weighting affects **recall and PR-AUC**  
- A business-aligned **operating threshold** (probability cutoff) for deployment

*(Exact metrics/plots are in the notebook.)*

## Conclusion & Business Impact

With careful preprocessing, resampling/class-weighting, and **threshold calibration**, the pipeline raises **minority recall** with manageable precision, enabling effective interventions while avoiding alert fatigue.

## Files in this Repository

| File Name | Description |
|-----------|-------------|
| `Handling Imbalanced Dataset - Case Study.ipynb` | Full workflow: EDA → feature engineering → modeling → resampling/weights → thresholding → calibration |
| `train.csv.zip` | Compressed training dataset (zip). Unzip to `train.csv` before running the notebook. |

---

## Author

**Aakash Maskara**  
*M.S. Robotics & Autonomy, Drexel University*  
Data Science | Machine Learning | Quantitative Finance

[LinkedIn](https://linkedin.com/in/aakashmaskara) • [GitHub](https://github.com/AakashMaskara)
