# Perioperative Delirium Prediction Framework

An interpretable, domain-based stacked ensemble model for predicting postoperative delirium (POD) using comprehensive perioperative electronic health record (EHR) data.

## Overview

This repository contains the analysis code for a study developing a multistage prediction model for postoperative delirium in adults undergoing non-cardiac, non-obstetric surgery. The framework organises 74 perioperative predictors into three clinician-defined domains (patient-related, surgery-related, and anaesthetic-related), trains domain-specific LightGBM base learners, and combines their predictions through a logistic regression meta-learner.

## Repository Structure

```
├── Notebooks/
│   ├── 01_baseline_and_tuned_models.ipynb
│   └── 02_domain_ensemble_metalearner.ipynb
├── artifacts/
│   └── (generated outputs: model files, predictions, SHAP values, plots)
└── README.md
```

## Notebooks

### 01 — Baseline and Tuned Models

- Data loading and preprocessing
- Feature engineering (ASA class extraction, emergency flag creation)
- Correlation filtering (phik coefficient, φK > 0.8)
- Missing data handling and feature schema definition
- Preprocessing pipeline construction (unfitted, leak-safe)
- Level 1: Baseline models (logistic regression, elastic net, random forest, histogram gradient boosting)
- Level 2: Tuned XGBoost with nested cross-validation and isotonic calibration

### 02 — Domain Ensemble and Meta-learner

- Domain definition (patient-related, surgery-related, anaesthetic-related)
- Level 3: Domain-specific LightGBM base learners with out-of-fold predictions
- Level 4: Logistic regression meta-learner (L2 regularisation)
- SHAP interpretability analysis (TreeExplainer, beeswarm plots)
- Model calibration (intercept, slope, calibration curves)
- Prevalence-adjusted decision curve analysis
- ROC and precision-recall evaluation

## Modelling Approach

All models were trained and evaluated using patient-level grouped five-fold cross-validation to prevent data leakage from patients with multiple encounters. The multistage framework progresses from global baseline models through hyperparameter-tuned comparators to domain-structured stacking, enabling direct comparison across modelling strategies and quantification of domain-level contributions.

## Requirements

- Python 3.9+
- scikit-learn
- LightGBM
- XGBoost
- SHAP
- pandas
- numpy
- statsmodels
- matplotlib
- phik

Install dependencies:

```bash
pip install scikit-learn lightgbm xgboost shap pandas numpy statsmodels matplotlib phik
```

## Data Availability

The study used de-identified EHR data from the Indiana Network for Patient Care (INPC). Due to data use agreements and patient privacy protections, the raw data cannot be shared publicly.

## Licence

This project is provided for academic and research purposes.
