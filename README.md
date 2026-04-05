# Perioperative Outcome Prediction Framework


## Overview

This repository contains the analysis code for a study developing a domain-structured ensemble prediction model for postoperative delirium in adults undergoing non-cardiac, non-obstetric surgery. The framework organises 74 perioperative predictors into three clinician-defined domains (patient-related, surgery-related, and anaesthetic-related), trains domain-specific LightGBM base learners, and combines their predictions through a logistic regression meta-learner. The modular architecture is designed to be extensible to diverse perioperative outcomes without modification to the modelling pipeline.

## Repository Structure

```
├── Notebooks/
│   ├── 01_baseline_and_tuned_models.ipynb
│   ├── 02_domain_ensemble_metalearner.ipynb
│   ├── 03_additional_analyses.ipynb
│   └── 04_sensitivity_analyses.ipynb
└── README.md
```

## Notebooks

### 01 - Baseline and Tuned Models

- Data loading and preprocessing
- Feature engineering (ASA class extraction, emergency flag creation)
- Correlation filtering (phik coefficient, phiK > 0.8)
- Missing data handling and feature schema definition
- Preprocessing pipeline construction (unfitted, leak-safe)
- Level 1: Baseline models (logistic regression, elastic net, random forest, histogram gradient boosting)
- Level 2: Tuned XGBoost with nested cross-validation and isotonic calibration

### 02 - Domain Ensemble and Meta-learner

- Domain definition (patient-related, surgery-related, anaesthetic-related)
- Level 3: Domain-specific LightGBM base learners with out-of-fold predictions
- Level 4: Logistic regression meta-learner (L2 regularisation)
- SHAP interpretability analysis (TreeExplainer, beeswarm plots)
- Model calibration (intercept, slope, calibration curves)
- Prevalence-adjusted decision curve analysis
- ROC and precision-recall evaluation

### 03 - Additional Analyses

- **Temporal validation**: Train on 2010--2017, test on 2018--2021, using OOF domain predictions within the training set to train the meta-learner (preventing calibration leakage)
- **Domain ablation study**: Systematic evaluation of all seven domain configurations (full three-domain, three two-domain drop-one, three single-domain) via grouped five-fold cross-validation
- **Subgroup and fairness analysis**: Performance stratified by age group, sex, race, and surgical specialty with fairness gap quantification
- **TRIPOD+AI adherence checklist**: Completed per Collins et al. (BMJ 2024)

### 04 - Sensitivity Analyses

- **PhiK correlation threshold sensitivity**: Qualitative assessment of feature filtering robustness
- **Meta-learner architecture sensitivity**: Comparison of logistic regression, unpenalised logistic regression, random forest, and gradient boosting as meta-learners
- **Case-control ratio sensitivity**: Performance under simulated 1:1, 1:2, and 1:3 case-to-control ratios
- **Cross-validation scheme sensitivity**: 5-fold vs. 10-fold grouped CV comparison
- **Temporal cutoff sensitivity**: Performance across 2016, 2017, 2018, and 2019 train/test boundaries using OOF-corrected meta-learner training
- **Bootstrap confidence intervals**: 1,000-resample 95% CIs for primary metrics
- **Comprehensive sensitivity summary**: Tornado plot of AUC deviations from the primary model

## Modelling Approach

All models were trained and evaluated using patient-level grouped five-fold cross-validation to prevent data leakage from patients with multiple encounters. The multistage framework progresses from global baseline models through hyperparameter-tuned comparators to domain-structured stacking, enabling direct comparison across modelling strategies and quantification of domain-level contributions.

Temporal validation uses out-of-fold domain predictions within the training set (via GroupKFold) to fit the meta-learner, followed by refitting domain models on the full training set for test-set inference. This prevents calibration leakage that would otherwise arise from training the meta-learner on in-sample domain predictions.

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

## Citation

If you use this framework, please cite:
