
# Postoperative Delirium Risk Modeling
Machine learning models for predicting postoperative delirium from perioperative EHR data, including baseline models, domain-structured ensembles, interpretability, calibration, and decision curve analysis.


This repository contains the modeling and analysis code for a retrospective study
developing and evaluating machine learning models to predict postoperative delirium
using perioperative electronic health record (EHR) data.

The workflow includes baseline global models, tuned gradient boosting models,
domain-specific models aligned with clinical reasoning, and a stacked meta-learner
with calibration, interpretability, and decision curve analysis.

## Repository Contents

- `PODelirium_Baselines.ipynb`  
  Baseline model development using all predictors, including:
  - Logistic regression (L2 and elastic net)
  - Random forest
  - Histogram-based gradient boosting  
  Models are evaluated using patient-grouped cross-validation.

- `PODelirium_DomainGrouping_LGBM_Metalearner_Calibration_DCA.ipynb`  
  Domain-structured modeling pipeline including:
  - Domain-specific LightGBM models
  - Stacked meta-learner using domain-level predictions
  - Probability calibration
  - SHAP-based interpretability
  - Decision curve analysis

## Data Availability

Due to patient privacy and institutional restrictions, the underlying EHR data
are not publicly available. The code is provided to support transparency,
reproducibility, and methodological review.

## Software

Analyses were conducted in Python 3.9 using:
- scikit-learn
- LightGBM
- XGBoost
- SHAP
- pandas, numpy, matplotlib

## Reproducibility

Random seeds are fixed throughout the workflow. Model specifications, preprocessing
pipelines, and intermediate artifacts are saved to enable reproducibility of results
given access to the source data.

## License

This project is released under the MIT License.
