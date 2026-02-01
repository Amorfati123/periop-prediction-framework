# Domain-Structured Ensemble Framework for Perioperative Outcome Prediction


Official implementation of the domain-structured ensemble framework for perioperative risk prediction, as described in:

> **A Domain-Structured Ensemble Framework for Perioperative Outcome Prediction Using Electronic Health Record Data**  
> Shukla S, Barboi C  
> IEEE ICHI 2026

## Overview

This framework organizes perioperative predictors into three clinically motivated domains and integrates domain-level predictions through stacked ensemble learning:

- **Patient-related domain** (n=31): Demographics, comorbidities, baseline medications
- **Surgery-related domain** (n=12): Procedure duration, emergency status, surgical specialty
- **Anesthetics-related domain** (n=31): Hemodynamics, anesthetic agents, vasoactive medications

## Key Features

- **Modular architecture**: Domain models can be retrained independently
- **Outcome-agnostic design**: Substitute outcome definitions without pipeline modification
- **Interpretable predictions**: SHAP-based feature attribution within clinically coherent domains
- **Calibration-aware evaluation**: Prevalence adjustment and decision curve analysis for case-control designs

## Repository Structure
```
├── 01_baseline_and_tuned_models.ipynb    # Baseline models (LR, RF, HistGB) and tuned XGBoost
├── 02_domain_ensemble_metalearner.ipynb  # Domain-specific LightGBM + stacked meta-learner
├── LICENSE                                # MIT License
└── README.md
```

## Requirements
```
python>=3.9
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
shap==0.42.1
statsmodels==0.14.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Usage

### 1. Baseline Models
```python
# See 01_baseline_and_tuned_models.ipynb for:
# - Logistic regression (L2, Elastic Net)
# - Random Forest
# - Histogram Gradient Boosting
# - Tuned XGBoost with nested CV
```

### 2. Domain-Structured Ensemble
```python
# See 02_domain_ensemble_metalearner.ipynb for:
# - Domain-specific LightGBM models
# - Out-of-fold probability generation
# - Stacked logistic regression meta-learner
# - SHAP interpretability analysis
# - Calibration assessment and DCA
```

## Results Summary

| Model | AUROC | PR-AUC | Brier Score |
|-------|-------|--------|-------------|
| Best Baseline (HistGB) | 0.849 | 0.832 | 0.158 |
| **Stacked Meta-Learner** | **0.899** | **0.881** | **0.126** |

## Data Availability

 **Patient data cannot be shared** due to IRB restrictions (Indiana University Protocol #13577).

The code demonstrates the complete framework architecture. To apply this framework to your own data:

1. Prepare your dataset with predictors organized into three domains
2. Define case/control criteria for your outcome of interest
3. Follow the notebook pipelines for model training and evaluation

## Extending to Other Outcomes

The framework is designed for extensibility. To adapt for different perioperative outcomes (e.g., AKI, respiratory failure, SSI):
```python
# Modify only the outcome labeling function:
def define_outcome(df):
    # Your outcome-specific logic here
    cases = df[your_case_criteria]
    controls = df[your_control_criteria]
    return cases, controls

# Pipeline, cross-validation, and evaluation remain unchanged
```

## Citation

If you use this framework in your research, please cite:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the framework implementation, please open an issue or contact:
- Shikhar Shukla - [shikshuk@iu.edu] or [shikharshuklams@gmail.com]


---
