# Interpretable Multimodal Learning of Postoperative Neuroinflammatory States from Biomarkers, Diffusion MRI, and Cognitive Dat

This repository contains the **regression-based machine learning pipeline** for continuous glioma grading using radiomic features from multiple MRI modalities.

> **Beyond Binary Classification: A Pilot Study of Continuous Glioma Grading Using T1-Weighted and Diffusion MRI Radiomics**

The objective is to evaluate radiomic features for **continuous glioma grading** using regression models with interpretability and visualization.

---

## Pipeline Overview

![Pipeline Overview](https://github.com/pamelaFranco/continuous-glioma-grading/blob/main/Graphical_Abstract.png)

---

## Repository Contents

- `NormRadiomic_alldata.py` — Python script for radiomics extraction
- `RadiomicsVisualization.py` — Python script for GLCM texture visualization
- `TumorPatchandRadiomicFEatureImportance.py` — Python script for Tumor Patch and Radiomic Feature Importance visualization (5 best-importance features by LIME)
- `radiomics_with_classes_cleaned.csv` — CSV with extracted radiomic features for all patients and modalities.
- `Regression_StrategyCV.py` — Main machine learning pipeline with:
  - Hierarchical clustering for feature correlation analysis
  - Ridge regression for initial feature importance
  - Sequential Feature Selection (SFS) with Random Forest
  - 15 regression models with hyperparameter tuning
  - 5-fold nested CV repeated 10 times for robust evaluation
  - Model evaluation using MSE, MAE, and R-squared
  - SHAP explainability for model interpretation
  - Comprehensive visualizations

---

## Methods Summary

**Modalities**: T1-weighted, FA, MD, RD, AD  
**Subjects**: 36 glioma patients (LGG vs. HGG, histologically confirmed)  
**MRI Scanner**: GE Signa HDxt 1.5T  

Radiomic features were extracted using [PyRadiomics](https://pyradiomics.readthedocs.io/), including:
- First-order stats
- Shape (T1 only)
- Texture features: GLCM, GLRLM, GLSZM, GLDM, NGTDM

### Feature Processing:
- Hierarchical clustering to identify feature correlations
- Ridge regression for initial feature importance ranking
- Sequential Feature Selection (SFS) with Random Forest
- Pareto analysis for optimal feature subset selection

### Models Evaluated (15 regression algorithms):
- **Tree-based**: Random Forest, XGBoost, CatBoost, Gradient Boosting, Decision Tree, AdaBoost
- **Linear models**: Ridge, Lasso, ElasticNet, Bayesian Ridge, Logistic Regression, Quantile Regressor
- **Other**: SVM (RBF Kernel), K-Neighbors Regressor, MLP Regressor

### Model Selection & Validation:
- **GridSearchCV** with 5-fold stratified cross-validation
- **5-fold stratified cross-validation repeated 10 times** for robust performance estimation
- Hyperparameter optimization for each model
- Performance metrics: MSE, MAE, R-squared with mean ± standard deviation

### Interpretability:
- **SHAP analysis** for global and local feature importance
- Residual plots and regression diagnostics
- Hierarchical clustering visualization
- Feature importance rankings

---

## Reproducibility

- Python 3.9 (Radiomic extraction and visualization)
- Python 3.10 (ML-pipeline)
- All random seeds fixed using `random`, `numpy`, `check_random_state`
- Classifiers use `random_state=42`
- StratifiedKFold CV is reproducible
- All model outputs and metrics exported to CSV

> MRI scans are not publicly available. Contact **Steren Chabert (steren.chabert@uv.cl)** for access or clinical inquiries.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/pamelaFranco/continuous-glioma-grading.git
cd continuous-glioma-grading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python Regression_StrategyCV.py
```

---

## License

MIT License

---

## Acknowledgements

This work is supported by the Agencia Nacional de Investigación y Desarrollo de Chile (ANID), through the FONDECYT Iniciación 2025 N° 11250867 grant and the Competition for Research Regular Projects, year 2023, code LPR23-17, Universidad Tecnológica Metropolitana. PF was funded by ANID FONDECYT de Postdoctorado 2024 N°3240078.