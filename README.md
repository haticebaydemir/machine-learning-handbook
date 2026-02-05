# Machine Learning Master Handbook

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/haticebaydemir)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive machine learning handbook covering supervised learning, unsupervised learning, ensemble methods, and production deployment. Built with real datasets and production-ready code.

---

## Overview

This repository contains 6 Jupyter notebooks that systematically cover machine learning from fundamentals to production deployment. Each notebook builds on previous concepts and uses real-world datasets.

**Time commitment:** ~25-35 hours total

---

## Repository Structure

```
machine-learning-handbook/
├── notebooks/
│   ├── 00-setup-and-standards.ipynb
│   ├── 01-ml-foundations.ipynb
│   ├── 02-data-mastery-eda-and-preprocessing.ipynb
│   ├── 03-regression-models.ipynb
│   ├── 04-classification-models.ipynb
│   └── 05-advanced-ml-unsupervised-and-production.ipynb
├── utils/
│   └── utils.py
└── README.md
```

---

## Notebooks

### Notebook 0: Setup & Standards
**Status:** Complete | **Time:** 30 minutes

Environment configuration, random seed management, and utility functions.

**Topics:**
- Reproducibility standards
- Random state locking
- Plotting configurations
- Utility function library

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/00-setup-and-standards)

---

### Notebook 1: ML Foundations  
**Status:** Complete | **Time:** 2-3 hours

Core machine learning concepts and theoretical foundations.

**Topics:**
- Supervised vs unsupervised learning
- Training vs inference
- Loss functions
- Bias-variance tradeoff
- Overfitting and underfitting
- Curse of dimensionality
- No Free Lunch theorem

**Datasets:** Synthetic (sklearn)

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/01-ml-foundations)

---

### Notebook 2: Data Mastery - EDA and Preprocessing
**Status:** Complete | **Time:** 3-4 hours

Data preprocessing pipeline from raw data to model-ready format.

**Topics:**
- Exploratory Data Analysis (univariate, bivariate, multivariate)
- Missing value handling (MCAR/MAR/MNAR)
- Outlier detection and treatment
- Categorical encoding (label, one-hot, target, frequency)
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Feature engineering
- Data leakage prevention

**Datasets:** House Prices, Credit Default

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/02-data-mastery-eda-and-preprocessing)

---

### Notebook 3: Regression Models
**Status:** Complete | **Time:** 4-6 hours

Comprehensive coverage of regression algorithms with implementation and comparison.

**Topics:**
- Linear Regression (simple and multiple)
- Regularization: Ridge (L2), Lasso (L1), ElasticNet
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Model evaluation (RMSE, MAE, R², residual analysis)
- Hyperparameter tuning (Grid Search, Random Search)
- Feature importance analysis

**Datasets:** House Prices

**Results:** 12 models compared. Best: XGBoost (RMSE: $27,230, R²: 0.9033)

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/03-regression-models)

---

### Notebook 4: Classification Models
**Status:** Complete | **Time:** 5-7 hours

Classification algorithms with focus on evaluation metrics and class imbalance.

**Topics:**
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machines (Linear, RBF, Polynomial kernels)
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrices
- Class imbalance techniques (SMOTE, class weights)
- Hyperparameter tuning

**Datasets:** Titanic

**Results:** 11 algorithms compared. Best: Random Forest (F1: 0.8030)

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/04-classification-models)

---

### Notebook 5: Advanced ML - Unsupervised Learning, Ensembles & Production
**Status:** Complete | **Time:** 5-7 hours

Advanced topics including unsupervised learning, ensemble methods, interpretability, and production deployment.

**Topics:**

**Unsupervised Learning:**
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA (Principal Component Analysis)

**Ensemble Methods:**
- Voting (Hard and Soft)
- Stacking
- Bagging
- Boosting comparison

**Model Interpretability:**
- Feature importance (tree-based)
- Permutation importance
- Partial Dependence Plots (PDP)
- SHAP values

**Production ML:**
- Model serialization
- Deployment code
- Data drift detection
- Performance monitoring
- A/B testing framework

**Datasets:** Mall Customers, House Prices, Titanic

[View on Kaggle →](https://www.kaggle.com/code/haticebaydemir/05-advanced-ml-unsupervised-and-production)

---

## Algorithms Covered

**Regression (12):**
Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting Regressor, Stacking Regressor, Bagging Regressor

**Classification (11):**
Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting Classifier, Stacking Classifier, Bagging Classifier

**Unsupervised (4):**
K-Means, Hierarchical Clustering, DBSCAN, PCA

---

## Prerequisites

- Python 3.10+
- Basic understanding of Python (NumPy, Pandas)
- Kaggle account (for running notebooks)

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm catboost
pip install imbalanced-learn shap
```

---

## Usage

### Option 1: Run on Kaggle (Recommended)

1. Click notebook link above
2. Fork the notebook
3. Add `ml-handbook-utils` dataset
4. Run cells sequentially

### Option 2: Run Locally

```bash
git clone https://github.com/haticebaydemir/ml-master-handbook.git
cd ml-master-handbook
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Progress Tracker

| # | Notebook | Status | Topics | Time |
|---|----------|--------|--------|------|
| 0 | Setup & Standards | ✅ Complete | Environment, utilities | 30 min |
| 1 | ML Foundations | ✅ Complete | Core concepts, bias-variance | 2-3 hours |
| 2 | Data Mastery | ✅ Complete | EDA, preprocessing, feature engineering | 3-4 hours |
| 3 | Regression Models | ✅ Complete | 12 regression algorithms | 4-6 hours |
| 4 | Classification Models | ✅ Complete | 11 classification algorithms | 5-7 hours |
| 5 | Advanced ML & Production | ✅ Complete | Unsupervised, ensembles, interpretability, deployment | 5-7 hours |

**Total:** 6/6 notebooks complete | ~25-35 hours

---

## Key Features

- **Systematic progression:** Each notebook builds on previous concepts
- **Real datasets:** House Prices, Titanic, Mall Customers
- **Production-ready code:** Includes deployment, monitoring, and drift detection
- **Comprehensive coverage:** 30+ algorithms, 100+ topics
- **Reproducible:** Fixed random seeds, standardized structure
- **Well-documented:** Theory + implementation + evaluation

---

## Updates

| Date | Update |
|------|--------|
| Feb 5, 2026 | Completed Notebook 5: Advanced ML & Production (14 sections, 70 code cells, unsupervised learning, ensembles, interpretability, production ML) |
| Feb 3, 2026 | Completed Notebook 4: Classification Models (14 sections, 60+ demonstrations, 11 algorithms) |
| Jan 31, 2026 | Completed Notebook 3: Regression Models (13 sections, 50+ demonstrations, 12 models) |
| Jan 30, 2026 | Completed Notebook 2: Data Mastery (10 sections, 35+ demonstrations) |
| Jan 29, 2026 | Completed Notebook 1: ML Foundations (11 sections, 15 visualizations) |
| Jan 28, 2026 | Completed Notebook 0: Setup & Standards |
| Jan 25, 2026 | Repository initialized |

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

- **Kaggle:** [haticebaydemir](https://www.kaggle.com/haticebaydemir)
- **LinkedIn:** [haticebaydemir](https://www.linkedin.com/in/haticebaydemir/)
- **GitHub:** [haticebaydemir](https://github.com/haticebaydemir)

---

**Last Updated:** February 5, 2026 | **Status:** Complete (6/6 notebooks)
