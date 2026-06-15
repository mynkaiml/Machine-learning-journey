# Loan Default Prediction Using Decision Trees, Random Forest, and XGBoost

## Overview

This project focuses on predicting whether a borrower will fully repay a loan or default on it using machine learning classification algorithms.

The project explores:

- Decision Trees
- Random Forests
- XGBoost
- Feature Engineering
- Feature Importance Analysis
- Imbalanced Data Handling
- Model Evaluation using Classification Metrics

The target variable is:

```python
not.fully.paid
0 → Loan Paid Successfully
1 → Loan Defaulted / Not Fully Paid
Dataset Information

The dataset contains borrower information such as:

Feature	Description
credit.policy	Whether borrower meets credit policy criteria
purpose	Purpose of the loan
int.rate	Interest rate
installment	Monthly installment amount
log.annual.inc	Log transformed annual income
dti	Debt-to-income ratio
fico	FICO credit score
days.with.cr.line	Credit history length
revol.bal	Revolving balance
revol.util	Revolving line utilization rate
inq.last.6mths	Number of recent credit inquiries
delinq.2yrs	Delinquencies in the last 2 years
pub.rec	Public derogatory records
Project Workflow
1. Data Exploration

Performed initial analysis to understand:

Dataset shape
Feature distributions
Correlations
Class imbalance
loans.info()
loans.describe()
2. Data Preprocessing
One-Hot Encoding

The purpose column contained categorical values:

credit_card
small_business
debt_consolidation
home_improvement
...

Converted using:

pd.get_dummies()

Example:

purpose_credit_card
purpose_small_business
purpose_home_improvement
3. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
Models Implemented
Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

Concepts learned:

Entropy
Information Gain
Recursive Splitting
Overfitting
Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

Concepts learned:

Bagging
Bootstrap Sampling
Feature Randomization
Ensemble Learning
XGBoost Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

Concepts learned:

Gradient Boosting
Sequential Learning
Residual Error Correction
Regularization
Class Imbalance Problem

The dataset is imbalanced:

Class 0 (Paid)      ≈ 84%
Class 1 (Default)   ≈ 16%

Using accuracy alone can be misleading.

Example:

Predict everyone as Class 0
Accuracy ≈ 84%
Recall = 0%

Therefore, additional metrics were used.

Evaluation Metrics
Accuracy
Overall correctness of predictions
Precision
Of predicted defaulters,
how many actually defaulted?

Formula:

TP / (TP + FP)
Recall
Of actual defaulters,
how many were correctly identified?

Formula:

TP / (TP + FN)
F1 Score

Balances Precision and Recall.

Formula:

2 × (Precision × Recall) / (Precision + Recall)
ROC-AUC Score

Measures the model's ability to separate:

Safe Borrowers
vs
Risky Borrowers

Higher ROC-AUC indicates better ranking performance.

Feature Importance Analysis

Using XGBoost:

xgb.feature_importances_

Most important features:

credit.policy
purpose_small_business
inq.last.6mths
int.rate
fico

Insights:

Credit policy approval is highly predictive.
Recent credit inquiries strongly indicate risk.
Interest rate and credit score are important predictors.
Feature Engineering
Installment-Income Ratio

Created:

loans["installment_income_ratio"]

Formula:

installment / annual_income

Purpose:

Measures repayment burden
relative to income.
Inverse FICO Score

Created:

loans["fico_inverse"]

Formula:

1 / fico

Purpose:

Emphasizes risk associated
with lower credit scores.
Results
Model	Accuracy	Recall	F1 Score	ROC-AUC
Decision Tree	0.64	0.47	0.29	0.575
Random Forest	0.78	0.22	0.24	0.629
XGBoost	0.70	0.39	0.29	0.648
XGBoost + Feature Engineering	0.66	0.49	0.31	0.649
Key Findings
Accuracy alone is not sufficient for imbalanced datasets.
Recall and ROC-AUC provide better insight into model quality.
XGBoost achieved the best overall balance between recall and ranking performance.
Feature engineering provided a small but measurable improvement.
Credit policy, interest rate, recent inquiries, and FICO score are strong indicators of loan default risk.
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
XGBoost
Matplotlib
Seaborn
Jupyter Notebook
Future Improvements
Apply SMOTE for balancing classes.
Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Compare with LightGBM and CatBoost.
Cross-validation for more robust evaluation.
Build an end-to-end prediction pipeline.
Deploy as a web application using Flask or FastAPI.
Learning Outcomes

This project covers the complete supervised learning workflow:

Data Exploration
      ↓
Data Cleaning
      ↓
Feature Encoding
      ↓
Train-Test Split
      ↓
Decision Trees
      ↓
Entropy & Information Gain
      ↓
Random Forests
      ↓
Bagging
      ↓
XGBoost
      ↓
Boosting
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Feature Engineering
      ↓
Performance Analysis

This project serves as a practical implementation of tree-based machine learning algorithms for credit risk and loan default prediction.