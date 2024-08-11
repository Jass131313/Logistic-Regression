# Logistic Regression Analysis

## Overview
This repository contains an in-depth analysis of a logistic regression model. The analysis includes data preprocessing, feature selection, model training, and evaluation using various statistical metrics and visualization techniques.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Encoding Categorical Variables](#encoding-categorical-variables)
  - [Feature Scaling](#feature-scaling)
- [Feature Selection](#feature-selection)
  - [Variance Inflation Factor (VIF)](#variance-inflation-factor-vif)
  - [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe)
- [Model Training](#model-training)
  - [Generalized Linear Model (GLM)](#generalized-linear-model-glm)
  - [Logistic Regression Model](#logistic-regression-model)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Accuracy, Precision, Recall, and F1 Score](#accuracy-precision-recall-and-f1-score)
  - [ROC Curve and AUC](#roc-curve-and-auc)
- [Visualizations](#visualizations)
  - [Confusion Matrix Heatmap](#confusion-matrix-heatmap)
  - [ROC Curve](#roc-curve)
- [Conclusion](#conclusion)

## Introduction
Logistic regression is a statistical method used for modeling the probability of a binary outcome. It is widely used in classification problems where the goal is to predict one of two possible outcomes. Unlike linear regression, which predicts continuous values, logistic regression estimates probabilities and classifies data points based on a threshold.

### Applications
- **Binary Classification:** Predicting whether an email is spam or not.
- **Medical Diagnosis:** Predicting whether a patient has a particular disease.

## Data Preprocessing

### Handling Missing Values
Missing values are imputed using appropriate strategies such as mean, median, or mode imputation to ensure the dataset is complete.

python
data.fillna(data.median(), inplace=True)

## Encoding Categorical Variables
Categorical variables are converted into numerical format using techniques like One-Hot Encoding to make them suitable for logistic regression.

data = pd.get_dummies(data, drop_first=True)

## Feature Scaling
Feature scaling standardizes the range of independent variables to improve the performance and convergence of the logistic regression model.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

## Feature Selection
### Variance Inflation Factor (VIF)
VIF is used to detect multicollinearity among independent variables. High VIF values indicate that a variable is highly correlated with other variables, which can be problematic.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

## Recursive Feature Elimination (RFE)
RFE selects the most important features by recursively removing the least significant features and ranking them by importance.

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

## Model Training
### Generalized Linear Model (GLM)
GLM is used to fit a logistic regression model, providing a framework that generalizes linear regression to include binary outcomes.

import statsmodels.api as sm

X_const = sm.add_constant(X_rfe)
glm_model = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
print(glm_model.summary())

## Logistic Regression Model
The logistic regression model is trained on the selected features to predict the probability of the binary outcome.

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_rfe, y)

### Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV to find the best model parameters and enhance model performance.

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
grid_model = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_model.fit(X_rfe, y)
print("Best Parameters:", grid_model.best_params_)

## Model Evaluation
### Confusion Matrix
A confusion matrix compares the actual and predicted labels to evaluate the performance of the classification model.

from sklearn.metrics import confusion_matrix

y_pred = log_model.predict(X_rfe)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

## Accuracy, Precision, Recall, and F1 Score
These metrics provide a comprehensive assessment of the model’s classification performance.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

## ROC Curve and AUC
The ROC curve and AUC (Area Under the Curve) evaluate the model's ability to distinguish between classes.

from sklearn.metrics import roc_curve, auc

y_prob = log_model.predict_proba(X_rfe)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

print(f'AUC: {roc_auc}')

## Visualizations
### Confusion Matrix Heatmap
A heatmap visualizes the confusion matrix, helping to understand the performance of the classification model.

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

## ROC Curve
The ROC curve shows the trade-off between the true positive rate and the false positive rate.

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
Conclusion
This repository provides a comprehensive guide to building and evaluating a logistic regression model. By following the steps outlined, you can develop a robust model for binary classification tasks and assess its performance using various statistical metrics and visualization techniques.
