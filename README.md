Logistic Regression Analysis
Overview
This repository contains an in-depth analysis of a logistic regression model. The analysis includes data preprocessing, feature selection, model training, and evaluation using various statistical metrics and visualization techniques.

Table of Contents
Introduction
Data Preprocessing
Handling Missing Values
Encoding Categorical Variables
Feature Scaling
Feature Selection
Variance Inflation Factor (VIF)
Recursive Feature Elimination (RFE)
Model Training
Generalized Linear Model (GLM)
Logistic Regression Model
Hyperparameter Tuning
Model Evaluation
Confusion Matrix
Accuracy, Precision, Recall, and F1 Score
ROC Curve and AUC
Visualizations
Confusion Matrix Heatmap
ROC Curve
Conclusion
Introduction
Logistic regression is a statistical method used to model a binary dependent variable. This repository demonstrates the steps involved in building and evaluating a logistic regression model for binary classification tasks.

Data Preprocessing
Handling Missing Values
Missing values are imputed using appropriate strategies such as mean, median, or mode imputation.

python
Copy code
data.fillna(data.median(), inplace=True)
Encoding Categorical Variables
Categorical variables are encoded into numerical format using techniques like One-Hot Encoding.

python
Copy code
data = pd.get_dummies(data, drop_first=True)
Feature Scaling
Feature scaling is applied to standardize the range of independent variables.

python
Copy code
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
Feature Selection
Variance Inflation Factor (VIF)
VIF is used to detect multicollinearity between independent variables. Features with high VIF values are considered for removal to improve the model's performance.

python
Copy code
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
Recursive Feature Elimination (RFE)
RFE is used to select the most important features by recursively considering smaller sets of features and ranking them by importance.

python
Copy code
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
Model Training
Generalized Linear Model (GLM)
GLM is used to fit a logistic regression model, allowing for more flexibility in model fitting.

python
Copy code
import statsmodels.api as sm

X_const = sm.add_constant(X_rfe)
glm_model = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
print(glm_model.summary())
Logistic Regression Model
The logistic regression model is trained using the selected features after applying RFE and VIF.

python
Copy code
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_rfe, y)
Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV to optimize model performance.

python
Copy code
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
grid_model = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_model.fit(X_rfe, y)
Model Evaluation
Confusion Matrix
A confusion matrix is used to evaluate the performance of the classification model by comparing actual and predicted labels.

python
Copy code
from sklearn.metrics import confusion_matrix

y_pred = log_model.predict(X_rfe)
cm = confusion_matrix(y, y_pred)
Accuracy, Precision, Recall, and F1 Score
These metrics provide a detailed evaluation of the model's performance.

python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
ROC Curve and AUC
The ROC curve and AUC are used to evaluate the model's discriminatory power.

python
Copy code
from sklearn.metrics import roc_curve, auc

y_prob = log_model.predict_proba(X_rfe)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
Visualizations
Confusion Matrix Heatmap
A heatmap is used to visualize the confusion matrix.

python
Copy code
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
ROC Curve
The ROC curve visualizes the trade-off between the true positive rate and false positive rate.

python
Copy code
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
Conclusion
This repository provides a comprehensive guide to building and evaluating a logistic regression model. By following the steps outlined in this analysis, you can develop a robust model for binary classification tasks and assess its performance using various statistical metrics and visualization techniques.
