# credit_score_prediction
# Credit Score Prediction Using Machine Learning

This project is focused on predicting credit default risk using machine learning models. It uses a dataset (`cs-training.csv`) with customer financial and credit information to classify whether a person is likely to default in the next two years.

## 📁 Dataset

The dataset includes features such as:

- SeriousDlqin2yrs (target variable)
- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

## ⚙️ Preprocessing

- Dropped the `Unnamed: 0` column.
- Handled missing values using median imputation for `MonthlyIncome` and `NumberOfDependents`.
- Dropped any remaining null values.
- Visualized correlations using a heatmap.

## 🧪 Models Used

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- XGBoost

Each model's performance was evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report

## 🧪 Hyperparameter Tuning (XGBoost)

Used `GridSearchCV` to find the best hyperparameters for XGBoost:

- `n_estimators`: [100, 200]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.01, 0.1, 0.2]

## 🧪 Model Evaluation

Final evaluation of the best XGBoost model included:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC Score
- ROC Curve Plot

## ⚖️ Handling Imbalanced Data (SMOTE)

Used **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance. Models were retrained on resampled data using:

- Random Forest
- XGBoost (with best hyperparameters)

Evaluation metrics were computed post-SMOTE to observe performance improvements.

## 📊 Results Summary

After applying SMOTE and tuning hyperparameters:

- XGBoost provided the best overall performance.
- ROC-AUC Score increased significantly.
- Accuracy and recall also improved compared to baseline models.

## 📦 Dependencies

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`

## 🔧 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
