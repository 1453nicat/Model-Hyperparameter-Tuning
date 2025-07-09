# Model-Hyperparameter-Tuning

_**Fraud Detection with XGBoost and Optuna**_

- Project Overview
This project implements a machine learning model for fraud detection using the creditcard.csv dataset, which contains anonymized credit card transactions with a highly imbalanced target (0.17% fraud cases). The goal was to optimize the hyperparameters of an XGBoost classifier to improve performance, as outlined in the task:

- Model Selection: XGBoost was chosen for its effectiveness with imbalanced data.
- Hyperparameter Tuning: Optuna was used for Bayesian optimization to find optimal hyperparameters.
- Evaluation: Model performance was assessed using 5-fold cross-validation with metrics: precision, recall, F1 score, and AUC-ROC.
- Environment: Developed in Google Colab using Scikit-Learn and Optuna.
- Documentation: The tuning process and results are detailed in the code and this README.

The project achieves strong test set performance: Precision (0.86), Recall (0.85), F1 Score (0.86), and AUC-ROC (0.98).

_Dataset_
The creditcard.csv dataset includes:

- Features: 28 PCA-transformed features (V1–V28), "Time" (seconds since first transaction), and "Amount" (transaction amount).
- Target: "Class" (0 = non-fraud, 1 = fraud).
- Size: 284,807 samples, with 492 fraud cases (0.17%).

_Preprocessing:_
Scaled features, and split data with stratification to preserve the 0.17% fraud ratio.

_Class Imbalance Handling:_
Used scale_positive_weight (~577, ratio of non-frauds to frauds) in XGBoost.

_Hyperparameter Tuning:_
Optuna performed 50 trials, optimizing F1 score via 5-fold cross-validation. In addition, tuned parameters are; learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators, reg_alpha, reg_lambda.

_Evaluation:_
Metrics: Precision, Recall, F1 Score, AUC-ROC.
Visualizations: Confusion matrix and ROC curve.

_Results:_
Precision: 0.86 (86% of predicted frauds were correct).
Recall: 0.85 (85% of actual frauds were detected).
F1 Score: 0.86 (balanced precision and recall).
AUC-ROC: 0.98 (excellent discrimination between fraud and non-fraud).


_Summary:_
Optuna optimized F1 score over 50 trials.
Cross-validation ensured robust performance estimates.
The confusion matrix visualized true positives, false positives, etc.
The ROC curve confirmed the model’s strong ranking ability.

License
This project is licensed under the MIT License.
Acknowledgments

Dataset: Kaggle Credit Card Fraud Detection: https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv
Tools: Scikit-Learn, Optuna, XGBoost, Google Colab
