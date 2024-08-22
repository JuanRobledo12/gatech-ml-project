# Diabetes Classification Project

## Overview

This project focuses on training and assessing various machine learning models to determine whether a woman has diabetes based on a range of clinical characteristics. The analysis utilizes the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

The dataset was originally provided by the National Institute of Diabetes and Digestive and Kidney Diseases. It was selectively gathered from a larger database, focusing on female patients who are at least 21 years old and of Pima Indian descent.

The dataset includes multiple medical predictor variables along with a single target variable, "Outcome." The predictor variables cover aspects such as the number of pregnancies, BMI, insulin levels, and age, among others.

## File Structure

- `eda.ipynb`: Exploratory data analysis and data cleaning notebook.
- `experiment_1.ipynb`: Model training and evaluation with the elimination of missing values in the original dataset.
- `experiment_2.ipynb`: Model training and evaluation with the imputation of missing values in the original dataset.
- `dataset`: Directory containing the original dataset and its cleaned versions.
- `experiments_results`: Directory containing the results of the experiments.

## Project Workflow

1. **Data Analysis and Cleaning**: The dataset was analyzed to ensure all columns had the correct data types. Missing values, represented by zeros in certain metrics, were identified as incomplete records and were either imputed or removed. Outliers were also detected and eliminated.
2. **Feature Engineering**: Relationships and associations between variables were explored using correlation matrices, scatterplots, side-by-side boxplots, and overlapping histograms. Features with a strong association with the target variable were identified using t-tests and correlation analysis.
3. **Data Preprocessing**: As the dataset was imbalanced, preprocessing techniques such as SMOTE for oversampling were incorporated into the pipeline. Stratified splits were also ensured during data partitioning.
4. **Model Selection**: Logistic regression was used as the baseline model, with k-NN and Random Forest models trained to achieve better performance.
5. **Hyperparameter Tuning**: Grid search was employed to tune the models and identify the best estimator.
6. **Evaluation**: Models were evaluated using metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

## Results and Conclusion

### Results

I trained and evaluated the models on two different versions of the dataset: `diabetes_v1.csv`, which had all missing values eliminated, and `diabetes_v2.csv`, where missing values were imputed.

#### `diabetes_v1.csv` Results

The `diabetes_v1.csv` dataset, with 321 entries, yielded the following performance:

- **Random Forest (RobustScaler, all features):** 
  - Accuracy: 81.5%
  - Precision: 66.7%
  - Recall: 66.7%
  - F1-Score: 66.7%
  - ROC AUC: 75.9%

- **Random Forest (StandardScaler, without BP and DPF):**
  - Accuracy: 80.0%
  - Precision: 63.2%
  - Recall: 66.7%
  - F1-Score: 64.9%
  - ROC AUC: 77.1%

- **Logistic Regression (RobustScaler, all features):**
  - Accuracy: 78.5%
  - Precision: 60.0%
  - Recall: 66.7%
  - F1-Score: 63.2%
  - ROC AUC: 79.8%

Overall, models trained on the `diabetes_v1.csv` dataset had moderate performance. The Random Forest models, particularly those with all features, performed the best in terms of accuracy, with one achieving an accuracy of 81.5%. However, the F1-scores and ROC AUC scores suggest room for improvement in balancing precision and recall.

#### `diabetes_v2.csv` Results

The `diabetes_v2.csv` dataset, with 665 entries, yielded the following performance:

- **Random Forest (StandardScaler, all features):**
  - Accuracy: 80.5%
  - Precision: 66.7%
  - Recall: 76.2%
  - F1-Score: 71.1%
  - ROC AUC: 88.2%

- **K-Nearest Neighbors (StandardScaler, all features):**
  - Accuracy: 75.9%
  - Precision: 58.1%
  - Recall: 85.7%
  - F1-Score: 69.2%
  - ROC AUC: 85.7%

- **Logistic Regression (RobustScaler, all features):**
  - Accuracy: 76.7%
  - Precision: 60.0%
  - Recall: 78.6%
  - F1-Score: 68.0%
  - ROC AUC: 84.1%

The models trained on the `diabetes_v2.csv` dataset generally performed better than those trained on `diabetes_v1.csv`. The Random Forest model with all features scaled by StandardScaler achieved the best performance, with an accuracy of 80.5% and a high ROC AUC score of 88.2%. The improved recall and ROC AUC scores across models suggest that imputation of missing values helped in retaining more data, leading to better generalization.

### Conclusion

The results highlight the importance of handling missing data appropriately. The models trained on the `diabetes_v2.csv` dataset, where missing values were imputed, consistently outperformed those trained on the `diabetes_v1.csv` dataset with missing values removed. This suggests that imputing missing data allowed the models to leverage more information, improving their predictive power.

Among the models, the Random Forest classifier with all features and StandardScaler preprocessing emerged as the best-performing model. It achieved the highest accuracy (80.5%) and ROC AUC score (88.2%), indicating a good balance between precision and recall. This model is likely the most robust choice for predicting diabetes based on this dataset.

The K-Nearest Neighbors and Logistic Regression models also showed competitive performance, particularly in recall and ROC AUC, making them viable alternatives depending on the specific use case. The higher recall in these models could be beneficial in scenarios where identifying as many true positive cases as possible is crucial.

Overall, this project demonstrates the value of thorough data preprocessing and the importance of experimenting with multiple models to identify the best-performing approach.