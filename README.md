# Diabetes Prediction Project

## Overview

This project aims to predict diabetes status using various machine learning models on the Diabetes Health Indicators Dataset from Kaggle. The dataset consists of over 70,000 rows and 21 features. Several models, including logistic regression, random forest, XGBoost, and gradient boosting, were tested. Feature engineering and hyperparameter tuning were also performed to optimize model performance.

## Dataset

The dataset used for this project is the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from Kaggle.

## Requirements

To set up the environment and install the required packages, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JuanRobledo12/gatech-ml-project.git
   cd gatech-ml-project
   ```

2. **Create a virtual environment** (optional but recommended, you can also use conda):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

The project is contained within a Jupyter Notebook named `main.ipynb`. To run the notebook:

1. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Open `main.ipynb`**:
   - In the Jupyter Notebook interface, navigate to `main.ipynb` and open it.
   - Execute the cells in the notebook to run the project and see the results.

## File Structure

- `main.ipynb`: The main Jupyter Notebook containing the project code and analysis.
- `requirements.txt`: The file listing all the required packages and their versions.

## Project Workflow

1. **Data Preprocessing**: Load and preprocess the dataset, including handling missing values and feature scaling.
2. **Model Training**: Train various machine learning models, including logistic regression, random forest, XGBoost, and gradient boosting.
3. **Feature Engineering**: Perform feature engineering to enhance model performance.
4. **Hyperparameter Tuning**: Optimize the models using hyperparameter tuning techniques.
5. **Evaluation**: Evaluate the models using metrics like precision, recall, f1-score, and confusion matrix.

## Results and Conclusion

After extensive testing and tuning, the XGBoost model provided the best performance with a precision of 0.73 and a recall of 0.80 for predicting diabetes. The project demonstrated the effectiveness of ensemble methods and the importance of feature engineering and hyperparameter tuning in improving model accuracy.

## References

- [Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
