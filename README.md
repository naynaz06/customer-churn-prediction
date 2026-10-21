# Customer Churn Prediction

## Project Overview
This project predicts whether a telecom customer is likely to leave the company (churn) using machine learning classification algorithms. It also includes an interactive Streamlit web application that allows users to enter customer details and obtain a churn prediction.

## Objectives
- Analyze customer data to identify churn patterns.
- Perform data cleaning, preprocessing, and exploratory data analysis (EDA).
- Train and compare multiple machine learning classification models.
- Improve model performance using hyperparameter tuning.
- Deploy the final model through a Streamlit web application.

## Dataset
The project uses the Telco Customer Churn dataset, which contains customer demographics, account information, subscribed services, payment methods, and churn status.

- Dataset size: 7,043 customer records
- Target variable: `Churn`
- Target classes: `Yes` (churn) and `No` (no churn)

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Workflow
1. Data cleaning and missing-value handling
2. Exploratory data analysis and visualization
3. Categorical encoding and feature engineering
4. Train-test split and feature scaling
5. Model training and evaluation
6. Hyperparameter tuning using GridSearchCV
7. Feature importance analysis
8. Model saving and Streamlit deployment

## Models Evaluated
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Machine (SVM)

## Results
The tuned SVM model was selected for deployment because it achieved the highest churn recall among the evaluated models.

- Accuracy: 74.59%
- Churn Precision: 51.42%
- Churn Recall: 77.54%
- Churn F1-score: 61.83%

The model prioritizes identifying customers who may churn, although this results in some customers being incorrectly flagged as potential churners.

## Streamlit Application
The application accepts customer information and predicts whether the customer is likely to churn or stay.

To run the application locally:

1. Clone or download this repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Run:
   `streamlit run app.py`

Ensure the saved model, scaler, and feature-column files are present in the project directory.

## Project Structure
- `customer_churn.ipynb` — Data analysis, model training, evaluation, and feature importance
- `app.py` — Streamlit application
- `final_svm_model.pkl` — Saved SVM model
- `scaler.pkl` — Saved feature scaler
- `feature_columns.pkl` — Training feature names and order
- `requirements.txt` — Python dependencies

## Key Learning Outcomes
- Data preprocessing and feature engineering
- Classification model evaluation
- Handling class imbalance using class weights
- Hyperparameter tuning
- Model serialization and deployment with Streamlit

## Application Preview

![Customer Churn Prediction Dashboard](streamlit_app.png)

