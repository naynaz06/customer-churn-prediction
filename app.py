import streamlit as st
import joblib
model = joblib.load("final_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
st.title("Customer Churn Prediction")
st.write(
    "Enter Customer details to predict whether the customer "
    "is likely to churn"
)

st.header("Customer Information")

# Customer demographics
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

# Account information
tenure = st.number_input(
    "Tenure (months)", min_value=0, max_value=72, value=12
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)

total_charges = st.number_input(
    "Total Charges", min_value=0.0, value=840.0
)

# Phone services
phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

# Internet services
internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

# Payment information
payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# Convert binary categorical inputs into numeric values

gender_encoded = 0 if gender == "Female" else 1

partner_encoded = 1 if partner == "Yes" else 0
dependents_encoded = 1 if dependents == "Yes" else 0
phone_service_encoded = 1 if phone_service == "Yes" else 0
paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0

# SeniorCitizen is already represented as 0 or 1
senior_citizen_encoded = senior_citizen

import pandas as pd

# Create a single-row DataFrame from categorical inputs
input_data = pd.DataFrame([{
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaymentMethod": payment_method
}])

# Apply one-hot encoding
input_encoded = pd.get_dummies(input_data, dtype=int)

st.write("Categorical encoding completed.")

# Create a DataFrame for binary and numerical features
base_data = pd.DataFrame([{
    "gender": gender_encoded,
    "SeniorCitizen": senior_citizen_encoded,
    "Partner": partner_encoded,
    "Dependents": dependents_encoded,
    "tenure": tenure,
    "PhoneService": phone_service_encoded,
    "PaperlessBilling": paperless_billing_encoded,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# Combine binary/numerical features with one-hot encoded features
input_df = pd.concat([base_data, input_encoded], axis=1)

# Match the exact 40 features and their training order
input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)

st.write("Total features prepared:", len(input_df.columns))


# Scale numerical features using the saved scaler
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

input_df[numerical_cols] = scaler.transform(
    input_df[numerical_cols]
)

st.write("Customer data prepared successfully!")


# Predict customer churn
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")
