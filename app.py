import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature columns
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Employee Leave Prediction", layout="centered")

st.title("💼 Employee Attrition Predictor")
st.write("Enter employee details to predict whether they will leave or not.")

# User Inputs
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["No", "Yes"])
experience = st.number_input("Experience in Current Domain (years)", min_value=0, max_value=30, value=3)

# Convert Inputs to DataFrame
user_input = {
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience,
    "EverBenched": 1 if ever_benched == "Yes" else 0,
    "isFemale": 1 if gender == "Female" else 0,
    "City_Bangalore": 1 if city == "Bangalore" else 0,
    "City_New Delhi": 1 if city == "New Delhi" else 0,
    "City_Pune": 1 if city == "Pune" else 0,
    "Education_Bachelors": 1 if education == "Bachelors" else 0,
    "Education_Masters": 1 if education == "Masters" else 0,
    "Education_PHD": 1 if education == "PHD" else 0
}

user_df = pd.DataFrame([user_input])

# Reindex to match training features
user_df = user_df.reindex(columns=feature_columns, fill_value=0)

# Scale
user_scaled = scaler.transform(user_df)

# Predict
if st.button("🔮 Predict"):
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee is **likely to leave**. (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Employee is **not likely to leave**. (Confidence: {1-prob:.2f})")

# Footer
st.markdown("---")
st.markdown("✨ Made with ❤️ by **Bhanu Joshi**")




