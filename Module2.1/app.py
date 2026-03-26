import streamlit as st
import joblib
import pandas as pd

# Load models
reg_model = joblib.load("reg_pipeline.pkl")
dt_model = joblib.load("dt_pipeline.pkl")
knn_model = joblib.load("knn_pipeline.pkl")

st.title(" Student Performance Predictor")

# Input
hours = st.number_input("Study Hours", min_value=0, max_value=24, value=7)
sleep = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)

new_student = [[hours, sleep, attendance]]

# Predict Button
if st.button("Predict"):
    # Regression
    marks = reg_model.predict(new_student)[0]
    
    # Classification
    result_dt = dt_model.predict(new_student)[0]
    result_knn = knn_model.predict(new_student)[0]
    
    st.subheader(" Prediction Results")
    st.write(f"Predicted Marks: {marks:.2f}")
    st.write(f"Decision Tree Prediction (Pass=1, Fail=0): {result_dt}")
    st.write(f"KNN Prediction (Pass=1, Fail=0): {result_knn}")