import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="centered")
st.title("👥 Customer Segmentation")
st.write("Enter customer details to predict the customer segment (A, B, C or D).")

model = joblib.load("customer_segmentation_model.pkl")

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Ever Married", ["Yes", "No"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    graduated = st.selectbox("Graduated", ["Yes", "No"])
    profession = st.selectbox("Profession",
        ["Artist","Healthcare","Entertainment","Engineer","Doctor",
         "Lawyer","Executive","Marketing","Homemaker"])
    work_experience = st.number_input("Work Experience (years)",
        min_value=0.0, max_value=50.0, value=3.0)
    spending_score = st.selectbox("Spending Score", ["Low", "Average", "High"])
    family_size = st.number_input("Family Size", min_value=1.0, max_value=20.0, value=3.0)
    var_1 = st.selectbox("Var_1",
        ["Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Cat_7"])

    submitted = st.form_submit_button("Predict Segment")

if submitted:
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Ever_Married": married,
        "Age": age,
        "Graduated": graduated,
        "Profession": profession,
        "Work_Experience": work_experience,
        "Spending_Score": spending_score,
        "Family_Size": family_size,
        "Var_1": var_1
    }])
    result = model.predict(input_df)[0]
    st.success(f"Predicted Customer Segment: {result}")
    st.info("Segments A-D are the labels present in the supplied dataset.")
