import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Health Insurance Cost Predictor")

age = st.number_input("Age")
bmi = st.number_input("BMI")
children = st.number_input("Children")
smoker = st.selectbox("Smoker", ["No", "Yes"])

smoker = 1 if smoker == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[age, bmi, children, smoker]])
    result = model.predict(input_data)
    st.success(f"Estimated Cost: ₹{int(result[0])}")