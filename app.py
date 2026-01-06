import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Random Forest Prediction", layout="centered")
st.title("🧠 Random Forest Classifier Test App")

# Load model
model = joblib.load(r'/home/intellect-1001/Desktop/Data_Scientist-/Day 4 FN-20251212T063722Z-1-001/Day 4 FN/vin3.project/ans.pkl')

st.subheader("Enter Employee Details")

# User inputs
age = st.number_input("Age", min_value=18, max_value=65, value=30)
salary = st.number_input("Salary", min_value=10000, max_value=200000, value=50000)

department = st.selectbox(
    "Department",
    ["HR", "IT", "Sales"]
)

years_exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

# Encode department (MUST match training)
dept_mapping = {"HR": 0, "IT": 1, "Sales": 2}
department_encoded = dept_mapping[department]

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Salary": salary,
        "Department": department_encoded,
        "YearsExperience": years_exp
    }])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success(" Purchased")
    else:
        st.error(" Not Purchased")
