import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/credit_risk_model.pkl")

st.title("Credit Risk Decision System")
st.write("Enter borrower information to estimate loan approval probability.")

feature_names = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

st.subheader("Model Insights")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

st.sidebar.header("Input Borrower Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0, value=360)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

gender_map = {"Female": 0, "Male": 1}
married_map = {"No": 0, "Yes": 1}
education_map = {"Graduate": 0, "Not Graduate": 1}
self_employed_map = {"No": 0, "Yes": 1}
property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

input_data = pd.DataFrame([{
    "Gender": gender_map[gender],
    "Married": married_map[married],
    "Dependents": dependents,
    "Education": education_map[education],
    "Self_Employed": self_employed_map[self_employed],
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area_map[property_area]
}])

if st.sidebar.button("Predict Loan Decision"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.success("Loan Recommendation: Approve")
    else:
        st.error("Loan Recommendation: Reject / Review")

    st.write(f"Approval Probability: {probability:.2%}")

    if probability > 0.7:
        st.write("High confidence approval")
    elif probability > 0.5:
        st.write("Moderate confidence")
    else:
        st.write("Low confidence / higher risk")