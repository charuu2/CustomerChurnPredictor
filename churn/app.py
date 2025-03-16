import streamlit as st
import numpy as np
import pickle

# Load the trained churn prediction model
with open("improved_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction & Retention Recommendations")

# ✅ Collect all required inputs (ensure all 19 features are included)
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "Not available"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "Not available"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "Not available"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "Not available"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "Not available"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "Not available"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "Not available"])
phone_service = st.selectbox("Phone Service?", ["Yes", "No"])

# Convert categorical inputs into numeric values
gender = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
multiple_lines = 1 if multiple_lines == "Yes" else 0
phone_service = 1 if phone_service == "Yes" else 0
paperless_billing = 1 if paperless_billing == "Yes" else 0

contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_mapping = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}
internet_mapping = {"DSL": 0, "Fiber optic": 1, "No": 2}
binary_mapping = {"Yes": 1, "No": 0, "Not available": 2}

contract_type = contract_mapping[contract_type]
payment_method = payment_mapping[payment_method]
internet_service = internet_mapping[internet_service]
online_security = binary_mapping[online_security]
online_backup = binary_mapping[online_backup]
device_protection = binary_mapping[device_protection]
tech_support = binary_mapping[tech_support]
streaming_tv = binary_mapping[streaming_tv]
streaming_movies = binary_mapping[streaming_movies]

# Create input feature array
input_data = np.array([[tenure, monthly_charges, total_charges, contract_type, payment_method, internet_service,
                         online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                         paperless_billing, gender, senior_citizen, partner, dependents, multiple_lines, phone_service]])

# Ensure shape matches
if input_data.shape[1] != 19:
    st.error(f"Feature mismatch: Expected 19, but got {input_data.shape[1]}")
else:
    # Make prediction
    churn_prob = model.predict_proba(input_data)[0][1]  # Get churn probability

    # **Churn risk interpretation**
    if churn_prob < 0.3:
        risk_level = "🟢 Low Risk (Likely to Stay)"
        retention_message = "This customer is likely to stay. Continue providing good service!"
    elif 0.3 <= churn_prob < 0.7:
        risk_level = "🟡 Medium Risk (Might Churn)"
        retention_message = "Consider offering personalized discounts or improved customer support."
    else:
        risk_level = "🔴 High Risk (Likely to Churn)"
        retention_message = "Take immediate action! Provide special offers, loyalty programs, or better tech support."

    # Display prediction results
    st.subheader("🔍 Churn Prediction Result")
    st.metric(label="Churn Probability", value=f"{churn_prob * 100:.2f}%")
    st.warning(f"**Risk Level:** {risk_level}")
    st.info(f"**Retention Advice:** {retention_message}")

    # **Personalized Recommendations**
    st.subheader("🔹 Customer Retention Recommendations")
    retention_advice = []

    if contract_type == 0:  # Month-to-month contracts are risky
        retention_advice.append("Offer discounts for switching to yearly contracts.")
    if internet_service == 1:  # Fiber optic users may churn due to high prices
        retention_advice.append("Provide flexible payment plans for fiber users.")
    if tech_support == 0:  # No tech support users
        retention_advice.append("Offer free tech support trials to improve customer experience.")
    if payment_method == 0:  # Electronic check users tend to churn more
        retention_advice.append("Encourage credit card or bank transfer payments for better retention.")

    if retention_advice:
        for advice in retention_advice:
            st.write(f"✅ {advice}")
    else:
        st.write("No additional retention actions needed.")

    # Collect Customer Feedback for Improvement
    st.subheader("💬 Customer Feedback for Retention")
    feedback = st.text_area("How can we improve our service to retain this customer?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! This will help improve our retention strategies.")
