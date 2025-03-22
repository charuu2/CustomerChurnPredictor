from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
with open("improved_model.pkl", "rb") as f:
    model = pickle.load(f)



# Define the numerical features
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Route for home page (HTML Form)
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from request
        input_data = {
            'tenure': [float(request.form["tenure"])],
            'MonthlyCharges': [float(request.form["MonthlyCharges"])],
            'TotalCharges': [float(request.form["TotalCharges"])],
            'gender': [int(request.form["gender"])],
            'SeniorCitizen': [int(request.form["SeniorCitizen"])],
            'Partner': [int(request.form["Partner"])],
            'Dependents': [int(request.form["Dependents"])],
            'PhoneService': [int(request.form["PhoneService"])],
            'MultipleLines': [int(request.form["MultipleLines"])],
            'InternetService': [int(request.form["InternetService"])],
            'OnlineSecurity': [int(request.form["OnlineSecurity"])],
            'OnlineBackup': [int(request.form["OnlineBackup"])],
            'DeviceProtection': [int(request.form["DeviceProtection"])],
            'TechSupport': [int(request.form["TechSupport"])],
            'StreamingTV': [int(request.form["StreamingTV"])],
            'StreamingMovies': [int(request.form["StreamingMovies"])],
            'Contract': [int(request.form["Contract"])],
            'PaperlessBilling': [int(request.form["PaperlessBilling"])],
            'PaymentMethod': [int(request.form["PaymentMethod"])]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_text = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"

        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
