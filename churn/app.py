from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, feature names, label encoders, and scaler
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

best_xgb = model_data["model"]
feature_names = model_data["feature_names"]

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define categorical and numerical features
categorical_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Root route
@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API! Use the /predict endpoint to make predictions."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert JSON data to a DataFrame
        new_data = pd.DataFrame([data])
        
        # Preprocess the data
        # 1. Convert 'TotalCharges' to numeric and handle missing values
        new_data['TotalCharges'] = pd.to_numeric(new_data['TotalCharges'], errors='coerce')
        new_data['TotalCharges'].fillna(new_data['TotalCharges'].median(), inplace=True)
        
        # 2. Label encode categorical features
        for col in categorical_features:
            if col in new_data.columns:
                new_data[col] = label_encoders[col].transform(new_data[col])
        
        # 3. Scale numerical features
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        
        # 4. Select the same features used during training
        new_data_selected = new_data[feature_names]
        
        # Make predictions
        prediction = best_xgb.predict(new_data_selected)
        probability = best_xgb.predict_proba(new_data_selected)[:, 1]
        
        # Map prediction to "Churn" or "Stay"
        churn_labels = ["Stay", "Churn"]
        prediction_mapped = churn_labels[prediction[0]]
        
        # Return prediction as JSON response
        return jsonify({
            "prediction": prediction_mapped,
            "probability": float(probability[0])
        })
    
    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)