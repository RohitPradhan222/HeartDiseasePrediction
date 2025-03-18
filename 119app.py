
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Configure logging to save debugging information internally
logging.basicConfig(filename='debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

# Load trained model
with open("119model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load trained scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load dataset
csv_path = "119 Heart.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.error("Error: '119 Heart.csv' not found. Please place it inside the same folder.")
    st.stop()

# Use the optimized feature list from RFECV (dynamically inserted from PART 4)
feature_columns = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Log debugging information internally
logging.info("Features expected by model: %s", feature_columns)
logging.info("Number of features expected: %d", len(feature_columns))

# Streamlit UI
st.title("Heart Disease Prediction App")
st.subheader("Rohit Pradhan 31010922119")
st.write("Enter patient details below to predict heart disease.")

# Input fields aligned with selected features
inputs = {}
for feature in feature_columns:
    if feature == "age":
        inputs[feature] = st.number_input("Age", min_value=20, max_value=100, value=50)
    elif feature == "sex":
        inputs[feature] = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    elif feature == "cp":
        inputs[feature] = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    elif feature == "trestbps":
        inputs[feature] = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    elif feature == "chol":
        inputs[feature] = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    elif feature == "fbs":
        inputs[feature] = st.selectbox("Fasting Blood Sugar (>120 mg/dl, 1 = Yes, 0 = No)", [0, 1])
    elif feature == "restecg":
        inputs[feature] = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=1)
    elif feature == "thalach":
        inputs[feature] = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    elif feature == "exang":
        inputs[feature] = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    elif feature == "oldpeak":
        inputs[feature] = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0)

# Predict button
if st.button("Predict"):
    # Create input data with the exact order and number of features as trained
    input_data = np.array([[inputs[feature] for feature in feature_columns]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 0:
        st.success("Low Risk (No Heart Disease)")
    else:
        st.error("High Risk (Potential Heart Disease)")

    # Model Performance
    if "target" in df.columns:
        y_true = df["target"]
        X_df = df[feature_columns]
        y_pred = model.predict(scaler.transform(X_df))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [accuracy, precision, recall, f1]
        })
        st.subheader("Model Performance Metrics")
        st.table(metrics_df)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"], ax=ax)
        st.pyplot(fig)

        # Class Distribution
        st.subheader("Class Distribution")
        class_counts = pd.Series(y_pred).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low Risk", "High Risk"])
        ax.set_xlabel("Heart Disease Risk")
        ax.set_ylabel("Count")
        st.pyplot(fig)
