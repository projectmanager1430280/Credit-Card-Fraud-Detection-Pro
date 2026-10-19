import joblib
import numpy as np
import streamlit as st

# Load the trained model
model = joblib.load("saved.pkl")

st.title("Credit Card Fraud Detection")

# User inputs for features
amount = st.number_input("Amount", min_value=0.0)
v1 = st.number_input("V1")
v2 = st.number_input("V2")
# ... add inputs for other features as needed

if st.button("Predict"):
    features = np.array([v1, v2, amount]).reshape(1, -1)
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("Fraudulent Transaction!")
    else:
        st.success("Legitimate Transaction.")
