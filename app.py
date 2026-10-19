import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("Credit Card Fraud Detection System")
st.write("Enter transaction details to detect whether it is Legitimate or Fraudulent.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

model = load_model()

# Model kitne features expect kar raha hai (Credit card dataset me usually 30 features hote hain: Time, V1-V28, Amount)
num_features = getattr(model, "n_features_in_", 30)

# Input Fields
st.subheader("Transaction Inputs")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)
v1 = st.number_input("Principal Feature V1", value=0.0, step=0.1)
v2 = st.number_input("Principal Feature V2", value=0.0, step=0.1)

if st.button("Check Transaction", type="primary"):
    # 30-length ka array initialize karein zeros ke saath
    input_vector = np.zeros(num_features)
    
    # Specific indices par user input map karein
    # Index 1 = V1, Index 2 = V2, Last Index (-1) = Amount
    input_vector[1] = v1
    input_vector[2] = v2
    input_vector[-1] = amount
    
    # Reshape for single prediction: (1, num_features)
    final_features = input_vector.reshape(1, -1)
    
    prediction = model.predict(final_features)
    
    st.write("---")
    if prediction[0] == 1:
        st.error("⚠️ Alert: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Safe: Legitimate Transaction.")
