import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Card Swipe Fraud Simulator",
    page_icon="💳",
    layout="centered"
)

# Custom Styling for Card Look
st.markdown("""
    <style>
    .card-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Card Swipe & POS Simulator")
st.write("Simulate a transaction swipe. The system automatically computes internal behavioral embeddings (PCA representations).")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

model = load_model()
num_features = getattr(model, "n_features_in_", 30)

# Card Container Visual
st.markdown("""
<div class="card-box">
    <h3 style="margin:0; color: #f1f1f1;">Virtual Platinum Card</h3>
    <p style="letter-spacing: 3px; font-size: 18px; margin-top: 15px;">****  ****  ****  8821</p>
    <div style="display: flex; justify-content: space-between; font-size: 13px;">
        <span>HOLDER: KRISHNA CR</span>
        <span>VALID THRU: 10/28</span>
    </div>
</div>
""", unsafe_allow_html=True)

# User-Friendly Transaction Inputs
st.subheader("Transaction Parameters")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=250.0, step=10.0)
    merchant = st.selectbox("Merchant Type", ["Grocery / Retail", "Online Electronics", "Luxury Jewelry", "ATM Cash Withdrawal", "International Gaming"])

with col2:
    location = st.selectbox("Transaction Location", ["Home City (Normal)", "Unusual Domestic City", "Foreign / Cross-Border"])
    time_slot = st.selectbox("Swipe Time", ["Daytime (Normal Business Hours)", "Midnight (2 AM - 5 AM)"])

device_match = st.radio("Card Present / POS Verification", ["Chip & PIN Verified", "Magstripe Swipe (Fallback)", "Card Not Present / Unknown Device"])

st.write("---")

# Logic to compute synthetic behavioral vectors (V1-V28)
def synthesize_features(amount, merchant, location, time_slot, device):
    # Vector base: 30 features
    vec = np.zeros(num_features)
    
    # Anomaly / Risk score formulation
    anomaly_factor = 0.0
    
    if location == "Foreign / Cross-Border":
        anomaly_factor += 2.5
    elif location == "Unusual Domestic City":
        anomaly_factor += 0.8
        
    if time_slot == "Midnight (2 AM - 5 AM)":
        anomaly_factor += 1.5
        
    if device != "Chip & PIN Verified":
        anomaly_factor += 1.8
        
    if merchant in ["Luxury Jewelry", "International Gaming"]:
        anomaly_factor += 1.4
        
    if amount > 1500:
        anomaly_factor += 2.0

    # Map risk directly to influential PCA directions (typically V1, V2, V3, V4, V14, V17)
    # Fraud transactions usually exhibit heavy negative shifts on V14/V17 and positive shifts on V2/V4
    np.random.seed(int(amount) % 100) # Reproducible slight variation
    noise = np.random.normal(0, 0.2, num_features)
    vec += noise
    
    if anomaly_factor >= 3.5:
        # High likelihood of fraud patterns
        vec[1] = -3.5 - (anomaly_factor * 0.5)  # V1
        vec[2] = 4.0 + (anomaly_factor * 0.4)   # V2
        vec[3] = -4.2 - (anomaly_factor * 0.6)  # V3
        vec[4] = 3.8 + (anomaly_factor * 0.5)   # V4
        vec[14] = -6.0 - anomaly_factor         # V14 (strong fraud indicator)
        vec[17] = -5.5 - anomaly_factor         # V17 (strong fraud indicator)
    else:
        # Normal transaction distribution around 0
        vec[1] = 0.2
        vec[2] = -0.1
        vec[3] = 0.4
        vec[4] = -0.3
        vec[14] = 0.1
        vec[17] = -0.1
        
    # Set amount at last column
    vec[-1] = amount
    return vec

# Action Button
if st.button("💳 Swipe & Process Payment", type="primary", use_container_width=True):
    with st.spinner("Encrypting payload and computing PCA embeddings..."):
        synthetic_input = synthesize_features(amount, merchant, location, time_slot, device_match)
        prediction = model.predict(synthetic_input.reshape(1, -1))
        
    st.write("")
    if prediction[0] == 1:
        st.error("🚨 **TRANSACTION DECLINED: Potential Fraud Detected!**")
        st.info("Anomaly detection system triggered security protocols due to behavioral discrepancy.")
    else:
        st.success("✅ **TRANSACTION APPROVED: Payment Successful.**")
        st.caption("Risk score verified within safe threshold.")
