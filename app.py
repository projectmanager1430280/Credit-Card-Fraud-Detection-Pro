import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
from qreader import QReader

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .card-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 22px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Credit Card Fraud Detection System")
st.caption("AI-powered real-time transaction monitoring and anomaly detection engine.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

# Load QReader instance
@st.cache_resource
def load_qreader():
    return QReader()

model = load_model()
qreader = load_qreader()
num_features = getattr(model, "n_features_in_", 30)

# Session state initialization for dynamic persistence
if "card_number" not in st.session_state:
    st.session_state.card_number = "XXXX XXXX XXXX 8821"
if "card_holder" not in st.session_state:
    st.session_state.card_holder = "KRISHNA CR"
if "valid_thru" not in st.session_state:
    st.session_state.valid_thru = "10/28"

st.subheader("1. Card Identification")
input_mode = st.radio("Card Input Mode:", ["Manual / Direct Entry", "Real-time QR Card Scanner"], horizontal=True)

if input_mode == "Real-time QR Card Scanner":
    camera_pic = st.camera_input("Scan Card QR Code")
    if camera_pic:
        # Image processing for QReader
        img = Image.open(camera_pic)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Decode QR
        decoded_texts = qreader.detect_and_decode(image=img_bgr)
        
        if decoded_texts and decoded_texts[0]:
            qr_content = decoded_texts[0].strip()
            st.success(f"Scanned QR Payload: `{qr_content}`")
            
            # Format parsing: supports 'CARD_NUMBER,NAME,EXPIRY' or plain CARD_NUMBER
            parts = [p.strip() for p in qr_content.split(",")]
            raw_num = parts[0].replace(" ", "").replace("-", "")
            
            if len(raw_num) >= 4:
                st.session_state.card_number = f"XXXX XXXX XXXX {raw_num[-4:]}"
            else:
                st.session_state.card_number = f"XXXX XXXX XXXX {raw_num}"
                
            if len(parts) >= 2:
                st.session_state.card_holder = parts[1].upper()
            if len(parts) >= 3:
                st.session_state.valid_thru = parts[2]
        else:
            st.warning("QR code detect nahi ho paya. Lighting aur angle check karein.")
else:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.session_state.card_holder = st.text_input("Cardholder Name", value=st.session_state.card_holder).upper()
    with col_c2:
        raw_card = st.text_input("Card Number", value="4532 8921 7731 8821")
        clean_num = raw_card.replace(" ", "").replace("-", "")
        if len(clean_num) >= 4:
            st.session_state.card_number = f"XXXX XXXX XXXX {clean_num[-4:]}"

# Real-time Card View
st.markdown(f"""
<div class="card-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 14px; color: #b0bec5; letter-spacing: 1.5px;">ACTIVE PAYMENT CARD</span>
        <span style="font-size: 16px; font-weight: bold; color: #ffb74d;">EMV SECURED</span>
    </div>
    <p style="letter-spacing: 4px; font-size: 20px; font-weight: 500; margin-top: 18px; margin-bottom: 18px;">
        {st.session_state.card_number}
    </p>
    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #eceff1;">
        <span>HOLDER: <b>{st.session_state.card_holder}</b></span>
        <span>VALID THRU: <b>{st.session_state.valid_thru}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Transaction Inputs in INR (₹)
st.subheader("2. Transaction Parameters")
col1, col2 = st.columns(2)

with col1:
    amount_inr = st.number_input("Transaction Amount (₹ INR)", min_value=10.0, value=2500.0, step=100.0)
    merchant_sector = st.selectbox("Merchant Category", [
        "Retail / Departmental Store",
        "Electronics & Gadgets",
        "Jewelry & Luxury Goods",
        "Automated Fuel Dispenser",
        "Online Foreign Gaming / Crypto"
    ])

with col2:
    txn_location = st.selectbox("Transaction Region", ["Domestic (Registered Home Location)", "Domestic (Other State / City)", "International / Cross-Border"])
    txn_time = st.selectbox("Transaction Time Window", ["Normal Hours (08:00 AM - 10:00 PM)", "Off-Peak / Midnight (01:00 AM - 05:00 AM)"])

channel = st.selectbox("Payment Channel Verification", ["Chip & PIN Secured", "Contactless Tap (NFC)", "Fallback / Card Not Present"])

st.write("---")

# Feature Vector Synthesizer
def build_feature_vector(amount, sector, loc, time_win, ch):
    vec = np.zeros(num_features)
    risk_score = 0.0

    if loc == "International / Cross-Border":
        risk_score += 2.8
    elif loc == "Domestic (Other State / City)":
        risk_score += 0.8

    if time_win == "Off-Peak / Midnight (01:00 AM - 05:00 AM)":
        risk_score += 1.6

    if ch == "Fallback / Card Not Present":
        risk_score += 1.9

    if sector in ["Jewelry & Luxury Goods", "Online Foreign Gaming / Crypto"]:
        risk_score += 1.5

    if amount >= 150000:
        risk_score += 2.2

    np.random.seed(int(amount) % 100)
    vec += np.random.normal(0, 0.15, num_features)

    if risk_score >= 3.8:
        vec[1] = -4.0 - (risk_score * 0.4)
        vec[2] = 4.2 + (risk_score * 0.3)
        vec[3] = -4.5 - (risk_score * 0.5)
        vec[4] = 4.0 + (risk_score * 0.4)
        vec[14] = -7.0 - risk_score
        vec[17] = -6.0 - risk_score
    else:
        vec[1] = 0.15
        vec[2] = -0.1
        vec[3] = 0.3
        vec[4] = -0.2
        vec[14] = 0.05
        vec[17] = -0.08

    vec[-1] = amount / 85.0
    return vec

# Prediction Evaluation
if st.button("Evaluate Transaction Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing transaction vectors against trained fraud boundaries..."):
        features = build_feature_vector(amount_inr, merchant_sector, txn_location, txn_time, channel)
        prediction = model.predict(features.reshape(1, -1))

    st.write("")
    if prediction[0] == 1:
        st.error("🚨 **TRANSACTION REJECTED: Suspicious / Fraudulent Activity Detected!**")
        st.warning(f"Flagged for card {st.session_state.card_number} | Amount: ₹{amount_inr:,.2f}")
    else:
        st.success("✅ **TRANSACTION APPROVED: Legitimate Activity Verified.**")
        st.info(f"Cleared for card {st.session_state.card_number} | Amount: ₹{amount_inr:,.2f}")
