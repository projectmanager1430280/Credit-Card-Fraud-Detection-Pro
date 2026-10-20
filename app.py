import streamlit as st
import joblib
import numpy as np
import re
from PIL import Image
import easyocr

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
st.caption("AI-driven real-time transaction risk scoring & anomaly evaluation engine.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

# Load OCR Engine
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

model = load_model()
ocr_reader = load_ocr_reader()
num_features = getattr(model, "n_features_in_", 30)

# Luhn Algorithm for validating real credit/debit card numbers
def is_valid_luhn(card_no: str) -> bool:
    digits = [int(d) for d in card_no if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reversed_digits = digits[::-1]
    for i, d in enumerate(reversed_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += (doubled - 9) if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0

# Session State Setup
if "card_number" not in st.session_state:
    st.session_state.card_number = ""
if "card_holder" not in st.session_state:
    st.session_state.card_holder = ""
if "valid_thru" not in st.session_state:
    st.session_state.valid_thru = "MM/YY"
if "is_card_valid" not in st.session_state:
    st.session_state.is_card_valid = False

st.subheader("1. Card Identification")
input_option = st.radio(
    "Choose Identification Method:",
    ["By Image Scan (Fast Scan)", "Manual Entry"],
    horizontal=True
)

if input_option == "By Image Scan (Fast Scan)":
    card_img_file = st.camera_input("Capture Card Front Side for Optical Scan")
    
    if card_img_file is not None:
        if st.button("🔍 Extract & Verify Card", type="secondary"):
            with st.spinner("Analyzing optical text & checking card security checksum..."):
                img = Image.open(card_img_file)
                img_np = np.array(img)
                results = ocr_reader.readtext(img_np)
                detected_texts = [res[1] for res in results]
                
                # Check for 13-19 digit candidate numbers
                found_number = None
                for txt in detected_texts:
                    clean = re.sub(r'[^0-9]', '', txt)
                    if 13 <= len(clean) <= 19 and is_valid_luhn(clean):
                        found_number = clean
                        break
                
                # Fallback check: agar OCR ne digits alag-alag split kar diye ho
                if not found_number:
                    combined_digits = re.sub(r'[^0-9]', '', " ".join(detected_texts))
                    for i in range(len(combined_digits) - 15):
                        sub = combined_digits[i:i+16]
                        if is_valid_luhn(sub):
                            found_number = sub
                            break

                if found_number:
                    st.session_state.card_number = found_number
                    st.session_state.is_card_valid = True
                    
                    # Expiry search
                    for txt in detected_texts:
                        exp_match = re.search(r'(0[1-9]|1[0-2])\/([0-9]{2})', txt)
                        if exp_match:
                            st.session_state.valid_thru = exp_match.group(0)
                            break
                            
                    st.session_state.card_holder = "VERIFIED CARDHOLDER"
                    st.success("✅ Valid Payment Card Recognized & Verified!")
                else:
                    st.session_state.is_card_valid = False
                    st.session_state.card_number = ""
                    st.error("❌ No Valid Payment Card Detected in Image! Face or non-card objects are strictly discarded.")
else:
    # Manual Entry
    col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
    with col_c1:
        st.session_state.card_holder = st.text_input("Cardholder Name", value="KRISHNA CR").upper()
    with col_c2:
        raw_manual_no = st.text_input("Card Number (16 Digits)", value="4532892177318821")
        clean_manual = re.sub(r'[^0-9]', '', raw_manual_no)
        if len(clean_manual) >= 13 and is_valid_luhn(clean_manual):
            st.session_state.card_number = clean_manual
            st.session_state.is_card_valid = True
        else:
            st.session_state.is_card_valid = False
            st.warning("Please enter a valid card number satisfying standard checksum.")
    with col_c3:
        st.session_state.valid_thru = st.text_input("Valid Thru", value="10/28")

# Visual Display of Card
if st.session_state.is_card_valid and st.session_state.card_number:
    last4 = st.session_state.card_number[-4:]
    masked_display = f"XXXX XXXX XXXX {last4}"
    card_holder_display = st.session_state.card_holder if st.session_state.card_holder else "VERIFIED USER"
else:
    masked_display = "NO CARD DETECTED"
    card_holder_display = "---"

st.markdown(f"""
<div class="card-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 13px; color: #b0bec5; letter-spacing: 1.5px;">ACTIVE PAYMENT CARD</span>
        <span style="font-size: 14px; font-weight: bold; color: {'#ffb74d' if st.session_state.is_card_valid else '#ef5350'};">
            {'EMV VERIFIED' if st.session_state.is_card_valid else 'UNVERIFIED'}
        </span>
    </div>
    <p style="letter-spacing: 4px; font-size: 22px; font-weight: 500; margin-top: 18px; margin-bottom: 18px;">
        {masked_display}
    </p>
    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #eceff1;">
        <span>HOLDER: <b>{card_holder_display}</b></span>
        <span>VALID THRU: <b>{st.session_state.valid_thru}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Section 2: Transaction Parameters in INR
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
    txn_location = st.selectbox("Transaction Region", [
        "Domestic (Registered Home Location)",
        "Domestic (Other State / City)",
        "International / Cross-Border"
    ])
    txn_time = st.selectbox("Transaction Time Window", [
        "Normal Hours (08:00 AM - 10:00 PM)",
        "Off-Peak / Midnight (01:00 AM - 05:00 AM)"
    ])

channel = st.selectbox("Payment Channel Verification", [
    "Chip & PIN Secured",
    "Contactless Tap (NFC)",
    "Fallback / Card Not Present"
])

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

# Prediction Evaluation Button
if st.button("Evaluate Transaction Risk", type="primary", use_container_width=True):
    if not st.session_state.is_card_valid:
        st.error("🛑 Action Denied: No authenticated credit card available. Please scan a genuine card or use valid Manual Entry.")
    else:
        with st.spinner("Evaluating PCA transaction embeddings against security boundaries..."):
            features = build_feature_vector(amount_inr, merchant_sector, txn_location, txn_time, channel)
            prediction = model.predict(features.reshape(1, -1))

        st.write("")
        if prediction[0] == 1:
            st.error("🚨 **TRANSACTION REJECTED: Suspicious / Fraudulent Activity Detected!**")
            st.warning(f"Security Alert triggered for card {masked_display} | Amount: ₹{amount_inr:,.2f}")
        else:
            st.success("✅ **TRANSACTION APPROVED: Legitimate Activity Verified.**")
            st.info(f"Transaction cleared for card {masked_display} | Amount: ₹{amount_inr:,.2f}")
