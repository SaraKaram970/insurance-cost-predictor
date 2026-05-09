import streamlit as st
import numpy as np
import pickle

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Insurance Predictor", page_icon="💰")

# =========================
# Background
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #74ebd5, #ACB6E5);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load model
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# Title
# =========================
st.markdown('<div class="title">💰 Insurance Cost Predictor</div>', unsafe_allow_html=True)

# =========================
# Form
# =========================
with st.form("form"):

    st.subheader("📝 Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        sex = 1 if sex == "male" else 0

        smoker = st.selectbox("Smoker", ["yes", "no"])
        smoker = 1 if smoker == "yes" else 0

        region = st.selectbox(
            "Region",
            ["southeast", "southwest", "northeast", "northwest"]
        )

        region_map = {
            "southeast": 0,
            "southwest": 1,
            "northeast": 2,
            "northwest": 3
        }
        region = region_map[region]

    submit = st.form_submit_button("🚀 Predict Insurance Cost")

# =========================
# Prediction
# =========================
if submit:

    bmi_deviation = abs(bmi - 22)
    smoker_intensity = smoker * age
    health_index = bmi * 0.3 + age * 0.5 + smoker * 30
    high_risk = int(smoker == 1 and bmi > 30 and age > 50)
    age_bmi_smoker = age * bmi * smoker

    features = np.array([[
        age, sex, bmi, children, smoker, region,
        bmi_deviation, smoker_intensity,
        health_index, high_risk, age_bmi_smoker
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    real_value = np.exp(prediction[0])

    st.success(f"💵 Estimated Insurance Cost: ${real_value:,.2f}")