import streamlit as st
import pandas as pd
import pickle
import requests
from streamlit_lottie import st_lottie

# ---------- Page Config ----------
st.set_page_config(
    page_title="Lung Cancer Detection",
    layout="centered",
    page_icon="🏥"
)

# ---------- Load Model ----------
with open("lung_cancer_xgb_model.pkl", "rb") as file:
    model = pickle.load(file)


# ---------- Load Lottie Animation ----------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Opening animation
lottie_opening = load_lottie("https://lottie.host/12c4f9d4-8d2b-4e88-a3d5-5e8b8d9e4ff3/3y7LzHje0F.json")

# ---------- CSS Styling ----------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
h1 {
    color: #0a3d62;
}
.input-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #0a3d62;
    color: white;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border-radius: 10px;
}
.stButton>button:hover {
    background-color: #1e90ff;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Lung Cancer Detection</h1>
        <p style="font-size:18px;">Answer a few questions to predict risk.</p>
    </div>
    """, unsafe_allow_html=True
)

# ---------- Display Opening Animation ----------
if lottie_opening:
    st_lottie(lottie_opening, height=200, key="opening")

# ---------- User Input in Box ----------
st.subheader("Enter Your Details")
Gender = st.radio("Select Gender", ("Male", "Female"), index=None, horizontal=True)
age = st.slider("Age", 18, 100, 30)
smoking = st.radio("Do you smoke?", ("Yes", "No"), index=None, horizontal=True)
yellow_fingures = st.radio("Do you have yellow fingures?", ("Yes", "No"), index=None, horizontal=True)
anxiety = st.radio("Anxiety?", ("Yes", "No"), index=None, horizontal=True)
peer_pressure = st.radio("Peer Pressure?", ("Yes", "No"), index=None, horizontal=True)
chronic_disease = st.radio("Do you have any Chronic Disease?", ("Yes", "No"), index=None, horizontal=True)
fatigue = st.radio("Do you have fatigue?", ("Yes", "No"), index=None, horizontal=True)
allergy = st.radio("Do you have allergy?", ("Yes", "No"), index=None, horizontal=True)
wheezing = st.radio("Do you have wheezing?", ("Yes", "No"), index=None, horizontal=True)
alchol_consuming = st.radio("Do you consume alchol?", ("Yes", "No"), index=None, horizontal=True)
coughing = st.radio("Coughing?", ("Yes", "No"), index=None, horizontal=True)
shortness_of_breath = st.radio("Do you have short breath?", ("Yes", "No"), index=None, horizontal=True)
swallowing_difficulty = st.radio("Swallowing Difficulty?", ("Yes", "No"), index=None, horizontal=True)
chest_pain = st.radio("Do you have chest pain?", ("Yes", "No"), index=None, horizontal=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prepare Input ----------
input_data = {
    "Gender": 1 if Gender == "Male" else 0,
    'Age': age,
    'Smoking': 1 if smoking == 'Yes' else 0,
    'Yellow Fingures': 1 if yellow_fingures == 'Yes' else 0,
    'Anxiety': 1 if anxiety == 'Yes' else 0,
    'Peer Pressure': 1 if peer_pressure == 'Yes' else 0,
    'Chronic Disease': 1 if chronic_disease == 'Yes' else 0,
    'Fatigue': 1 if fatigue == 'Yes' else 0,
    'Allergy': 1 if allergy == 'Yes' else 0,
    'Wheezing': 1 if wheezing == 'Yes' else 0,
    'Alchol Consuming': 1 if alchol_consuming == 'Yes' else 0,
    'Coughing': 1 if coughing == 'Yes' else 0,
    'Shortness of Breath': 1 if shortness_of_breath == 'Yes' else 0,
    'Swallowing Difficulty': 1 if swallowing_difficulty == 'Yes' else 0,
    'Chest Pain': 1 if chest_pain == 'Yes' else 0
}
input_df = pd.DataFrame([input_data])

# ---------- Prediction ----------
if st.button("🔍 Predict Lung Cancer"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

        # Determine risk level
        if prob:
            if prob < 0.4:
                risk_text = "Low Risk ✅"
                color = "#2ecc71"
            elif prob < 0.7:
                risk_text = "Moderate Risk ⚠️"
                color = "#f39c12"
            else:
                risk_text = "High Risk 🔴"
                color = "#ff4b4b"
        else:
            risk_text = "Cancer ⚠️" if pred == 1 else "No Cancer ✅"
            color = "#ff4b4b" if pred == 1 else "#2ecc71"

        # Display result
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:20px;
                text-align:center;
                color:white;
                font-size:25px;
                font-weight:bold;">
                {risk_text}<br>
                {"Probability: " + str(round(prob * 100, 2)) + "%" if prob else ""}
            </div>
            """, unsafe_allow_html=True
        )

        # Progress bar for probability
        if prob:
            st.progress(min(int(prob * 100), 100))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown('---')
