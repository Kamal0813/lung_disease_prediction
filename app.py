import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from streamlit_carousel import carousel
import gdown
import os
import pandas as pd
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score,f1_score,confusion_matrix
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Lung Cancer Detection System",
    layout="wide"
)

carousel_items = [
    {"img": "images/img1.jpg", "title": "Early Detection Saves Lives", "text": "Detect lung cancer faster with AI"},
    {"img": "images/img2.jpeg", "title": "CT Scan Based Analysis", "text": "Upload scan and get instant prediction"}
]

@st.cache_resource
def load_xgb_model():
    with open("lung_cancer_xgb_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


@st.cache_resource
def load_cnn_model():
    model_path = "model.keras"

    if not os.path.exists(model_path):
        file_id = "1CHLjsExzKznbDbYgUiwZTc1lyGXbhqqQ"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output=model_path, quiet=False, fuzzy=True)

    # Debug: Show file size
    st.write("Downloaded file size:", os.path.getsize(model_path))

    return load_model(model_path)


model_xgb = load_xgb_model()
model_cnn = load_cnn_model()


st.markdown("""
<style>

.stButton>button {
    background-color: #0077b6 !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #00b4d8;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00b4d8 !important;
    color: black !important;
}
div[data-testid="stTabs"] button {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

carousel(items=carousel_items, fade=True, container_height=600)
st.markdown("<h1 style='text-align:center;'>Lung Cancer Detection System</h1>", unsafe_allow_html=True)



tab1, tab2 = st.tabs(["Patient Evaluation", "Image Diagnosis"])


with tab1:
    st.subheader("Enter Medical Information")

    def radio(label):
        return 1 if st.pills(label, ("Yes", "No"), selection_mode="single") == "Yes" else 0

    inputs = {}


    col_age, col_gender = st.columns([1, 1])
    with col_age:
        inputs['Age'] = st.number_input(
            "Enter your age",
            min_value=18,
            max_value=100,
            step=1,
            format="%d"
        )
    with col_gender:
        gender = st.selectbox("Gender:", ("Male", "Female"), index=None, placeholder="Select Gender")
        inputs["Gender"] = 1 if gender == "Male" else 0

    # Symptoms in 2 columns
    symptoms = [
        "Do you Smoke?", "Do you have yellow fingers?", "Do you feel Anxiety?",
        "Do you experience any kind of peer pressure?", "Do you have any chronic medical conditions?",
        "Do you often feel unusually tired or fatigued?", "Do you have any known allergies?",
        "Do you experience wheezing (a whistling sound while breathing)?", "Do you consume alcohol?",
        "Do you have frequent or persistent coughing?", "Do you feel breathless during normal activities?",
        "Do you have difficulty swallowing food or liquids?", "Do you have frequent chest discomfort?"
    ]

    col1, col2 = st.columns(2)
    for i, s in enumerate(symptoms):
        with (col1 if i % 2 == 0 else col2):
            inputs[s] = radio(s)

    input_df = pd.DataFrame([inputs])


    if st.button("Predict Lung Cancer Risk"):
        prediction = model_xgb.predict(input_df)[0]
        prob = model_xgb.predict_proba(input_df)[0][1]

        if prob < 0.4:
            status, color = "Low Risk", "#2ecc71"
        elif prob < 0.7:
            status, color = "Moderate Risk", "#f39c12"
        else:
            status, color = "High Risk", "#e74c3c"

        st.markdown(f"""
        <div style="background-color:{color};padding:20px;border-radius:15px;
                    text-align:center;font-size:25px;color:white;font-weight:bold;">
            {status}<br>Probability: {prob * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob * 100))

with tab2:
    st.subheader("Upload and Analyze CT Scan Image")

    CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

    file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

    if file:
        st.image(file, caption="Uploaded CT Scan", width=600)

        if st.button("Analyze CT Scan"):
            with st.spinner("Processing image..."):
                img = load_img(file, target_size=(224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                img = np.expand_dims(img, 0)

                pred = model_cnn.predict(img)
                class_id = np.argmax(pred)
                confidence = np.max(pred) * 100

                st.success(f"Prediction: **{CLASS_NAMES[class_id]}**")
                st.info(f"Confidence: {confidence:.2f}%")


