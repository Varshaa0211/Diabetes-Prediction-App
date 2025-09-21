import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🩸 Diabetes Prediction App",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Background Styling (Colorful)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
        color: #000000;
        font-family: 'Comic Sans MS', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Title & Description
# ----------------------------
st.title("🩸 Diabetes Prediction App 🩸")
st.write("""
Welcome! 🖐️  
This app predicts whether a person **has diabetes or not** based on health parameters.  
Please enter the details below ⬇️
""")

# ----------------------------
# Input Fields
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("🩺 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("🩹 Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("💉 Insulin Level (IU/mL)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")  # Ensure this file is in the same folder

model = load_model()

# ----------------------------
# Prediction
# ----------------------------
if st.button("🩸 Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Oh no! The person *may have diabetes*. Take care! 🍏🥦")
    else:
        st.success("✅ Great! The person *is unlikely to have diabetes*. Keep healthy! 🥗🏃‍♂️")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed with ❤️ and 🩸 by Varsha A. 🌟✨")
