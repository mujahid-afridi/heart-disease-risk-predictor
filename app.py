import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and columns
model = joblib.load('LogisticRegression.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# --- Sidebar ---
st.sidebar.title("About this App")
st.sidebar.markdown("""
This app predicts your **risk of heart disease** based on clinical and lifestyle parameters.  

**How to use:**  
1. Fill in your personal and clinical details in the main panel.  
2. Click **Predict** to see your risk level.  

**Disclaimer:**  
This tool is for **educational purposes only** and **does not replace professional medical advice**. Consult a doctor for an accurate diagnosis.
""")

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
Welcome! Fill in your details below and click **Predict** to check your risk of heart disease.
""")

# --- Organize input in sections ---
with st.expander("Patient Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox('Sex', ['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        
    with col2:
        chestPainType = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
        resting_ECG = st.selectbox("Resting ECG", ["Normal", "LHV", "ST"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Predict button ---
if st.button("Predict", type="primary"):
    # Build raw input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'FastingBS': fasting_bs,
        'Sex_'+sex: 1,
        'ChestPainType_'+chestPainType: 1,
        'RestingECG_'+resting_ECG: 1,
        'ExerciseAngina_'+exercise_angina: 1,
        'ST_Slope_'+st_slope: 1
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale numeric columns
    numeric_cols = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result in a nice box
    if prediction == 1:
        st.markdown(
            "<div style='padding: 15px; background-color: #ffcccc; border-radius: 5px; text-align:center; font-size:18px;'>"
            "⚠️ <b>High Risk of Heart Disease</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding: 15px; background-color: #ccffcc; border-radius: 5px; text-align:center; font-size:18px;'>"
            "✅ <b>Low Risk of Heart Disease</b></div>",
            unsafe_allow_html=True
        )
