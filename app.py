import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD AI TOOLS
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')
encoders = joblib.load('heart_encoders.pkl')

st.set_page_config(page_title="PulseGuard AI", page_icon="❤️")
st.title("❤️ PulseGuard: Heart Disease Predictor")

# 2. SIDEBAR INPUTS
def get_input():
    st.sidebar.header("Patient Data")
    age = st.sidebar.number_input("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["MALE", "FEMALE"])
    cp = st.sidebar.selectbox("Chest Pain", ["TYPICAL ANGINA", "ASYMPTOMATIC", "NON-ANGINAL", "ATYPICAL ANGINA"])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", ["TRUE", "FALSE"])
    restecg = st.sidebar.selectbox("Resting ECG", ["NORMAL", "LV HYPERTROPHY", "ST-T ABNORMALITY"])
    thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Angina", ["TRUE", "FALSE"])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Slope", ["UPSLOPING", "FLAT", "DOWNSLOPING"])
    ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", ["NORMAL", "FIXED DEFECT", "REVERSABLE DEFECT"])
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(data, index=[0])

user_input_df = get_input()

# 3. PREDICTION LOGIC
if st.button("Run Diagnostic"):
    proc_df = user_input_df.copy()

    # Normalize Text to Uppercase to match Encoders
    for col in proc_df.columns:
        if proc_df[col].dtype == 'object':
            proc_df[col] = proc_df[col].astype(str).str.upper()

    # Encode Categorical Data
    for col, le in encoders.items():
        if col in proc_df.columns:
            # Sync encoder classes to uppercase as well
            le.classes_ = np.array([str(c).upper() for c in le.classes_])
            proc_df[col] = le.transform(proc_df[col])
    
    # Scale and Predict
    scaled_data = scaler.transform(proc_df)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 High Risk: {prob*100:.1f}% probability.")
    else:
        st.success(f"✅ Low Risk: {prob*100:.1f}% probability.")