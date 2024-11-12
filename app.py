# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# Load the trained model
model = joblib.load('diabetes_model_advanced.pkl')

# Function to load lottie animations
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
lottie_success = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")  # Success animation
lottie_warning = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_j5hhd2xk.json")  # Warning animation

# Custom Styling for Better UX with Centered Heading and Spacing
st.set_page_config(page_title="Diabetes Prediction App", page_icon="💉", layout="centered")
st.markdown(
    """
    <style>
        .main {padding: 20px;}
        .title {text-align: center; color: #34495e; font-size: 2.5em; margin-top: -10px; padding-bottom: 10px;}
        .subheader {text-align: center; color: #2e4053; font-size: 1.5em; margin-top: -10px; margin-bottom: 20px;}
        .description {text-align: center; font-size: 1.1em; color: #7f8c8d; margin-bottom: 30px; padding-left: 10px; padding-right: 10px;}
        .stButton>button {background-color: #5dade2; color: white; border-radius: 12px; width: 100%; font-size: 16px; padding: 10px; margin-top: 20px;}
        .stAlert {border-radius: 12px;}
        .footer {text-align: center; font-size: 14px; color: #7f8c8d; margin-top: 40px;}
        .section-header {background-color: #ecf0f1; padding: 15px; margin-top: 20px; border-radius: 8px; font-weight: bold; color: #2e4053; font-size: 1.2em;}
        .metric-label {color: #34495e; font-weight: bold; font-size: 16px; margin-bottom: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Header and Introduction
st.markdown('<div class="title">🔍 Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">A Smart Prediction System for Diabetes Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Use this application to predict the likelihood of diabetes based on key health metrics. Adjust the inputs, and the prediction will update in real time.</div>', unsafe_allow_html=True)

# Sidebar with Instructions
st.sidebar.title("User Instructions")
st.sidebar.write(
    """
    1. Adjust the sliders to match the patient's health metrics.
    2. Observe real-time risk assessment on the main screen.
    3. Note: This tool provides preliminary guidance. Consult with a healthcare provider for definitive insights.
    """
)

# Health Metrics Section
st.markdown('<div class="section-header">🩺 Patient Health Metrics</div>', unsafe_allow_html=True)
st.write("Adjust the sliders below based on the patient's health metrics:")

# Sliders for input metrics
pregnancies = st.slider("Number of Pregnancies", 0, 20, step=1)
glucose = st.slider("Glucose Level", 0, 200, step=1)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 180, step=1)
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, step=1)
insulin = st.slider("Insulin Level", 0, 900, step=10)
bmi = st.slider("BMI", 0.0, 70.0, step=0.1)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
age = st.slider("Age", 0, 120, step=1)

# Feature engineering for additional inputs
bmi_age = bmi * age
glucose_blood_pressure = glucose * blood_pressure

# Define input data for prediction
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, bmi_age, glucose_blood_pressure]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Age', 'Glucose_BloodPressure'])

# Prediction process with loading spinner
with st.spinner("Analyzing data..."):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

# Display results with high-impact visuals and animations
st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)

# Interpretation based on prediction result with animations
if prediction == 1:
    if lottie_warning:
        st_lottie(lottie_warning, height=100, width=100)
    st.error(f"⚠️ **High Risk of Diabetes Detected** (Confidence: {prediction_proba:.2f})")
    st.write("**Recommendation:** Please consult a healthcare provider for further guidance.")
    st.markdown(
        """
        <style>
        .main {background-color: #ffe6e6;}
        </style>
        """, unsafe_allow_html=True)
else:
    if lottie_success:
        st_lottie(lottie_success, height=100, width=100)
    st.success(f"✅ **Low Risk of Diabetes** (Confidence: {1 - prediction_proba:.2f})")
    st.write("**Note:** Regular check-ups are recommended to ensure ongoing health.")
    st.markdown(
        """
        <style>
        .main {background-color: #e6ffe6;}
        </style>
        """, unsafe_allow_html=True)

# Gauge meter for confidence level
st.markdown('<div class="section-header">📊 Diabetes Risk Confidence Level</div>', unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction_proba if prediction == 1 else (1 - prediction_proba),
    gauge={'axis': {'range': [0, 1]},
           'bar': {'color': 'red' if prediction == 1 else 'green'},
           'steps': [
               {'range': [0, 0.5], 'color': 'lightgreen'},
               {'range': [0.5, 1], 'color': 'lightcoral'}]},
    title={'text': "Confidence Level"}
))
st.plotly_chart(fig_gauge)

# Health Tips Section
st.markdown('<div class="section-header">💡 Health Tips</div>', unsafe_allow_html=True)
if prediction == 1:
    st.write("1. **Maintain a balanced diet**: Include fiber-rich foods and reduce sugar intake.")
    st.write("2. **Exercise regularly**: Physical activity helps in regulating blood sugar levels.")
    st.write("3. **Monitor blood pressure**: High blood pressure is a known risk factor.")
else:
    st.write("1. **Keep up the good work**: Continue with a balanced diet and regular exercise.")
    st.write("2. **Get regular check-ups**: Monitoring health parameters helps in early detection.")

# Explanation of metrics using separate expanders (not nested)
st.markdown('<div class="section-header">🔍 More Information on Health Metrics</div>', unsafe_allow_html=True)
st.write("Learn more about how each health metric affects diabetes risk:")

st.expander("Pregnancies").write("Higher numbers of pregnancies may increase diabetes risk due to hormonal changes.")
st.expander("Glucose Level").write("Higher glucose levels are associated with an increased risk of diabetes.")
st.expander("Blood Pressure").write("High blood pressure is a known risk factor for diabetes.")
st.expander("Skin Thickness").write("Skinfold thickness can indicate the level of body fat.")
st.expander("Insulin Level").write("High insulin levels can indicate insulin resistance, a risk factor for diabetes.")
st.expander("BMI").write("BMI measures body mass index, where higher values may indicate overweight or obesity.")
st.expander("Diabetes Pedigree Function").write("Higher values indicate a stronger family history of diabetes.")
st.expander("Age").write("Higher age can increase diabetes risk due to physiological changes over time.")

# Footer for professional touch
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="footer">© 2024 Diabetes Prediction App | Powered by Machine Learning and Data Science<br>Contact us at support@diabetesapp.com</div>',
    unsafe_allow_html=True
)
