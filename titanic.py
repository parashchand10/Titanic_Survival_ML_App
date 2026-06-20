import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load assets
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
    /* Overall page */
    .main {
        background-color: #f7f8fa;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* App title / hero */
    .hero {
        padding: 18px 0 6px 0;
        border-bottom: 1px solid #e6e8eb;
        margin-bottom: 24px;
    }
    .hero {
        text-align: center;
    }
    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 2px;
    }
    .hero p {
        color: #6b7280;
        font-size: 15px;
        margin-top: 0;
    }

    /* Card container (st.container border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
        padding: 14px 10px 6px 10px;
        min-height: 560px;
    }
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Section labels inside form */
    label, .stSlider label, .stSelectbox label, .stNumberInput label, .stRadio label {
        font-weight: 600 !important;
        color: #374151 !important;
        font-size: 13.5px !important;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #ff4b4b, #e63946);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 0;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(230, 57, 70, 0.35);
    }

    /* Result card */
    .result-box {
        text-align: center;
        padding: 36px 20px;
        border-radius: 14px;
        margin-top: 20px;
        border: 1px solid #eef0f2;
    }
    .result-box h1 {
        font-size: 44px;
        font-weight: 900;
        margin: 8px 0 4px 0;
        letter-spacing: 0.5px;
    }
    .result-box p {
        color: #6b7280;
        font-size: 14px;
        margin: 0;
    }
    .result-icon {
        font-size: 42px;
    }

    /* Info note */
    .note {
        background-color: #eef2ff;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13.5px;
        color: #4338ca;
        margin-top: 18px;
    }

    /* Placeholder before prediction */
    .placeholder {
        text-align: center;
        padding: 60px 20px;
        color: #9ca3af;
    }
    .placeholder .emoji {
        font-size: 48px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Hero Header ----------------
st.markdown("""
    <div class="hero">
        <h1>Titanic Survival Prediction</h1>
        <p>Enter passenger details to estimate the likelihood of survival, powered by a Decision Tree model.</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- Layout ----------------
col1, col2 = st.columns([1.3, 1.4], gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<div class="card-title">🧍 Passenger Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            age = st.slider("Age", 0, 80, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            pclass = st.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
            sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        with c2:
            fare = st.number_input("Fare (Ticket Price)", 0, 512, 32)
            embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
            has_cabin = st.radio("Has a Cabin?", ["Yes", "No"], horizontal=True)
            parch = st.number_input("Parents/Children", 0, 10, 0)

        predict_clicked = st.button("Predict Survival", use_container_width=True, type="primary")

with col2:
    with st.container(border=True):
        st.markdown('<div class="card-title">Prediction Result</div>', unsafe_allow_html=True)

        # Preprocessing
        is_female = 1 if gender == "Female" else 0
        cabin_encoded = 1 if has_cabin == "Yes" else 0
        p_high = 1 if pclass == "High (1st)" else 0
        p_mid = 1 if pclass == "Mid (2nd)" else 0
        p_low = 1 if pclass == "Low (3rd)" else 0
        emb_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
        emb_encoded = emb_map[embarked]

        input_data = {'Age': age, 'has_cabin': cabin_encoded, 'Fare': fare, 'Pclass_High': p_high,
                      'Pclass_Mid': p_mid, 'Pclass_Low': p_low, 'Embarked': emb_encoded,
                      'SibSp': sibsp, 'Parch': parch, 'is_Female': is_female}
        input_df = pd.DataFrame([input_data])
        input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])
        input_df = input_df[columns]

        if predict_clicked:
            prediction = model.predict(input_df)
            survived = prediction[0] == 1
            status = "SURVIVED" if survived else "NOT SURVIVED"
            color = "#28a745" if survived else "#dc3545"
            bg = "#eafaf0" if survived else "#fdecee"
            icon = "" if survived else ""

            st.markdown(f"""
                <div class="result-box" style="background-color: {bg};">
                    <div class="result-icon">{icon}</div>
                    <h1 style="color: {color};">{status}</h1>
                    <p>Based on the passenger profile provided</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="placeholder">
                    <div class="emoji">🧭</div>
                    <p>Fill in the passenger details and click<br><b>Predict Survival</b> to see the result.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
            <div class="note">
                ℹ️ This prediction is based on a Decision Tree model trained on the Titanic dataset.
            </div>
            """, unsafe_allow_html=True)
