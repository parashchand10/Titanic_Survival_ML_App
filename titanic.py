import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page layout to wide to give more space
st.set_page_config(layout="wide")

# Load assets
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('titanic_scaler.pkl', 'rb'))
columns = pickle.load(open('titanic_columns.pkl', 'rb'))

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Increase font size of sidebar headers */
    [data-testid="stSidebar"] h2 {
        font-size: 28px !important;
        color: #4A90E2;
    }
    /* Style the result container */
    .result-box {
        text-align: center;
        padding: 40px;
        border-radius: 15px;
        margin-top: 20px;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Passenger Details")
    age = st.slider("Age", 0, 80, 25)
    fare = st.number_input("Fare (Ticket Price)", 0, 512, 32)
    gender = st.selectbox("Gender", ["Male", "Female"])
    pclass = st.selectbox("Ticket Class", ["High (1st)", "Mid (2nd)", "Low (3rd)"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    has_cabin = st.radio("Has a Cabin?", ["Yes", "No"])

with col2:
    st.title("Titanic Survival Prediction")
    st.write("---")
    
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

    # Predict Button
    if st.button("Predict Survival", use_container_width=True, type="primary"):
        prediction = model.predict(input_df)
        status = "SURVIVED" if prediction[0] == 1 else "NOT SURVIVED"
        color = "#28a745" if prediction[0] == 1 else "#dc3545"
        
        st.markdown(f"""
            <div class="result-box">
                <h1 style="color: {color}; font-size: 60px; font-weight: 500;">{status}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    st.info("Note: This prediction is based on the Decision Tree model use for the Titanic Dataset notebook.")
