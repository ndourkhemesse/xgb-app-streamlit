
import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgb_model.pkl")

st.title("Prédiction avec le modèle XGBoost")

Debit_sang_pompe = st.number_input("Débit sang pompe", value=300.0)
UF_H = st.number_input("UF_H", value=1.5)
Debit_eau_dialysat = st.number_input("Débit eau dialysat", value=500.0)
PA = st.number_input("PA", value=120.0)
PV = st.number_input("PV", value=40.0)
PTM = st.number_input("PTM", value=200.0)
Poul = st.number_input("Poul", value=70.0)
Anticoagulant_HS = st.selectbox("Anticoagulant HS", options=[0, 1])
Anticoagulant_HC = st.selectbox("Anticoagulant HC", options=[0, 1])
Anticoagulant_lovenox = st.selectbox("Anticoagulant Lovenox", options=[0, 1])
Periode_enc = st.number_input("Période enc", value=1, step=1)

if st.button("Prédire"):
    input_array = np.array([[ 
        Debit_sang_pompe, UF_H, Debit_eau_dialysat, PA,
        PV, PTM, Poul, Anticoagulant_HS, Anticoagulant_HC,
        Anticoagulant_lovenox, Periode_enc
    ]])
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)
    st.success(f"Prédiction : {int(prediction[0])}")
    st.write(f"Probabilités : {proba[0].tolist()}")
