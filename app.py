import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("ulcer_model.pkl", "rb"))

st.title("Peptic Ulcer Risk Predictor")

age = st.slider("Age", 18, 85, 40)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
nsaid = st.selectbox("NSAID use", [0,1])
hp = st.selectbox("H.pylori", [0,1])
smoking = st.selectbox("Smoking", [0,1])
alcohol = st.selectbox("Alcohol", [0,1])
hb = st.slider("Hemoglobin", 90,160,130)

if st.button("Predict"):
    input_data = np.array([[age, sex, nsaid, hp, smoking, alcohol, hb]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write("Ulcer prediction:", prediction[0])
    st.write("Risk probability:", round(probability*100,2), "%")
