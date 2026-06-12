import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==========================
# Load Files
# ==========================

model = load_model(
    "models/airline_rnn.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================
# UI
# ==========================

st.set_page_config(
    page_title="Airline Passenger Forecast",
    page_icon="✈️"
)

st.title("✈️ Airline Passenger Forecasting")

st.write(
    "Enter passenger counts for the previous 12 months"
)

values = []

for i in range(12):
    value = st.number_input(
        f"Month {i+1}",
        min_value=0,
        value=100
    )

    values.append(value)

# ==========================
# Prediction
# ==========================

if st.button("Predict Next Month"):

    arr = np.array(values).reshape(-1,1)

    arr = scaler.transform(arr)

    arr = arr.reshape(
        1,
        12,
        1
    )

    prediction = model.predict(arr)

    prediction = scaler.inverse_transform(
        prediction
    )

    st.success(
        f"Predicted Next Month Passengers: {int(prediction[0][0])}"
    )