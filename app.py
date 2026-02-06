# Creating app (Streamlit), for easy deployment by app.py
# It will receive user inputs, show prediction to the user, predicting kind of transport etc.
# It will have option for regression and decision tree

# app.py - Professional Streamlit App with Animations

import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# --- Helper function to load Lottie animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load animation ---
lottie_transport = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_9xxyyb.json")

# --- App Config ---
st.set_page_config(page_title="AI Transport Predictor", page_icon="🚀", layout="centered")

# --- Header ---
st.title("🚀 AI Transport Mode Predictor - Dar es Salaam")
st.write("Smart predictions using **Decision Tree** or **Logistic Regression** models.")

# --- Show animation ---
if lottie_transport:
    st_lottie(lottie_transport, height=200, key="transport")

# --- Model Selection ---
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Decision Tree", "Logistic Regression"])

if model_choice == "Decision Tree":
    model, features = joblib.load("decision_tree_transport.pkl")
else:
    model, features = joblib.load("logistic_regression_transport.pkl")

# --- User Input ---
st.header("📝 Enter Your Trip Details")
distance = st.selectbox("Distance", ["short", "medium", "long"])
time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"])
weather = st.selectbox("Weather", ["sunny", "rainy", "cloudy"])
traffic = st.selectbox("Traffic", ["low", "medium", "high"])
budget = st.selectbox("Budget", ["low", "medium", "high"])

# --- Convert input to dataframe ---
input_data = pd.DataFrame({
    "Distance": [distance],
    "TimeOfDay": [time_of_day],
    "Weather": [weather],
    "Traffic": [traffic],
    "Budget": [budget]
})

# --- Encode input ---
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# --- Prediction ---
prediction = model.predict(input_encoded)[0]

# --- Display Result ---
st.success(f"🎯 Predicted Transport Mode: **{prediction}**")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by whildady | Powered by Streamlit")

