import os
import joblib
import pandas as pd
import streamlit as st

# Title of the Streamlit app
st.title("🚢 Titanic Survival Prediction")

# Load the trained model
model_path = "logistic_regression.pkl"

if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file '{model_path}' not found! Please ensure it is uploaded correctly.")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
