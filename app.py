import os
import joblib
import pandas as pd
import streamlit as st

# Title of the Streamlit app
st.title("🚢 Titanic Survival Prediction")

# Load the trained model
model_path = "logistic_regression.pkl"

if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file '{model_path}' not found! Please upload it.")
    st.stop()
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")

# Function to predict survival
def predict_survival(pclass, age, sibsp, parch, fare):
    input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare]],
                              columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
    prediction = model.predict(input_data)[0]
    return "Survived 🟢" if prediction == 1 else "Not Survived 🔴"

# User input form
st.header("Enter Passenger Details")
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", min_value=1, max_value=100, value=30)
sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.slider("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=50.0)

# Predict button
if st.button("Predict Survival"):
    result = predict_survival(pclass, age, sibsp, parch, fare)
    st.subheader(f"Prediction: {result}")
